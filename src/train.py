from copy import deepcopy

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import transformers
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, f1_score
transformers.logging.set_verbosity_error()

from utils import *
from constants import *
from test_datasets import *
from data_utils import *

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

class Trainer:

    def __init__(self, cmd, data_loaders, samplers, model, logger, tokenizer):
        self.cmd = cmd
        self.train_loader = data_loaders['train']
        self.valid_loader = data_loaders['valid']
        self.test_loader = data_loaders['test']
        self.train_sampler = samplers['train']
        self.model = model
        self.logger = logger
        self.tokenizer = tokenizer
        self.save_dir = cmd.save_dir
        self.evaluator = Evaluator(self.valid_loader, self.model, self.logger, tokenizer, split='valid')
        self.test_evaluator = Evaluator(self.test_loader, self.model, self.logger, tokenizer, split='test')
        self.task = cmd.task
        self.model_sig = cmd.model_sig

    def init_optimizer_scheduler(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': self.cmd.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0}
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.cmd.learning_rate)

        train_steps_per_epoch = len(self.train_loader)
        train_steps_total = self.cmd.epochs * train_steps_per_epoch
        self.logger.info(f'train_step_per_epoch: {train_steps_per_epoch}')
        self.logger.info(f'epoch: {self.cmd.epochs}')
        self.logger.info(f'total step: {train_steps_total}')
        self.scheduler = None
        if self.cmd.scheduler == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.cmd.warmup_steps,
                num_training_steps=train_steps_total
            )
        # Scaler for Fp 16 training
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cmd.fp16)

    def train_step(self, batch_idx, batch_input):
        with torch.cuda.amp.autocast(enabled=self.cmd.fp16):
            # loss = self.model(**dict((k, batch_data[k]) for k in train_keys))
            loss = self.model(**batch_input, return_dict=True).loss
        self.scaler.scale(loss).backward()

        if batch_idx % self.cmd.grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(), max_norm=self.cmd.clip_grad)
            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()
        return loss.item()

    def train_eval_per_epoch(self):
        global_step = 0
        best_valid_metric = 0.
        patience_counter = 0
        for epoch in range(1, self.cmd.epochs + 1):
            self.model.train()
            epoch_loss = 0.
            for batch_idx, (batch_input, batch) in enumerate(self.train_loader):
                global_step += 1
                loss = self.train_step(batch_idx, batch_input)
                if writer:
                    writer.add_scalar('training/lr', self.scheduler.get_last_lr()[0], global_step)
                    writer.add_scalar('training/batch_loss', loss, global_step)
                epoch_loss += loss

            epoch_loss = sum_distributed_scalar(epoch_loss, local_rank, current_device())
            self.logger.info('')
            self.logger.info(f'epoch {epoch}/{self.cmd.epochs} loss: {epoch_loss}')

            if 'check' in self.task:
                metric = self.evaluator.evaluate_checker(global_step)
            elif 'resolve' in self.task:
                metric = self.evaluator.evaluate_resolver(global_step)

            if metric > best_valid_metric:
                self.logger.info("Found new best metric")
                best_valid_metric = metric
                if local_rank in [-1, 0]:
                    # Save best model (transformers version) 
                    self.tokenizer.save_vocabulary(self.save_dir)
                    getattr(self.model, 'module', self.model).save_pretrained(self.save_dir)

                patience_counter = 0
            else:
                patience_counter += 1
            # self.test_evaluator.evaluate(global_step)
            if patience_counter >= self.cmd.patience_num and epoch >= self.cmd.min_epoch:
                logger.info('Final test set performance: ')
                # Load best model (transformers version) 
                best_model = model_class.from_pretrained(self.save_dir).to(current_device())
                if hasattr(self.model, 'module'):
                    self.model.module = best_model
                else:
                    self.model = best_model

                if 'check' in self.task:
                    self.test_evaluator.evaluate_checker(global_step, log_metrics=True, hparams=self.cmd.__dict__)
                elif 'resolve' in self.task:
                    self.test_evaluator.evaluate_resolver(global_step, log_metrics=True, hparams=self.cmd.__dict__)
                break


class Evaluator:

    def __init__(self, data_loader, model, logger, tokenizer, split=None):
        self.data_loader = data_loader
        self.model = model
        self.logger = logger
        self.tokenizer = tokenizer
        self.split = split # 'test' or 'valid'

    def evaluate_checker(self, global_step, log_metrics=False, hparams=None):
        pred_all, gold_all = [], []
        for batch_input, batch in self.data_loader:
            prediction = get_checker_prediction(batch_input, self.model)
            pred_all.extend(prediction)
            gold_all.extend(batch_input['labels'].view(-1).tolist())
        
        precision = precision_score(gold_all, pred_all, average='binary', pos_label=1, zero_division=0)
        recall = recall_score(gold_all, pred_all, average='binary', pos_label=1, zero_division=0)
        f1 = f1_score(gold_all, pred_all, average='binary', pos_label=1, zero_division=0)
        self.logger.info(
            f"({self.split}) precision: {precision:.3f} recall: {recall:.3f} f1: {f1:.3f}")
        if writer:
            if log_metrics:
                metrics = {'precision': precision, 'recall': recall, 'f1': f1}
                writer.add_hparams(hparams, metrics, run_name='res')
            else:
                writer.add_scalar(f'{self.split}/precision', precision, global_step)
                writer.add_scalar(f'{self.split}/recall', recall, global_step)
                writer.add_scalar(f'{self.split}/f1', f1, global_step)
        return f1

    def evaluate_resolver(self, global_step, log_metrics=False, hparams=None):
        gen_all, gold_all = [], []
        for batch_input, batch in self.data_loader:
            generate_strings = get_resolver_generation(batch_input, self.model, self.tokenizer)
            gold_strings = get_gold_strings(batch_input, self.tokenizer)
            gen_all.extend(generate_strings) 
            gold_all.extend(gold_strings)
        
        bleu1, bleu2, bleu4, rouge1, rouge2, rougel = bleu_rouge(gen_all, gold_all, self.split)
        self.logger.info(
            f"({self.split}) bleu1:{bleu1} bleu2:{bleu2} bleu4:{bleu4} rouge1:{rouge1} rouge2:{rouge2} rougel:{rougel}")

        if writer:
            if log_metrics:
                metrics = {'bleu1': bleu1, 'bleu2': bleu2, 'bleu4': bleu4, 'rouge1': rouge1, 'rouge2': rouge2, 'rougel': rougel}
                writer.add_hparams(hparams, metrics, run_name='res')
            else:
                writer.add_scalar(f'{self.split}/bleu1', bleu1, global_step)
                writer.add_scalar(f'{self.split}/bleu2', bleu2, global_step)
                writer.add_scalar(f'{self.split}/bleu4', bleu4, global_step)
                writer.add_scalar(f'{self.split}/rouge1', rouge1, global_step)
                writer.add_scalar(f'{self.split}/rouge2', rouge2, global_step)
                writer.add_scalar(f'{self.split}/rougel', rougel, global_step)

        return bleu4

class Command:

    def init_from_argparse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--debug', type=str2bool, default='y', help='debug model')
        # data parameters
        parser.add_argument('--dataset', type=str, default='both', choices=['both', 'lccc', 'naturalconv'])
        parser.add_argument('--add_tcon', type=str2bool, default='true')
        parser.add_argument('--add_cdconv', type=str2bool, default='false')
        parser.add_argument('--add_stance', type=str2bool, default='false')
        parser.add_argument('--add_ocnli', type=str2bool, default='false')
        parser.add_argument('--add_cmnli', type=str2bool, default='false')

        # size of extra datasets compared to tcon. 1.0 means the same size as tcon. -1 means the original size of
        # the datasets. can be any other float
        parser.add_argument('--size_as_of_tcon', default=-1.0, type=float)
        parser.add_argument('--save_dir', type=str, default='exp/checker' + os.sep + current_time())
        parser.add_argument('--plms_dir', type=str, default='plms')
        parser.add_argument('--ckpt_dir', type=str, default='')

        # model parameters
        parser.add_argument('--arch', type=str, required=True,
                            help='Architechture of model.')
        parser.add_argument('--model_sig', type=str, required=True,
                            help='Name of transformer model, namely the name of the model directory.')
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--max_length', type=int, default=512)
        parser.add_argument('--with_explanation', type=str2bool, default='false',
                            help='Whether to add explanation when doing specific tasks.')

        # training parameters
        parser.add_argument('--task', choices=[CHECK_TURN, CHECK_DIAG, RESOLVE_TURN, RESOLVE_DIAG], default=CHECK_TURN)
        parser.add_argument('--fp16', type=str2bool, default='n')
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--scheduler', type=str, default='linear')
        parser.add_argument('--min_epoch', type=int, default=0)
        parser.add_argument('--learning_rate', type=float, default=1e-5)
        parser.add_argument('--grad_accum_steps', type=int, default=1)
        parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warm up ratio")
        parser.add_argument('--clip_grad', type=float, default=5)
        parser.add_argument('--patience_num', type=int, default=3)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--num_workers', type=int, default=10)
        parser.add_argument('--seed', type=int, default=1111)
        parser.add_argument('--use_tensorboard', type=str2bool, default='y')
        parser.add_argument('--balanced_sampling', type=str2bool, default='n')
        parser.add_argument('--validate_on_tcon', type=str2bool, default='n')
        self.args = parser.parse_args()

        self.args.plms_dir = ROOT + os.sep + self.args.plms_dir
        self.args.save_dir = ROOT + os.sep + self.args.save_dir

        args_dict = vars(self.args)
        self._set_attr_from_dict(args_dict)

    def _set_attr_from_dict(self, _dict):
        for k, v in _dict.items():
            setattr(self, k, v)
        del self.args

    def get_cmd_repr(self):
        ret = ''
        for k, v in vars(self.args).items():
            ret += f'{k}={v}\n'
        return ret


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    global local_rank
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_seed(2023)

    cmd = Command()
    cmd.init_from_argparse()

    # Tensorboard
    global writer
    if local_rank in [-1, 0] and cmd.use_tensorboard:
        writer = SummaryWriter(cmd.save_dir)
    else:
        writer = None

    # Logger
    global logger
    logger = Logger(logging.INFO) if local_rank in [-1, 0] else Logger(logging.WARN)
    logger.add_console_handler()
    # Make log dir
    if local_rank in [-1, 0]:
        if not os.path.exists(cmd.save_dir):
            os.makedirs(cmd.save_dir)
            logger.add_file_handler(cmd.save_dir + os.sep + 'train.log')
            logger.info(cmd.get_cmd_repr())
        else:
            logger.add_file_handler(cmd.save_dir + os.sep + 'train.log')
        # save cmd
        torch.save(cmd, cmd.save_dir + os.sep + 'cmd.bin')
    logger.warning(f"Running process {local_rank}")

    # Distributed
    init_distributed_if_needed(local_rank)

    # Create data loaders
    model_dir = cmd.plms_dir + os.sep + cmd.model_sig
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    train_all, valid_all, test_all = [], [], []
    if cmd.add_tcon:
        train, valid, test = create_tcon_data(cmd.task, logger)
        train_all.extend(train)
        valid_all.extend(valid)
        test_all.extend(test)
    if cmd.add_cdconv:
        train, valid, test = create_cdconv_data(cmd.task, logger)
        train_all.extend(train)
        if not cmd.validate_on_tcon:
            valid_all.extend(valid)
        test_all.extend(test)
    if cmd.add_stance:
        train, valid, test = create_stance_data(cmd.task, logger)
        train_all.extend(train)
        if not cmd.validate_on_tcon:
            valid_all.extend(valid)
        test_all.extend(test)
    if cmd.add_ocnli:
        train, valid, test = create_ocnli_data(cmd.task, logger)
        train_all.extend(train)
        if not cmd.validate_on_tcon:
            valid_all.extend(valid)
        test_all.extend(test)
    if cmd.debug:
        train_all = train_all[:2000]
        valid_all = valid_all[:200]
        test_all = test_all[:200]
    trainset = ConsistencyDataset(train_all, tokenizer, cmd.task, current_device(), max_length=cmd.max_length, with_explanation=cmd.with_explanation)
    validset = ConsistencyDataset(valid_all, tokenizer, cmd.task, current_device(), max_length=cmd.max_length, with_explanation=cmd.with_explanation)
    testset = ConsistencyDataset(test_all, tokenizer, cmd.task, current_device(), max_length=cmd.max_length, with_explanation=cmd.with_explanation)

    train_sampler = DistributedSampler(trainset, shuffle=True, drop_last=False) if local_rank != -1 else None 
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=cmd.batch_size,
                              num_workers=0,
                              collate_fn=trainset.collate_fn)
    valid_loader = DataLoader(validset,
                              shuffle=False,
                              batch_size=cmd.batch_size,
                              num_workers=0,
                              collate_fn=validset.collate_fn)
    test_loader = DataLoader(testset,
                             shuffle=False,
                             batch_size=cmd.batch_size,
                             num_workers=0,
                             collate_fn=testset.collate_fn)
    
    optimization_report(cmd, local_rank, logger, train_loader)

    if cmd.ckpt_dir:
        load_dir = ROOT + os.sep + cmd.ckpt_dir
        logger.info(f"Load checkpoint from {load_dir}")
    else:
        load_dir = model_dir
    
    global model_class
    model_class = ARCH_MAPPING[cmd.arch]
    model = model_class.from_pretrained(load_dir).to(current_device())

    # if cmd.ckpt_dir:
    #     ckpt_path = ROOT + os.sep + cmd.ckpt_dir + os.sep + 'model.pt'
    #     model.load_state_dict(torch.load(ckpt_path, map_location=current_device()))
    #     logger.info(f"Load checkpoint from {ckpt_path}")
    
    # Distribute model to procs
    if local_rank != -1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True)

    trainer = Trainer(cmd,
                      {'train': train_loader, 'valid': valid_loader, 'test': test_loader},
                      {'train': train_sampler, 'valid': None, 'test': None},
                      model,
                      logger,
                      tokenizer)
    trainer.init_optimizer_scheduler()
    trainer.train_eval_per_epoch()

    if 'check' in cmd.task:
        _, _, test = create_tcon_data(cmd.task, logger)
        if cmd.debug:
            test = test[0:50]
        test_checker_results(test, model, tokenizer, 'tcon', current_device(), cmd, flog=logger)

        _, _, test = create_stance_data(cmd.task, logger)
        if cmd.debug:
            test = test[0:50]
        test_checker_results(test, model, tokenizer, 'stance', current_device(), cmd, flog=logger)

        _, _, test = create_cdconv_data(cmd.task, logger)
        if cmd.debug:
            test = test[0:50]
        test_checker_results(test, model, tokenizer, 'cdconv', current_device(), cmd, flog=logger)

        _, _, test = create_ocnli_data(cmd.task, logger)
        if cmd.debug:
            test = test[0:50]
        test_checker_results(test, model, tokenizer, 'ocnli', current_device(), cmd, flog=logger)
    else:
        _, _, test = create_tcon_data(cmd.task, logger)
        if cmd.debug:
            test = test[0:50]
        test_resolver_results(test, model, tokenizer, 'tcon', current_device(), cmd, flog=logger)
