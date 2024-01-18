import os
import argparse
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BartForConditionalGeneration
)
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import torchmetrics

from data_utils import *
from constants import *
from utils import *



def p_r_f(prediction, labels):
    precision = precision_score(labels, prediction, average='binary', pos_label=1, zero_division=0)
    recall = recall_score(labels, prediction, average='binary', pos_label=1, zero_division=0)
    f1 = f1_score(labels, prediction, average='binary', pos_label=1, zero_division=0)
    return precision, recall, f1


def bleu_rouge(generation, references, split):
    def cn_normalizer(text):
        return text
    for i, sent in enumerate(generation):
        generation[i] = ' '.join(c for c in sent)
    for i, sent in enumerate(references):
        references[i] = ' '.join(c for c in sent)

    rouge = torchmetrics.functional.rouge_score(generation,
                                                references,
                                                rouge_keys=('rouge1', 'rouge2', 'rougeL'),
                                                normalizer=cn_normalizer)
    bleu1 = torchmetrics.functional.bleu_score(generation, references, n_gram=1)
    bleu2 = torchmetrics.functional.bleu_score(generation, references, n_gram=2)
    bleu4 = torchmetrics.functional.bleu_score(generation, references, n_gram=4)
    rouge1 = rouge['rouge1_fmeasure']
    rouge2 = rouge['rouge2_fmeasure']
    rougel = rouge['rougeL_fmeasure']
    print(f"({split})  bleu1:{bleu1:.3f}  bleu2:{bleu2:.3f}  bleu4:{bleu4:.3f}  rouge1:{rouge1:.3f}  rouge2:{rouge2:.3f}  rougel:{rougel:.3f}")
    return bleu1, bleu2, bleu4, rouge1, rouge2, rougel


def get_checker_prediction(batch_input, model):
    model.eval()
    with torch.no_grad():
        output_dict = getattr(model, 'module', model)(**batch_input, return_dict=True)
        logits = output_dict.logits
        prediction = logits.max(-1)[1].tolist()
    return prediction


def test_checker_results(data, model, tokenizer, dataset_name, device, args, flog=None):
    testset = ConsistencyDataset(data, tokenizer, args.task, device, with_explanation=args.with_explanation) 

    test_loader = DataLoader(testset,
                             shuffle=False,
                             batch_size=16,
                             num_workers=0,
                             collate_fn=testset.collate_fn)
    pred_all, gold_all = [], []
    results = []
    for batch_input, batch in test_loader:
        prediction = get_checker_prediction(batch_input, model)
        pred_all.extend(prediction)
        gold_all.extend(batch_input['labels'].view(-1).tolist())
        for item, pred in zip(batch, prediction):
            item['prediction'] = 'consistent' if pred == 0 else 'inconsistent'
        results.extend(batch)

    precision, recall, f1 = p_r_f(pred_all, gold_all)
    flog.info(f"[{dataset_name} test] pos_precision: {precision:.3f} pos_recall: {recall:.3f} pos_f1: {f1:.3f}")

    with open(args.save_dir + os.sep + f"{dataset_name}_test_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def get_gold_strings(batch_input, tokenizer):
    pad_token_id = tokenizer.pad_token_id
    labels = batch_input['labels']
    labels[labels == -100] = pad_token_id
    gold_strings = tokenizer.batch_decode(labels, skip_special_tokens=True)
    gold_strings = [sent for sent in gold_strings]
    return gold_strings


def get_resolver_prediction(batch_input, model, tokenizer):
    pad_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id
    labels = batch_input['labels']
    labels[labels == -100] = pad_token_id
    single_model =  getattr(model, 'module', model)
    single_model.eval()
    with torch.no_grad():
        output_dict = single_model(**batch_input, return_dict=True)
        loss = output_dict.loss.detach()
        logits = output_dict.logits.detach()
        preds = torch.argmax(logits, dim=-1)
        f1 = torchmetrics.functional.f1_score(preds,  # prediction of last token is not included
                                              labels,
                                              mdmc_average='global',
                                              ignore_index=pad_token_id)
        perplexity = torchmetrics.functional.perplexity(logits,
                                                        labels,
                                                        ignore_index=pad_token_id)
        return preds.tolist(), loss, f1, perplexity


def get_resolver_generation(batch_input, model, tokenizer):
    pad_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id
    single_model =  getattr(model, 'module', model)
    generation = single_model.generate(input_ids=batch_input['input_ids'],
                                       attention_mask=batch_input['attention_mask'],
                                       do_sample=True,
                                       pad_token_id=pad_token_id,
                                       eos_token_id=sep_token_id,
                                       top_p=0.9,
                                       max_new_tokens=50)
    generated_strings = tokenizer.batch_decode(generation, skip_special_tokens=True)
    generated_strings = [sent for sent in generated_strings]
    return generated_strings


def test_resolver_results(data, model, tokenizer, dataset_name, device, args, flog=None):
    testset = ConsistencyDataset(data, tokenizer, args.task, device, with_explanation=args.with_explanation)
    test_loader = DataLoader(
        testset,
        shuffle=False,
        batch_size=16,
        num_workers=0,
        collate_fn=testset.collate_fn
    )
    gen_all, gold_all = [], []
    results = []

    gen_all, gold_all = [], []
    for batch_input, batch in test_loader:
        generate_strings = get_resolver_generation(batch_input, model, tokenizer)
        gold_strings = get_gold_strings(batch_input, tokenizer)
        gen_all.extend(generate_strings) 
        gold_all.extend(gold_strings)
        for item, gen in zip(batch, generate_strings):
            item['generation'] = gen.replace(' ', '')
        results.extend(batch)

    bleu1, bleu2, bleu4, rouge1, rouge2, rougel = bleu_rouge(gen_all, gold_all, 'test')
    flog.info(f"(test) ({dataset_name}) bleu1:{bleu1:.3f} bleu2:{bleu2:.3f} bleu4:{bleu4:.3f} rouge1:{rouge1:.3f} rouge2:{rouge2:.3f} rougel:{rougel:.3f}")

    with open(args.save_dir + os.sep + f"{dataset_name}_test_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def random_results(data, dataset_name):
    labels = [0 if item['label'] == 'consistent' else 1 for item in data]
    prediction = [random.randint(0, 1) for _ in data] 
    precision, recall, f1 = p_r_f(prediction, labels)
    print(f"[{dataset_name} random test] pos_precision: {precision:.3f} pos_recall: {recall:.3f} pos_f1: {f1:.3f}")


def allpos_results(data, dataset_name):
    labels = [0 if item['label'] == 'consistent' else 1 for item in data]
    prediction = [1 for _ in data] 
    precision, recall, f1 = p_r_f(prediction, labels)
    print(f"[{dataset_name} random test] pos_precision: {precision:.3f} pos_recall: {recall:.3f} pos_f1: {f1:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=str2bool, default='y', help='debug model')
    parser.add_argument('--task', type=str, choices=[CHECK_TURN, CHECK_DIAG, RESOLVE_TURN, RESOLVE_DIAG], default=CHECK_TURN)
    parser.add_argument('--arch', type=str, required=True, help='Architechture of model.')
    parser.add_argument('--save_dir', type=str, default='exp/checker' + os.sep + current_time())
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--with_explanation', type=str2bool, default='false',
                        help='Whether to add explanation when doing specific tasks.')
    args = parser.parse_args()

    args.save_dir = ROOT + os.sep + args.save_dir
    args.model_dir = ROOT + os.sep + args.model_dir

    flog = Logger()
    flog.add_console_handler()
    flog.add_file_handler(args.save_dir + os.sep + 'test.log')

    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model_class = ARCH_MAPPING[args.arch]
    model = model_class.from_pretrained(args.model_dir).to(current_device())

    flog.info('')
    flog.info(f"task: {args.task}")

    if 'check' in args.task:
        _, _, test = create_tcon_data(args.task, flog)
        if args.debug:
            test = test[0:50]
        test_checker_results(test, model, tokenizer, 'tcon', current_device(), args, flog=flog)

        _, _, test = create_stance_data(args.task, flog)
        if args.debug:
            test = test[0:50]
        test_checker_results(test, model, tokenizer, 'stance', current_device(), args, flog=flog)

        _, _, test = create_cdconv_data(args.task, flog)
        if args.debug:
            test = test[0:50]
        test_checker_results(test, model, tokenizer, 'cdconv', current_device(), args, flog=flog)

        _, _, test = create_ocnli_data(args.task, flog)
        if args.debug:
            test = test[0:50]
        test_checker_results(test, model, tokenizer, 'ocnli', current_device(), args, flog=flog)
    else:
        _, _, test = create_tcon_data(args.task, flog)
        if args.debug:
            test = test[0:50]
        test_resolver_results(test, model, tokenizer, 'tcon', current_device(), args, flog=flog)
