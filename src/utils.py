import random
import logging

import torch
import numpy as np
import torch.distributed as dist

from constants import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def current_time():
    from datetime import datetime
    return datetime.now().strftime('%b%d_%H-%M-%S')


def flat_array(array):
    ret = []

    def flat(item):
        if not isinstance(item, list):
            ret.append(item)
            return
        for x in item:
            flat(x)
    flat(array)
    return ret


def tokenize(obj, tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o, tokenizer)) for n, o in obj.items())
    return list(tokenize(o, tokenizer) for o in obj)


def combine_one_seq(seq, max_len=512, cls_token_id=101, sep_token_id=102):  # [CLS] + seq + [SEP]
    available_len = max_len - 2
    assert available_len >= len(seq), "response can not be truncated"
    seq_tmp = seq[:available_len] if len(seq) > available_len else seq
    input_id = [cls_token_id] + seq_tmp + [sep_token_id]
    seg_id = [0] * len(input_id)
    seq_mask = [0] + [1] * len(seq_tmp) + [0]
    return input_id, seg_id, seq_mask


def combine_two_seq(seq_a, seq_b, max_len=512, cls_token_id=101, sep_token_id=102):
    '''
    [CLS] + seq_a + [SEP] + seq_b + [SEP]
    '''
    available_len = max_len - 3
    if len(seq_b) >= available_len:
        seq_b_tmp = seq_b[:available_len]  # drop the tail
        seq_a_tmp = []
    else:
        seq_b_tmp = seq_b
        seq_a_available_len = available_len - len(seq_b)
        if seq_a_available_len >= len(seq_a):
            seq_a_tmp = seq_a
        else:
            seq_a_tmp = seq_a[len(seq_a) - seq_a_available_len:]  # drop the front by character
    input_id = [cls_token_id] + seq_a_tmp + \
                [sep_token_id] + seq_b_tmp + \
                [sep_token_id]
    seg_id = [0] * (1 + len(seq_a_tmp)) + [1] * (2 + len(seq_b_tmp))
    seq_b_mask = [0] * (2 + len(seq_a_tmp)) + [1] * len(seq_b_tmp) + [0]
    return input_id, seg_id, seq_b_mask, len(seq_a_tmp)


def pad_id_seq(seq, pad_value=0):
    batch_size, max_len = len(seq), max([len(ids) for ids in seq])
    paded_id_seq = torch.empty((batch_size, max_len))
    paded_id_seq.fill_(pad_value)
    for i in range(batch_size):
        paded_id_seq[i][:len(seq[i])] = torch.tensor(seq[i])
    return paded_id_seq


def listRightIndex(alist, value):
    return len(alist) - alist[-1::-1].index(value) - 1


def batcher(instances, batch_size):
    ret = []
    i = 0
    while i < len(instances):
        j = i
        while j < len(instances) and (j - i) < batch_size:
            j += 1
        ret.append(instances[i: j])
        i = j
    return ret


def is_chinese(c):
    return '\u4e00' <= c <= '\u9fa5'


def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def en_ch_split(sent):
    result = []
    english_token = []
    tokens = list(sent)
    for i in range(len(tokens)):
        if is_all_chinese(tokens[i]):
            if len(english_token) > 0:
                result.append("".join(english_token))
                english_token = []
            result.append(tokens[i])
        else:
            english_token.append(tokens[i])
    if len(english_token) > 0:
        result.append("".join(english_token))
    return result


class Logger:

    def __init__(self, level=logging.INFO):
        self.logger = logging.getLogger()
        self.logger.setLevel(level)

    def add_file_handler(self, path):
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        self.logger.addHandler(file_handler)

    def add_console_handler(self):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(stream_handler)

    def info(self, msg):
        self.logger.info(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Adapted From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


import argparse
def str2bool(v):
    if v.lower() in ('y', 'yes', 't', 'true', '1', 'False'):
        return True
    elif v.lower() in ('n', 'no', 'f', 'false', '0', 'True'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def init_distributed_if_needed(local_rank):
    if local_rank != -1:
        torch.cuda.set_device(local_rank)

        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
        # 'env://' if officially recommended. 
        # (This blocks until all processes have joined.)

import os
import json
def merge_json_files(file_list, target_file):
    data = []
    for file in file_list:
        with open(file, 'r', encoding='utf-8') as f:
            local_data = json.load(f)
            data.extend(local_data)
        os.remove(file) 
    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def sum_distributed_scalar(scalar, local_rank, device):
    """
    Sum a scalar over the nodes if we are in distributed training.
    We use this for distributed evaluation.
    """
    if local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=device)
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def span_aware_en_ch_split(sent, span_boundaries):
    '''
    span boundaries should be in order
    '''
    indexes = [x for span in span_boundaries for x in span]
    if len(indexes) == 0 or indexes[-1] < len(sent):
        indexes.append(len(sent))
    st = 0
    words = []
    word_spans = []
    for ed in indexes:
        span = sent[st: ed]
        span_words = en_ch_split(span)
        words.extend(span_words)
        word_spans.append(len(words))
        st = ed
    word_spans = word_spans if len(word_spans) % 2 == 0 else word_spans[:-1]  # remove dummy index
    if word_spans:
        word_spans = [(word_spans[i], word_spans[i + 1]) for i in range(0, len(word_spans), 2)]
    return words, word_spans


def tokenize_with_alignment(words, tokenizer):
    tokens = []
    token2word, word2token = [], []
    for word_idx, word in enumerate(words):
        word_tokens = tokenizer.tokenize(word)
        st = len(tokens)
        tokens.extend(word_tokens)
        ed = len(tokens)
        word2token.append((st, ed))
        token2word.extend([word_idx] * len(word_tokens))
    return tokens, Alignment(word2token, token2word)


from typing import List, Tuple
class Alignment:
    '''
    alignment before and after tokenization
    word must be splited to multiple (>=1) tokens
    '''
    def __init__(
        self,
        word2token: List[Tuple[int, int]],
        token2word: List[int],
        words: List[str] = None,
        tokens: List[str] = None
    ) -> None:
        self.word2token = word2token
        self.token2word = token2word
        self.words = words
        self.tokens = tokens

    def word_index_to_token_index(self, index):
        return self.word2token[index]

    def token_index_to_word_index(self, index):
        return self.token2word[index]

    def word_spans_to_token_spans(self, word_spans):
        ret = []
        for st, ed in word_spans:
            span_token_st, _ = self.word_index_to_token_index(st)
            _, span_token_ed = self.word_index_to_token_index(ed) if ed < len(self.word2token) else \
                len(self.token2word), len(self.token2word)
            ret.append((span_token_st, span_token_ed))
        return ret

    def token_spans_to_word_spans(self, token_spans):
        '''
        if a token is in the span, the whole word is included in the span
        '''
        ret = []
        for st, ed in token_spans:
            span_word_ed = self.token_index_to_word_index(ed - 1) + 1
            # not used: if the tokens of a word are included in two separate spans, we include the word to the former one
            ret.append((self.token_index_to_word_index(st), span_word_ed))
        return ret

    def get_word_tokenized_starts(self):
        return [st for st, _ in self.word2token]

            
def label_to_spans(label: List[int]):
    '''
    binary tagging scheme: label seq -> span tuples 
    '''
    ret = []
    i = 0
    while i < len(label):
        if label[i] == 1:
            j = i + 1
            while j < len(label) and label[j] == 1:
                j += 1
            ret.append((i, j))
            i = j
        else:
            i += 1
    return ret


def bio_label_to_spans(label: List[int]):
    '''
    bio tagging scheme: label seq -> span tuples 
    '''
    ret = []
    i = 0
    while i < len(label):
        if bio_id2label[label[i]] == 'B-USF':
            j = i + 1
            while j < len(label) and bio_id2label[label[j]] == 'I-USF':
                j += 1
            ret.append((i, j))
            i = j
        else:
            i += 1
    return ret


def bioes_label_to_spans(label: List[int]):
    '''
    bioes tagging scheme: label seq -> span tuples 
    '''
    ret = []
    i = 0
    while i < len(label):
        if bioes_id2label[label[i]] == 'B-USF':
            j = i + 1
            while j < len(label) and bioes_id2label[label[j]] != 'E-USF':
                j += 1
            ret.append((i, j + 1))
            i = j + 1
        elif bioes_id2label[label[i]] == 'S-USF':
            ret.append((i, i + 1))
            i += 1
        else:
            i += 1
    return ret


def match_metric_sets(pred_set, gold_set):
    tp = len(pred_set.intersection(gold_set))
    tp_fp = len(pred_set)
    tp_fn = len(gold_set)

    return tp, tp_fp, tp_fn


def optimization_report(cmd, local_rank, logger, train_loader):
    train_step_per_epoch = len(train_loader)
    logger.info(f"#train step per epoch per proc: {train_step_per_epoch}")
    total_train_step = train_step_per_epoch * cmd.epochs
    logger.info(f"#total train step: {total_train_step}")
    cmd.warmup_steps = int(total_train_step * cmd.warmup_ratio)
    logger.info(f"#warmup step: {cmd.warmup_steps}")
    world_size = 1 if not local_rank != 1 else dist.get_world_size()
    logger.info(f"world size: {world_size}")
    logger.info(f"world batch size: {cmd.batch_size * world_size}")
    logger.info(f"update batch size: {cmd.batch_size * world_size * cmd.grad_accum_steps}")


def move_tensors_to(data: dict, device):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.to(device)


def current_device():
    if torch.cuda.is_available():
        # equivalent to torch.device(f"cuda:{cmd.local_rank}")
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device('cpu')


def save_model_dist_pt_version(model, save_dir):
    # Save best model (pt version)
    torch.save(getattr(model, 'module', model).state_dict(),
        save_dir + os.sep + 'model.pt')

def load_model_dist_pt_version(model, save_dir, device):
    # Load best model (pt version)
    getattr(model, 'module', model).load_state_dict(
        torch.load(save_dir + os.sep + 'model.pt', map_location=device))

                    