import os
import json
import random
from copy import deepcopy
from torch.utils.data import Dataset

from constants import *
from utils import *


# entry read func
def read_single_file(filename):
    with open(filename, 'r', encoding='utf8') as fh:
        entries = json.load(fh)
    return entries


def read_stance(filename):
    identifiers = ['A1']
    with open(filename, 'r', encoding='utf8') as fh:
        entries = []
        for line in fh:
            items = line.strip().split('\t')
            label = '1' if items[-1] == 'Against' else '0'
            conv = items[:-1]
            reply = conv[-1]
            context = [conv[-2]]
            new_entry = {}
            if label == '1':
                new_entry["inconsistent_response"] = reply
                new_entry["inconsistent_source"] = 'A1'
            else:
                new_entry["inconsistent_response"] = reply
                new_entry["inconsistent_source"] = ''
            new_entry["dialogue_context"] = []
            for index, sent in enumerate(context):
                new_entry['dialogue_context'].append({'sent': sent, 'identifier': identifiers[index]})
            new_entry['dialogue_identifier'] = 'stance'
            entries.append(new_entry)
    return entries


def read_cdconv(filename):
    identifiers = ['A1', 'B1', 'A2']
    with open(filename, 'r', encoding='utf8') as fh:
        entries = []
        for line in fh:
            items = line.strip().split('\t')
            label = items[-1]
            conv = items[:-1]
            reply = conv[-1]
            context = conv[:-1]
            new_entry = {}
            if label == '1':
                new_entry["inconsistent_response"] = reply
                new_entry["inconsistent_source"] = 'B1'
            else:
                new_entry["inconsistent_response"] = reply
                new_entry["inconsistent_source"] = ''
            new_entry["dialogue_context"] = []
            for index, sent in enumerate(context):
                new_entry['dialogue_context'].append({'sent': sent, 'identifier': identifiers[index]})
            new_entry['dialogue_identifier'] = 'cdconv'
            entries.append(new_entry)
    return entries


def read_ocnli_or_cmnli_file(filename):
    identifiers = ['A1']
    with open(filename, 'r', encoding='utf8') as fh:
        entries = []
        for line in fh:
            items = json.loads(line)
            label = '1' if items['label'] == CNLI_LABEL_CONTRADICTION else '0'
            conv = [items['sentence1']]
            reply = items['sentence2']
            new_entry = {}
            if label == '1':
                new_entry["inconsistent_response"] = reply
                new_entry["inconsistent_source"] = 'A1'
            else:
                new_entry["inconsistent_response"] = reply
                new_entry["inconsistent_source"] = ''
            new_entry["dialogue_context"] = []
            for index, sent in enumerate(conv):
                new_entry['dialogue_context'].append({'sent': sent, 'identifier': identifiers[index]})
            new_entry['dialogue_identifier'] = 'ocnli'
            entries.append(new_entry)
    return entries



def next_turn_identifier(turn):
    speaker = turn[0] 
    id = turn[1]
    next_speaker = 'A' if speaker == 'B' else 'B'
    next_id = str(int(id) + 1) if next_speaker == 'A' else str(int(id))
    return next_speaker + next_id 


def create_instances_check_turn(data):
    instances = []
    info = {}
    for entry in data:
        gold_target = entry['inconsistent_source']
        if entry['dialogue_identifier'] in ['stance', 'ocnli']:
            # stance and ocnli are regarded as two utterances by the same speaker ---> A1, A2
            response_identifier = 'A2'
        else: 
            response_identifier = next_turn_identifier(entry['dialogue_context'][-1]['identifier'])
        for turn in entry['dialogue_context']:
            inst = deepcopy(entry)
            inst['task'] =CHECK_TURN 
            inst['sentence1'] = turn['sent']
            inst['sentence2'] = entry["inconsistent_response"]
            del inst['inconsistent_response']
            if turn['identifier'] == gold_target:
                # gold target utterance is labeled as insconsist
                inst['label'] = 'inconsistent'
                instances.append(inst)
            elif turn['identifier'][0] == response_identifier[0]:
                # other utterances by the speker are labeled as consistent
                inst['label'] = 'consistent'
                instances.append(inst)
    info['#total'] = len(instances)
    info['#pos'] = sum(1 if inst['label'] == 'inconsistent' else 0 for inst in instances)
    info['#neg'] = sum(0 if inst['label'] == 'inconsistent' else 1 for inst in instances)
    info['pos_rate'] = f"{info['#pos'] / info['#total']:.4f}"
    return instances, info


def create_instances_check_diag(data):
    instances = [] 
    info = {}
    for entry in data:
        if not entry['inconsistent_source']:
            # inconsistent source is none means consistent
            prev_turns = [x['sent'] for x in entry['dialogue_context']]
            response = entry['inconsistent_response']
            entry_copy = deepcopy(entry)
            entry_copy['task'] = CHECK_DIAG 
            entry_copy['prev_turns'] = prev_turns
            entry_copy['response'] = response
            del entry_copy['inconsistent_response']
            entry_copy['label'] = 'consistent'
            instances.append(entry_copy)
        else:
            if entry['dialogue_identifier'] == 'cdconv':
                prev_turns = [x['sent'] for x in entry['dialogue_context']]
            else:
                prev_turns = [x['sent'] for x in entry['dialogue_context'][:-1]]
            response = entry['inconsistent_response']
            entry_copy = deepcopy(entry)
            entry_copy['task'] = CHECK_DIAG 
            entry_copy['prev_turns'] = prev_turns
            entry_copy['response'] = response
            entry_copy['label'] = 'inconsistent'
            del entry_copy['inconsistent_response']
            instances.append(entry_copy)
            
            if 'LCCC' in entry['dialogue_identifier'] or 'NaturalConv' in entry['dialogue_identifier']:
                response = entry['dialogue_context'][-1]['sent']
                entry_copy = deepcopy(entry)
                entry_copy['task'] = CHECK_DIAG 
                entry_copy['prev_turns'] = prev_turns
                entry_copy['response'] = response
                del entry_copy['inconsistent_response']
                entry_copy['label'] = 'consistent'
                instances.append(entry_copy)

    info['#total'] = len(instances)
    info['#pos'] = sum(1 if inst['label'] == 'inconsistent' else 0 for inst in instances)
    info['#neg'] = sum(0 if inst['label'] == 'inconsistent' else 1 for inst in instances)
    info['pos_rate'] = f"{info['#pos'] / info['#total']:.4f}"
    return instances, info


def create_instances_resolve_turn(data):
    instances = [] 
    info = {}
    for entry in data:
        gold_target = entry['inconsistent_source']
        for index, turn in enumerate(entry['dialogue_context']):
            if turn['identifier'] == gold_target:
                entry['task'] = RESOLVE_TURN
                entry['sentence1'] = turn['sent']
                entry['sentence2'] = entry['inconsistent_response']
                del entry['inconsistent_response']
                instances.append(entry)
                break
    info['#total'] = len(instances)
    return instances, info


def create_instances_resolve_diag(data):
    instances = [] 
    info = {}
    for entry in data:
        prev_turns = [x['sent'] for x in entry['dialogue_context']]
        response = entry['inconsistent_response']
        entry['task'] = RESOLVE_DIAG
        entry['prev_turns'] = prev_turns
        entry['response'] = response
        del entry['inconsistent_response']
        instances.append(entry)
    info['#total'] = len(instances)
    return instances, info


instance_generation_funcs = {CHECK_TURN: create_instances_check_turn,
                             CHECK_DIAG: create_instances_check_diag,
                             RESOLVE_TURN: create_instances_resolve_turn,
                             RESOLVE_DIAG: create_instances_resolve_diag}


def create_tcon_data(task, flog=None):
    create_instance_func = instance_generation_funcs[task]
    log = flog.info if flog is not None else print

    lccc_file = os.path.join(ROOT, 'data/cider/LCCC_consistency_.dataset.json')
    naturalconv_file = os.path.join(ROOT, 'data/cider/NaturalConv_consistency_.dataset.json')
    lccc_data = read_single_file(lccc_file)
    naturalconv_data = read_single_file(naturalconv_file)
    for key in lccc_data:
        lccc_data[key].extend(naturalconv_data[key])
    tcon_data = lccc_data
    tcon_instances = {}
    for split in ['train', 'validation', 'test']:
        tcon_instances[split], info = create_instance_func(tcon_data[split])
        log(f"[tcon {split}]" + str(info))
    return tcon_instances['train'], tcon_instances['validation'], tcon_instances['test']


def create_stance_data(task, flog=None):
    log = flog.info if flog is not None else print
    if 'resolve' in task:
        raise RuntimeError('stance not surporting resolving task')
    create_instance_func = instance_generation_funcs[task]
    randomizer = random.Random(42)
    stance_train_file = os.path.join(ROOT, "data/stance/train.txt")
    stance_test_file = os.path.join(ROOT, "data/stance/test.txt")

    stance_data = read_stance(stance_train_file)
    validation_index = set(randomizer.sample(range(len(stance_data)), int(len(stance_data) * 0.1)))
    train_data = [x for index, x in enumerate(stance_data) if index not in validation_index]
    train_instances, info = create_instance_func(train_data)
    log(f"[stance train]" + str(info))
    
    validation_data = [x for index, x in enumerate(stance_data) if index in validation_index]
    validation_instances, info = create_instance_func(validation_data)
    log(f"[stance validation]" + str(info))

    test_data = read_stance(stance_test_file)
    test_instances, info = create_instance_func(test_data)
    log(f"[stance test]" + str(info))

    return train_instances, validation_instances, test_instances


def create_cdconv_data(task, flog=None):
    log = flog.info if flog is not None else print
    if 'resolve' in task:
        raise RuntimeError('cdconv not surporting resolving task')
    create_instance_func = instance_generation_funcs[task]
    log = flog.info if flog is not None else print

    cdconv_train_file = os.path.join(ROOT, "data/cdconv/2class_train.tsv")
    cdconv_validation_file = os.path.join(ROOT, "data/cdconv/2class_dev.tsv")
    cdconv_test_file = os.path.join(ROOT, "data/cdconv/2class_test.tsv")

    train_data = read_cdconv(cdconv_train_file)
    train_instances, info = create_instance_func(train_data)
    log(f"[cdconv train]" + str(info))

    validation_data = read_cdconv(cdconv_validation_file)
    validation_instances, info = create_instance_func(validation_data)
    log(f"[cdconv validation]" + str(info))

    test_data = read_cdconv(cdconv_test_file)
    test_instances, info = create_instance_func(test_data)
    log(f"[cdconv test]" + str(info))

    return train_instances, validation_instances, test_instances


def create_ocnli_data(task, flog=None):
    log = flog.info if flog is not None else print
    if 'resolve' in task:
        raise RuntimeError('ocnli not surporting resolving task')
    create_instance_func = instance_generation_funcs[task]
    log = flog.info if flog is not None else print

    randomizer = random.Random(42)
    ocnli_train_file = os.path.join(ROOT, "data/ocnli/train.json")
    ocnli_validation_file = os.path.join(ROOT, "data/ocnli/dev.json")

    ocnli_data = read_ocnli_or_cmnli_file(ocnli_train_file)
    validation_index = set(randomizer.sample(range(len(ocnli_data)), int(len(ocnli_data) * 0.1)))
    train_data = [x for index, x in enumerate(ocnli_data) if index not in validation_index]
    train_instances, info = create_instance_func(train_data)
    log(f"[ocnli train]" + str(info))

    validation_data = [x for index, x in enumerate(ocnli_data) if index in validation_index]
    validation_instances, info = create_instance_func(validation_data)
    log(f"[ocnli validation]" + str(info))

    test_data = read_ocnli_or_cmnli_file(ocnli_validation_file)
    test_instances, info = create_instance_func(test_data)
    log(f"[ocnli test]" + str(info))

    return train_instances, validation_instances, test_instances


def check_turn_collate_fn(batch, tokenizer, device='cuda', max_length=512):
    xs = []
    ts = []

    for entry in batch:
        xs.append((entry['sentence1'], entry['sentence2']))
        ts.append(0 if entry['label'] == 'consistent' else 1)
        
    tokenized_batch = tokenizer(
        xs,
        max_length=max_length,
        truncation=True,
        padding="longest",
        return_tensors="pt"
    )

    tokenized_batch['labels'] = torch.tensor(ts)
    move_tensors_to(tokenized_batch, device) 
    return dict(tokenized_batch), batch


def check_diag_collate_fn(batch, tokenizer, device='cuda', max_length=512):
    xs = []
    ts = []

    for entry in batch:
        context = tokenizer.sep_token.join(entry['prev_turns'])
        xs.append((context, entry['response']))
        ts.append(0 if entry['label'] == 'consistent' else 1)

    tokenized_batch = tokenizer(
        xs,
        max_length=max_length,
        truncation=True,
        padding="longest",
        return_tensors="pt"
    )

    tokenized_batch['labels'] = torch.tensor(ts)
    move_tensors_to(tokenized_batch, device) 
    return dict(tokenized_batch), batch


def resolve_turn_collate_fn(batch, tokenizer, device, with_explanation=False, with_label=True):
    input_id_seq = []
    mask_id_seq = []
    decoder_input_id_seq = []
    decoder_mask_id_seq = []
    lm_label_seq = []
    for entry in batch:
        target_response = tokenize(entry['sentence1'], tokenizer)
        inconsistent_response = tokenize(entry['sentence2'], tokenizer)
        if not with_explanation:
            sequence1 = target_response
            sequence2 = inconsistent_response
        else:
            sequence1 = target_response + [tokenizer.sep_token_id] + inconsistent_response
            sequence2 = tokenize(entry['inconsistency'], tokenizer)
        input_id, _, _, _ = combine_two_seq(sequence1,
                                            sequence2,
                                            cls_token_id=tokenizer.cls_token_id,
                                            sep_token_id=tokenizer.sep_token_id)
        # encoder input
        input_id_seq.append(input_id) 
        mask_id_seq.append([1] * len(input_id))
        # decoder input
        decoder_input_id, lm_label = None, None
        if with_label:
            clarification_response = tokenize(entry['clarification_response'], tokenizer)
            decoder_input_id = [tokenizer.cls_token_id] + clarification_response
            lm_label = clarification_response + [tokenizer.sep_token_id]
            decoder_input_id_seq.append(decoder_input_id)
            decoder_mask_id_seq.append([1] * len(decoder_input_id))  # probably no use
            lm_label_seq.append(lm_label)

    pad = tokenizer.pad_token_id
    ret = {'input_ids': pad_id_seq(input_id_seq, pad).long(),
           'attention_mask': pad_id_seq(mask_id_seq, 0.).float(),
           'decoder_input_ids': pad_id_seq(decoder_input_id_seq, pad).long() if with_label else None,
           'decoder_attention_mask': pad_id_seq(decoder_mask_id_seq, 0.).float() if with_label else None,
           'labels': pad_id_seq(lm_label_seq, -100).long() if with_label else None}
    move_tensors_to(ret, device)
    return ret, batch


def resolve_diag_collate_fn(batch, tokenizer, device, with_explanation=False, with_label=True):
    input_id_seq = []
    mask_id_seq = []
    decoder_input_id_seq = []
    decoder_mask_id_seq = []
    lm_label_seq = []
    for entry in batch:
        response = tokenize(entry['inconsistent_source'], tokenizer)
        context = tokenize(tokenizer.sep_token.join(entry['prev_turns']), tokenizer)

        if not with_explanation:
            sequence1 = context
            sequence2 = response
        else:
            sequence1 = context + [tokenizer.sep_token_id] + response 
            sequence2 = tokenize(entry['inconsistency'], tokenizer)
        input_id, _, _, _ = combine_two_seq(sequence1,
                                            sequence2,
                                            cls_token_id=tokenizer.cls_token_id,
                                            sep_token_id=tokenizer.sep_token_id)
        # encoder input
        input_id_seq.append(input_id) 
        mask_id_seq.append([1] * len(input_id))
        # decoder input
        decoder_input_id, lm_label = None, None
        if with_label:
            clarification_response = tokenize(entry['clarification_response'], tokenizer)
            decoder_input_id = [tokenizer.cls_token_id] + clarification_response
            lm_label = clarification_response + [tokenizer.sep_token_id]
            decoder_input_id_seq.append(decoder_input_id)
            decoder_mask_id_seq.append([1] * len(decoder_input_id))  # probably no use
            lm_label_seq.append(lm_label)

    pad = tokenizer.pad_token_id
    ret = {'input_ids': pad_id_seq(input_id_seq, pad).long(),
           'attention_mask': pad_id_seq(mask_id_seq, 0.).float(),
           'decoder_input_ids': pad_id_seq(decoder_input_id_seq, pad).long() if with_label else None,
           'decoder_attention_mask': pad_id_seq(decoder_mask_id_seq, 0.).float() if with_label else None,
           'labels': pad_id_seq(lm_label_seq, -100).long() if with_label else None}
    move_tensors_to(ret, device)
    return ret, batch


class ConsistencyDataset(Dataset):

    def __init__(self, instances, tokenizer, task, device='cuda', max_length=512, with_explanation=False):
        self.instances = instances
        self.tokenizer = tokenizer
        self.task = task
        self.device = device
        self.max_length = max_length
        self.with_explanation = with_explanation

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return len(self.instances)
    
    def collate_fn(self, batch):
        if self.task == CHECK_TURN:
            return check_turn_collate_fn(batch, self.tokenizer, self.device, self.max_length) 
        elif self.task == CHECK_DIAG:
            return check_diag_collate_fn(batch, self.tokenizer, self.device, self.max_length)
        elif self.task == RESOLVE_TURN:
            return resolve_turn_collate_fn(batch, self.tokenizer, self.device, with_explanation=self.with_explanation)
        elif self.task == RESOLVE_DIAG:
            return resolve_diag_collate_fn(batch, self.tokenizer, self.device, with_explanation=self.with_explanation)
