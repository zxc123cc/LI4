import argparse
import csv
from typing import Any, Optional, Tuple

import torch
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BertTokenizer
import random
from functools import partial
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler


def create_dataloaders(args,tokenizer):
    train_dataset = GenDataset(args,tokenizer,mode='train')
    val_dataset = GenDataset(args,tokenizer,mode='dev')
    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    else:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)

    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.train_batch_size,
                                        sampler=train_sampler,
                                        drop_last=False,
                                        collate_fn=train_dataset.pad_collate)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False,
                                      collate_fn=val_dataset.pad_collate)
    return train_dataloader, val_dataloader

class GenDataset(Dataset):

    def __init__(self, args, tokenizer, mode='train') -> object:
        if mode == 'train':
            data = json.load(open(args.train_file, 'r', encoding='utf-8'))
        elif mode == 'dev':
            data = json.load(open(args.dev_file, 'r', encoding='utf-8'))
        else:
            data = json.load(open(args.test_file, 'r', encoding='utf-8'))
        self.samples = list(data.values())
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.mode = mode
        self.is_IGM = args.is_IGM
        # self.decoder_start_token_id = self.tokenizer.vocab['[SEP]']

    def __len__(self) -> int:
        return len(self.samples)

    def get_index_mask(self,input_ids):
        bs = input_ids.shape[0]
        claim_mask = torch.zeros_like(input_ids)
        evidence_mask = torch.zeros_like(input_ids)
        question_mask = torch.zeros_like(input_ids)
        for i in range(bs):
            # <s> token id = 0  , </s> token id = 2
            s_idxs = torch.nonzero(input_ids[i]==2).squeeze()
            cs,cd = int(0), int(s_idxs[0])
            es,ed = int(s_idxs[0]),int(s_idxs[1])
            qs,qd = int(s_idxs[1]),int(s_idxs[2])
            claim_mask[i][cs+1:cd] = 1
            evidence_mask[i][es+1:ed] = 1
            question_mask[i][qs+1:qd] = 1
        return claim_mask, evidence_mask, question_mask

    # def get_index_mask2(self,input_ids):
    #     bs = input_ids.shape[0]
    #     claim_mask = torch.zeros_like(input_ids)
    #     evidence_mask = torch.zeros_like(input_ids)
    #     question_mask = torch.zeros_like(input_ids)
    #     for i in range(bs):
    #         # <s> token id = 0  , </s> token id = 2
    #         c_idxs = torch.nonzero(input_ids[i]==2026).squeeze()
    #         e_idxs = torch.nonzero(input_ids[i]==1283).squeeze()
    #         q_idxs = torch.nonzero(input_ids[i]==864).squeeze()
    #         s_idxs = torch.nonzero(input_ids[i]==2).squeeze()
    #         cs,cd = int(c_idxs[1]),int(e_idxs[1])
    #         es,ed = int(e_idxs[1]),int(q_idxs[1])
    #         qs,qd = int(q_idxs[1]),int(s_idxs)
    #         claim_mask[i][cs+1:cd] = 1
    #         evidence_mask[i][es+1:ed] = 1
    #         question_mask[i][qs+1:qd] = 1
    #     return claim_mask, evidence_mask, question_mask
    def get_index_mask2(self,input_ids):
        bs = input_ids.shape[0]
        claim_mask = torch.zeros_like(input_ids)
        evidence_mask = torch.zeros_like(input_ids)
        question_mask = torch.zeros_like(input_ids)
        for i in range(bs):
            # <s> token id = 0  , </s> token id = 2
            s_idxs = torch.nonzero(input_ids[i]==2).squeeze()
            cs,cd = int(s_idxs[0]),int(s_idxs[1])
            es,ed = int(s_idxs[1]),int(s_idxs[2])
            qs,qd = int(s_idxs[2]),int(s_idxs[3])
            claim_mask[i][cs+1:cd] = 1
            evidence_mask[i][es+1:ed] = 1
            question_mask[i][qs+1:qd] = 1
        return claim_mask, evidence_mask, question_mask

    def get_label_idx(self,input_ids):
        s_idxs = torch.nonzero(input_ids[0]==2800).squeeze()
        r_idxs = torch.nonzero(input_ids[0]==33898).squeeze()
        n_idxs = torch.nonzero(input_ids[0]==7974).squeeze()
        try:
            s_idx = int(s_idxs[0])
        except:
            s_idx = int(s_idxs)
        try:
            r_idx = int(r_idxs[0])
        except:
            r_idx = int(r_idxs)
        try:
            n_idx = int(n_idxs[0])
        except:
            n_idx = int(n_idxs)

        return [s_idx,r_idx,n_idx]

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        if self.is_IGM:
            prefix = 'Determine whether the claim is supported, refuted, or neutral by the given evidence and question.'
            input_text = prefix + ' </s> ' + 'claim: ' + sample['claim'] + ' </s> ' + 'evidence: ' + sample['evidence'] + ' </s> ' + 'question: ' + sample['question']
        else:
            input_text = sample['claim'] + ' </s> ' + sample['evidence'] + ' </s> ' + sample['question']
        claim = sample['claim']
        evidence = sample['evidence']
        question = sample['question']
        if self.mode == 'test':
            label = 10000
        else:
            label = sample['label_id']
        return input_text, label, claim, evidence, question

    def pad_collate(self, batch):
        data = {}
        input_text, label, claim, evidence, question = zip(*batch)
        tokenizer_output = self.tokenizer.batch_encode_plus(
            input_text, padding=True, return_tensors='pt'
        )
        input_ids, attention_mask = tokenizer_output.input_ids, tokenizer_output.attention_mask
        if self.is_IGM:
            claim_mask, evidence_mask, question_mask = self.get_index_mask2(input_ids)
        else:
            claim_mask, evidence_mask, question_mask = self.get_index_mask(input_ids)

        data['input_ids'] = input_ids
        data['attention_mask'] = attention_mask
        data['label'] = torch.LongTensor(label)

        data['claim_mask'] = claim_mask
        data['evidence_mask'] = evidence_mask
        data['question_mask'] = question_mask

        label_idx = self.get_label_idx(input_ids)
        data['label_idx'] = torch.LongTensor(label_idx)

        return data