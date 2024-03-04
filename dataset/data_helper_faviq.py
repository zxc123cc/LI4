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


def create_dataloaders(args,tokenizer,test=False):
    train_dataset = FAVIQDataset(args,args.train_file,tokenizer,mode='gold')
    val_dataset = FAVIQDataset(args,args.dev_file,tokenizer,mode='gold')
    if test:
        test_dataset = FAVIQDataset(args,args.test_file,tokenizer,mode='gold')
    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    else:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    if test:
        test_sampler = SequentialSampler(test_dataset)

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
    if test:
        test_dataloader = dataloader_class(test_dataset,
                                          batch_size=args.val_batch_size,
                                          sampler=test_sampler,
                                          drop_last=False,
                                          collate_fn=test_dataset.pad_collate)
        return train_dataloader, val_dataloader, test_dataloader
    return train_dataloader, val_dataloader

class FAVIQDataset(Dataset):
    def __init__(self, args, file_path,tokenizer, mode='gold') -> object:
        data = json.load(open(file_path, 'r', encoding='utf-8'))
        self.samples = list(data.values())
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.mode = mode

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

    def get_index_mask2(self,input_ids):
        bs = input_ids.shape[0]
        claim_mask = torch.zeros_like(input_ids)
        evidence_mask = torch.zeros_like(input_ids)
        question_mask = torch.zeros_like(input_ids)
        for i in range(bs):
            # <s> token id = 0  , </s> token id = 2
            s_idxs = torch.nonzero(input_ids[i]==2).squeeze()
            cs,cd = int(s_idxs[0]),int(s_idxs[1])
            qs,qd = int(s_idxs[1]),int(s_idxs[2])
            es,ed = int(s_idxs[2]),int(s_idxs[3])
            claim_mask[i][cs+1:cd] = 1
            evidence_mask[i][es+1:ed] = 1
            question_mask[i][qs+1:qd] = 1
        return claim_mask, evidence_mask, question_mask

    def get_label_idx(self,input_ids):
        s_idxs = torch.nonzero(input_ids[0]==2800).squeeze()
        r_idxs = torch.nonzero(input_ids[0]==33898).squeeze()
        try:
            s_idx = int(s_idxs[0])
        except:
            s_idx = int(s_idxs)
        try:
            r_idx = int(r_idxs[0])
        except:
            r_idx = int(r_idxs)

        # return [[s_idx,r_idx,n_idx] for i in range(input_ids.shape[0])]
        return [s_idx,r_idx]

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        prefix = 'Determine whether the claim is supported or refuted by the given question and evidence.'
        sample['evidence'] = 'title: ' + sample['positive_evidence']['title'] + ' text: ' + sample['positive_evidence']['text']
        input_text = prefix + ' </s> ' + 'claim: ' + sample['claim']  + ' </s> ' + 'question: ' + sample['question'] + ' </s> ' + 'evidence: ' + sample['evidence']
        # input_text = sample['claim'] + ' </s> ' + sample['evidence'] + ' </s> ' + sample['question']
        claim = sample['claim']
        evidence = sample['evidence']
        question = sample['question']

        label = sample['label_id']
        label_tmp =  [0,0]
        label_tmp[label] = 1

        return input_text, label, label_tmp, claim, evidence, question

    def pad_collate(self, batch):
        data = {}
        input_text, label, label_tmp, claim, evidence, question = zip(*batch)
        tokenizer_output = self.tokenizer.batch_encode_plus(
            input_text, padding=True, max_length=self.max_length, truncation=True, return_tensors='pt'
        )
        input_ids, attention_mask = tokenizer_output.input_ids, tokenizer_output.attention_mask
        claim_mask, evidence_mask, question_mask = self.get_index_mask2(input_ids)

        data['input_ids'] = input_ids
        data['attention_mask'] = attention_mask
        data['label'] = torch.LongTensor(label)
        data['label_tmp'] = torch.LongTensor(label_tmp)

        data['claim_mask'] = claim_mask
        data['evidence_mask'] = evidence_mask
        data['question_mask'] = question_mask

        label_idx = self.get_label_idx(input_ids)
        data['label_idx'] = torch.LongTensor(label_idx)

        return data