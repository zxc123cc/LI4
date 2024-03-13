import csv
import sys
import torch
import torch.nn as nn
import os
import tqdm
import numpy as np
sys.path.append('models')
from transformers import RobertaTokenizer
from dataset.data_helper import create_dataloaders
from models.model import DialFactClassification
from config import parse_args
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

label_map = {0:"Supports", 1:"Refutes", 2:"Neutral"}

def compute_metrics_fn(preds, labels):
    assert len(preds) == len(labels)
    f1 = f1_score(y_true= labels, y_pred=preds, average="macro", labels=np.unique(labels))
    acc = accuracy_score(y_true= labels, y_pred=preds)
    p = precision_score(y_true= labels, y_pred=preds, average="macro", labels=np.unique(labels))
    r = recall_score(y_true= labels, y_pred=preds, average="macro", labels=np.unique(labels))
    return {
        "acc": acc,
        "macro_f1": f1,
        "macro_recall":r,
        "macro_precision": p
    }

def evaluate(args, model, loader):
    model.eval()
    with torch.no_grad():
        all_preds, all_labels = [], []
        with tqdm.tqdm(total=loader.__len__()) as t:
            for index, batch in enumerate(loader):
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)

                label_ids = batch['label'].to(args.device)

                claim_mask = batch['claim_mask'].to(args.device)
                evidence_mask = batch['evidence_mask'].to(args.device)
                question_mask = batch['question_mask'].to(args.device)

                label_idx = batch['label_idx'].to(args.device)

                output_dict = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=label_ids,

                    claim_mask=claim_mask,
                    evidence_mask=evidence_mask,
                    question_mask=question_mask,

                    label_idx=label_idx
                )
                logits=output_dict.get('logits')
                pred = torch.argmax(logits, dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(label_ids.cpu().numpy())
                t.update(1)

        score_dic = compute_metrics_fn(all_preds,all_labels)
        print(f"acc = {score_dic['acc']},  macro_f1 = {score_dic['macro_f1']}, macro_recall = {score_dic['macro_recall']}, macro_precision = {score_dic['macro_precision']}")

    model.train()
    return score_dic


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device)
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_model_dir, do_lower_case=args.do_lower_case)
    _, val_dataloader = create_dataloaders(args, tokenizer)
    model = DialFactClassification(args.pretrain_model_dir)
    model.to(device)

    model.load_state_dict(torch.load(args.ckpt_file, map_location='cpu'))

    score_dic = evaluate(args, model, val_dataloader)