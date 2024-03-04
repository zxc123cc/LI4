import logging
import sys
import time

from torch.cuda.amp import autocast

sys.path.append('models')
import os
import torch
import tqdm
from transformers import RobertaTokenizer
import transformers

transformers.logging.set_verbosity_error()
from util.utils import prepare_optimizer, prepare_scheduler,prepare_optimizer_delamination
from models.model_faviq import DialFactClassification
from dataset.data_helper_faviq import create_dataloaders
from evaluate import evaluate
from config import parse_args
from util.utils import init_distributed_mode, setup_logging, setup_device, setup_seed

def train_and_validate(args):
    device = torch.device(args.device)
    print('device: ', device)
    print("create dataloader")
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_model_dir, do_lower_case=args.do_lower_case)
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(args, tokenizer,test=True)
    print("load model")

    model = DialFactClassification(args.pretrain_model_dir,num_labels=2)
    model.to(device)

    optimizer = prepare_optimizer(model, args.learning_rate, args.weight_decay, args.eps)
    # optimizer = prepare_optimizer_delamination(model, igm_lr=3e-5,opt_lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.eps)

    scheduler = prepare_scheduler(optimizer, args.max_epochs, len(train_dataloader), args.warmup_ratio,
                                  gradient_accumulation_steps=args.gradient_accumulation_steps)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    best_score_val = 0
    best_score_test = 0
    step, global_step = 0, 0
    dev_step = 1000
    print("start training")
    logging.info(f"start time >>> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    for epoch in range(args.max_epochs):
        model.train()
        with tqdm.tqdm(total=train_dataloader.__len__(), desc=f"[{epoch + 1}] / [{args.max_epochs}]training... ") as t:
            for index, batch in enumerate(train_dataloader):
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)

                label_ids = batch['label'].to(args.device)

                claim_mask = batch['claim_mask'].to(args.device)
                evidence_mask = batch['evidence_mask'].to(args.device)
                question_mask = batch['question_mask'].to(args.device)
                label_idx = batch['label_idx'].to(args.device)

                output_dict = model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=label_ids,
                                    label_tmp=None,

                                    claim_mask=claim_mask,
                                    evidence_mask=evidence_mask,
                                    question_mask=question_mask,

                                    label_idx=label_idx
                                    )
                loss_dict = output_dict.get("loss")
                loss = loss_dict['loss']
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                tmp_dic = {}
                for k,v in loss_dict.items():
                    tmp_dic[k] = v.cpu().item()

                step += 1

                t.set_postfix(tmp_dic)
                t.update(1)

                if step % dev_step == 0 and (epoch+1) > 2:
                    meters = evaluate(args, model, val_dataloader)
                    score = meters['acc']
                    logging.info(f"val: acc = {meters['acc']},  macro_f1 = {meters['macro_f1']}, macro_recall = {meters['macro_recall']}, macro_precision = {meters['macro_precision']}")

                    if score >= best_score_val:
                        best_score_val = score
                        model_path = os.path.join(args.save_model_path, f'model_best_val.bin')
                        state_dict = model.state_dict()
                        torch.save(state_dict, model_path)

                    torch.cuda.empty_cache()

                    meters = evaluate(args, model, test_dataloader)
                    score = meters['acc']
                    logging.info(f"test: acc = {meters['acc']},  macro_f1 = {meters['macro_f1']}, macro_recall = {meters['macro_recall']}, macro_precision = {meters['macro_precision']}")

                    if score >= best_score_test:
                        best_score_test = score
                        model_path = os.path.join(args.save_model_path, f'model_best_test.bin')
                        state_dict = model.state_dict()
                        torch.save(state_dict, model_path)




if __name__ == '__main__':
    args = parse_args()
    # 设置显卡
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids
    # os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    if args.distributed:
        init_distributed_mode(args)

    os.makedirs(args.save_model_path, exist_ok=True)

    setup_logging(args)
    setup_device(args)
    setup_seed(args)

    logging.info("Training/evaluation parameters: %s", args)
    train_and_validate(args)
