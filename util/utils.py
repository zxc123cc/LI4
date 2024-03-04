import copy

import torch
import torch.nn as nn
from pathlib import Path
import time
import numpy as np
import random

import os
import torch.distributed as dist
import logging

from torch.cuda.amp import autocast
from transformers import get_scheduler, AdamW


def prepare_optimizer(model, opt_lr, weight_decay, eps):
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(model.named_parameters())
    optimizer_grouped_parameters = \
        [{'params': [p for n, p in model_param if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
         {'params': [p for n, p in model_param if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=opt_lr, eps=eps)

    return optimizer


def prepare_optimizer_delamination(model, igm_lr, opt_lr, weight_decay, eps):
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(model.named_parameters())

    igm_param_optimizer = []
    other_param_optimizer = []

    for name, param in model_param:
        if 'igm' in str(name):
            igm_param_optimizer.append((name, param))
        else:
            other_param_optimizer.append((name, param))

    optimizer_grouped_parameters = [
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay, 'lr': opt_lr},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt_lr},

        {"params": [p for n, p in igm_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay, 'lr': igm_lr},
        {"params": [p for n, p in igm_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': igm_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=opt_lr, eps=eps)

    return optimizer


def prepare_scheduler(optimizer, epochs, steps_per_epoch, warmup_rate,gradient_accumulation_steps=1):
    total_steps = (epochs * steps_per_epoch) / gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_rate)
    scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    return scheduler


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    # args.n_gpu = 1


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(args):
    time_ = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    log_file = os.path.join(args.save_model_path, f'{time_}.txt')
    logging.basicConfig(filename=log_file, format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    logger = logging.getLogger(__name__)
    return logger


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    print("dist", args)
    # prepare distributed
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    print(4)
    # set cuda device
    torch.cuda.set_device(args.gpu)


class Logger():
    def __init__(self, file_name, mode='w', buffer=100):
        (Path(file_name).parent).mkdir(exist_ok=True, parents=True)
        self.file_name = file_name
        self.fp = open(file_name, mode)
        self.cnt = 0
        self.stamp = time.time()
        self.buffer = buffer

    def log(self, *args, end='\n'):
        for x in args:
            if isinstance(x, dict):
                for y in x:
                    self.fp.write(str(y) + ':' + str(x[y]) + ' ')
            else:
                self.fp.write(str(x) + ' ')
        self.fp.write(end)
        self.cnt += 1
        if self.cnt >= self.buffer or time.time() - self.stamp > 5:
            self.cnt = 0
            self.stamp = time.time()
            self.fp.close()
            self.fp = open(self.file_name, 'a')
        pass

    def close(self):
        self.fp.close()


