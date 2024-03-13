import argparse
import os

base_dir = os.path.dirname(__file__)


def parse_args():
    print(base_dir)

    data_path = './data/tmp_data/'
    root_path = './'

    parser = argparse.ArgumentParser(description="DialFact Baseline")
    parser.add_argument("--seed", type=int, default=2023, help="random seed.")

    # ========================= Data Configs ==========================
    parser.add_argument('--train_file', type=str, default=data_path + 'healthver/healthver_train.json')
    parser.add_argument('--dev_file', type=str, default=data_path + 'healthver/healthver_dev.json')
    parser.add_argument('--test_file', type=str, default=data_path + 'healthver/healthver_test_with_label.json')

    parser.add_argument('--output_path', type=str, default='./results/eval.txt')

    # text length
    parser.add_argument('--max_length', default=512, type=int, help='input text length')

    # batch size
    parser.add_argument('--train_batch_size', default=2, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=2, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=32, type=int, help="use for testing duration per worker")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--save_model_path', type=str, default=root_path + 'model_storage/roberta_chk')
    parser.add_argument('--ckpt_file', type=str, default=root_path + 'checkpoint/v1/model_best.bin')
    parser.add_argument('--best_score', default=0.0, type=float, help='save checkpoint if now score > best_score')

    # ========================= Training Configs ==========================
    # fp16
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--max_epochs', type=int, default=20, help='How many epochs')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='initial learning rate')
    parser.add_argument('--warmup_ratio', default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--eps", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--prefetch', default=0, type=int, help="")
    parser.add_argument('--num_workers', default=0, type=int, help="num_workers for dataloaders")
    # parser.add_argument('--dropout_rate', default=0.1, type=float)

    # ========================== model =============================
    parser.add_argument('--pretrain_model_dir', type=str, default='F:/pretrained_model/roberta-base')
    # parser.add_argument('--pretrain_model_dir', type=str, default='F:/pretrained_model/bart-large')
    parser.add_argument("--do_lower_case", default=True, help="Set this flag if you are using an uncased model.")

    # DDP
    parser.add_argument('--device_ids', default='0')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--rank', type=int, default=0)


    parser.add_argument('--iter_num', default=3, type=int)
    parser.add_argument('--LabelEmb_dim', default=128, type=int)
    parser.add_argument('--is_IGM', action='store_true')
    parser.add_argument('--is_LabelEmb', action='store_true')


    return parser.parse_args(args=None)
