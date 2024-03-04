echo '开始预测'

device=$1
python evaluate.py\
    --ckpt_file ./model_storage/model_best_healthver.bin \
    --device_ids 0 \
    --device 'cuda' \
    --pretrain_model_dir F:/pretrained_model/roberta-large \
    --val_batch_size 8 \
    --is_LabelEmb \
    --is_IGM \
#    --num_workers 6 \
#    --prefetch 12 \


