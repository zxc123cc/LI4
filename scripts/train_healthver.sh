echo '开始训练'

device=$1
python train_healthver.py\
    --save_model_path ./model_storage/healthver \
    --device_ids 0 \
    --device 'cuda' \
    --num_workers 6 \
    --prefetch 12 \
    --pretrain_model_dir /gemini/code/roberta-large \
    --max_epochs 30 \
    --train_batch_size 32 \
    --val_batch_size 8 \
    --max_length 512 \
    --gradient_accumulation_steps 1 \
    --seed 2023 \
    --is_LabelEmb \
    --is_IGM \






