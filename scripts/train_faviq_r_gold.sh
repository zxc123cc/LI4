echo '开始训练'

device=$1
python train_faviq_r_gold.py\
    --save_model_path ./model_storage/faviq_r \
    --device_ids 0 \
    --device 'cuda' \
    --num_workers 6 \
    --prefetch 12 \
    --pretrain_model_dir /gemini/code/roberta-large \
    --max_epochs 30 \
    --train_batch_size 12 \
    --val_batch_size 16 \
    --max_length 512 \
    --gradient_accumulation_steps 4 \
    --seed 2023 \
    --is_LabelEmb \
    --is_IGM \
    --iter_num 3 \
    --train_file ./data/tmp_data/faviq_r_set/train.json \
    --dev_file ./data/tmp_data/faviq_r_set/dev.json \
    --test_file ./data/tmp_data/faviq_r_set/test.json \

