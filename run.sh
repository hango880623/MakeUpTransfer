#!/bin/bash

python main.py \
    --mode train \
    --dataset MT \
    --image_size 128 \
    --c_dim 3 \
    --sample_dir cagan_Data0421_new/samples \
    --log_dir cagan_Data0421_new/logs \
    --model_save_dir cagan_Data0421_new/models \
    --result_dir cagan_Data0421_new/results \
    --attr_path data/Data0421/clean.txt \
    --mt_image_dir data/Data0421/customized_cropped \
    --train_label data/Data0421/label.csv \
    --test_label data/Data0421/label.csv \
    --sample_step 1000 \
    --model_save_step 10000 \
    --lambda_cls 10 \
    --lambda_rec 200 \
    --lambda_gp 10 \
    --lambda_bkg 6 \
    --num_workers 2 \
    --num_iters_decay 20000 \
    --resume_iters 750000 \
    --num_iters 850000 \
    --g_lr 0.00004 \
    --d_lr 0.00004 