#!/bin/bash

python3 train.py \
    --model_name model_01 \
    --architecture resnet34 \
    --img_type square_resize \
    --seed 1048 \
    --multi_sample_aug mixup \
    --epoch 60 \
    --debug false

python3 train.py \
    --model_name model_02 \
    --architecture resnet34 \
    --img_type square_crop \
    --seed 1049 \
    --multi_sample_aug mixup \
    --epoch 60 \
    --debug false

python3 train.py \
    --model_name model_03 \
    --architecture resnet34 \
    --img_type square_resize \
    --seed 1050 \
    --multi_sample_aug cutmix \
    --epoch 60 \
    --debug false

python3 train.py \
    --model_name model_04 \
    --architecture resnet34 \
    --img_type square_crop \
    --seed 1051 \
    --multi_sample_aug cutmix \
    --epoch 60 \
    --debug false

python3 train.py \
    --model_name model_05 \
    --architecture resnet34 \
    --img_type square_resize \
    --seed 1052 \
    --multi_sample_aug cutmixup \
    --epoch 60 \
    --debug false
