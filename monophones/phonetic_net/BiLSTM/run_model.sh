#!/bin/bash

# EXPERIMENT NUMBER
exp_num=1

# learning rate
lr=0.0001

# batch size
bs=1

# optimizer
opt=adam

#########################
#########################

source activate tensorflow1.7_py3.5

CUDA_VISIBLE_DEVICES=5 python -u lstm_train.py $exp_num $lr $bs $opt > 'out_exp'$exp_num'.txt' 2>'out_exp'$exp_num'.err' &

source deactivate
disown
