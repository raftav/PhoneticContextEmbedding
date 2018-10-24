#!/bin/bash

# EXPERIMENT NUMBER
exp_num=2

# learning rate
lr=0.001

# batch size
bs=10

# optimizer
opt=adam


#########################
#########################

source activate tf1.7_py3.5

CUDA_VISIBLE_DEVICES=5 python -u lstm_train.py $exp_num $lr $bs $opt > 'out_exp'$exp_num'.txt' 2>'out_exp'$exp_num'.err' &

source deactivate
disown
