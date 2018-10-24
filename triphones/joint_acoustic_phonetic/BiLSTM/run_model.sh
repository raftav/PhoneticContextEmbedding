#!/bin/bash

# EXPERIMENT NUMBER
exp_num=4

# learning rate
lr=0.01

# batch size
bs=10

# optimizer
opt=adam

# lambda embedding
le=0.2

# lambda reconstruction
l_re=1.0

#########################
#########################

source activate tf1.7_py3.5

CUDA_VISIBLE_DEVICES=5 python -u lstm_train.py $exp_num $lr $bs $opt $le $l_re > 'out_exp'$exp_num'.txt' 2>'out_exp'$exp_num'.err' &

source deactivate
disown
