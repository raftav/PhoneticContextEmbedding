#!/bin/bash

# EXPERIMENT NUMBER
exp_num=2

# learning rate
lr=0.0001

# batch size
bs=1

# optimizer
opt=adam

# embedding size
es=100

# num nodes
nn=100

#num layer
nl=2

#########################
#########################

source activate tensorflow1.7_py3.5

CUDA_VISIBLE_DEVICES=5 python -u lstm_train.py $exp_num $lr $bs $opt $es $nn $nl > 'out_exp'$exp_num'.txt' 2>'out_exp'$exp_num'.err' &

source deactivate
disown
