#!/bin/bash

# EXPERIMENT NUMBER
exp_num=2

# restore epoch
re=30


#########################
#########################

source activate tf1.7_py3.5

CUDA_VISIBLE_DEVICES=5 python -u lstm_test.py $exp_num $re > 'test_exp'$exp_num'_epoch'$re'.txt' 2>'test_exp'$exp_num'_epoch'$re'.err' &

source deactivate
disown
