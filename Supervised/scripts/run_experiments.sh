#!/bin/bash

SEED=0
DEVICE=0
DEVICE_NAME='cal_05'
DATASET='CIFAR100'
MODEL='ResNet18'
CLASSIFIER='FC'
EPOCH=10
CL='Fine-tune'
INCREMENT=1

export CUDA_VISIBLE_DEVICES=$DEVICE

python -m main --seed $SEED --device $DEVICE --device_name $DEVICE_NAME --dataset $DATASET --model_name $MODEL --classifier $CLASSIFIER --epoch $EPOCH --cl_mode $CL --class_increment $INCREMENT