#!/bin/bash

SEED=0
DEVICE=0
DEVICE_NAME='hspark'
DATASET='CIFAR100'
MODEL='ImageNet_ResNet'
EPOCH=100
BATCH_SIZE=512
TEST_SIZE=256
CL='Fine-tuning'
INCREMENT=5

export CUDA_VISIBLE_DEVICES=$DEVICE

# Fine-tuning using FC
python -m main --seed $SEED --device $DEVICE --device_name $DEVICE_NAME --dataset $DATASET --batch_size $BATCH_SIZE --test_size $TEST_SIZE --model_name $MODEL --pre_trained --epoch $EPOCH --cl_mode $CL --class_increment $INCREMENT