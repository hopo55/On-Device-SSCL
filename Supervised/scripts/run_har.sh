#!/bin/bash

SEED=0
DEVICE=0
DEVICE_NAME='hspark'
DATASET='HAR'
MODEL='ResNet18_HAR'
EPOCH=100
BATCH_SIZE=512
TEST_SIZE=256
CL='FC'
INCREMENT=1

export CUDA_VISIBLE_DEVICES=$DEVICE

# # Full training
# # FC / DeepNCM
python -m main --seed $SEED --device $DEVICE --device_name $DEVICE_NAME --dataset $DATASET --batch_size $BATCH_SIZE --test_size $TEST_SIZE --model_name $MODEL --epoch $EPOCH --cl_mode $CL --class_increment $INCREMENT