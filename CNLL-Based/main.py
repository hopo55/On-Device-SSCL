import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models import resnet
from losses import SupConLoss
import data_generator
import dataloader

parser = argparse.ArgumentParser()
# General Settings
parser.add_argument('--gpuid', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
# Dataset Settings
parser.add_argument('--root', type=str, default='./data/')
parser.add_argument('--dataset', default='CIFAR100')
parser.add_argument('--mode', type=str, default='super')
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--label_ratio', type=float, default=0.2, help="Labeled data ratio")
# Model Settings
parser.add_argument('--num_class', type=int, default=100)
parser.add_argument('--lr', '--learning_rate', type=float, default=0.0005) ### Learning Rate Should not be more than 0.001
parser.add_argument('--buffer_size', type=int, default=0, help="size of buffer for replay")

# Not yet used
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)

args = parser.parse_args()

def main():
    ## GPU Setup
    torch.cuda.set_device(args.gpuid)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # Dataset Generator
    data_generator.__dict__[args.dataset + '_Generator'](args)

    # Create Model
    model = resnet.ResNet18(args.num_class)

    # Semi-Supervised Loss
    criterion = nn.CrossEntropyLoss()
    contrastive_criterion = SupConLoss()

    # Optimizer and Scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=240, eta_min=2e-4)    # 실험을 통해 수정

    if args.dataset == 'CIFAR10':
        task_mode_list = ['task_0', 'task_1', 'task_2', 'task_3', 'task_4']
    elif args.dataset == 'CIFAR100':
        task_mode_list = ['task_0', 'task_1', 'task_2', 'task_3', 'task_4', 'task_5', 'task_6', 'task_7', 'task_8', 'task_9', 'task_10', 'task_11', 'task_12', 'task_13', 'task_14', 'task_15', 'task_16', 'task_17', 'task_18', 'task_19']

    for task_mode in task_mode_list:
        train_loader = dataloader.dataloader(args)
        train_loader.load()


if __name__ == '__main__':
    main()