import random
import argparse
import numpy as np
from models.resnet import *

import torch
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
parser.add_argument('--warm_up', default=5, type=int, help='warmup epochs') 
parser.add_argument('--warmup_epochs', default=5, type=int)
parser.add_argument('--finetune_epochs', default=10, type=int)
parser.add_argument('--lr', '--learning_rate', default=0.0005, type=float, help='initial learning rate')    ### Learning Rate Should not be more than 0.001
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--r', default=0.2, type=float, help='noise ratio')

parser.add_argument('--seed', default=0)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=100, type=int)
parser.add_argument('--data_path', default='./data/CIFAR100', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar100', type=str)
args = parser.parse_args()

## GPU Setup 
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.cuda.set_device(args.gpuid)
cudnn.benchmark = True

if args.dataset== 'cifar100':
    torchvision.datasets.CIFAR100(args.data_path, train=True, download=True)
    torchvision.datasets.CIFAR100(args.data_path, train=False, download=True)

model = ResNet18(args.num_class)
model = model.cuda()

criterion = None

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer, 240, 2e-4)

task_mode_list = ['task_0', 'task_1', 'task_2', 'task_3', 'task_4', 'task_5', 'task_6', 'task_7', 'task_8', 'task_9', 'task_10', 'task_11', 'task_12', 'task_13', 'task_14', 'task_15', 'task_16', 'task_17', 'task_18', 'task_19']

for task_mode in task_mode_list:
    