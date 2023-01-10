import os
import yaml
import random
import argparse
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--config', '-c', default='configs/cifar100.yaml')
parser.add_argument('--dataset', default='CIFAR100')
parser.add_argument('--label_ratio', type=float, default=0.2, help="Labeled data ratio")
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--root', type=str, default='./data/')
parser.add_argument('--mode', type=str, default='super')

args = parser.parse_args()

class DataGenerator():
    def __init__(self, args, config):
        self.config = config
        self.train_datasets = {}
        self.test_datasets = {}

        self.train_datasets = DATASET[args.dataset](args, self.config)
        self.test_datasets = DATASET[args.dataset](args, self.config, train=False)

        self.task_datasets = []
        for step in self.config:
            subsets = []
            for _, subset_name in step['subsets']:
                subsets.append(self.train_datasets.subsets[subset_name])
            
            dataset = ConcatDataset(subsets)
            self.task_datasets.append(dataset)

    def __iter__(self):
        for idx, task in enumerate(self.task_datasets):
            print('\nProgressing to Task %d' % idx)
            for data in DataLoader(task, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=False):
                yield data, idx

    def __len__(self):
        return len(self.train_datasets)


class CIFAR10(datasets.CIFAR10):
    name = 'CIFAR10'
    num_classes = 10

class CIFAR100(datasets.CIFAR100):
    name = 'CIFAR100'
    num_classes = 100

    def __init__(self, args, config, train=True):
        self.subsets = dict()
        transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor()])
        datasets.CIFAR100.__init__(self, root=os.path.join(args.root, args.dataset), train=train, transform=transform, download=True)

        # Create subset for each class
        for y in range(self.num_classes):
            self.subsets[y] = Subset(self, torch.nonzero((torch.Tensor(self.targets) == y)).squeeze(1).tolist())

    def __getitem__(self, idx):
        x, y = datasets.CIFAR100.__getitem__(self, idx)

        return x, y

DATASET = {
    CIFAR10.name: CIFAR10,
    CIFAR100.name: CIFAR100
}

def CIFAR100_Generator():
    pass

def CIFAR100_Generator():
    config_path = args.config

    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_generator = DataGenerator(args, config)

    for step, (x, y) in enumerate(data_generator):
        print(y)
        if step > 40:
            break

if args.dataset == 'CIFAR10': CIFAR100_Generator()
elif args.dataset == 'CIFAR100': CIFAR100_Generator()