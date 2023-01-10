import os
import yaml
import random
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, ConcatDataset


class DataLoader():
    def __init__(self, config, data_config):
        self.config = config
        self.data_config = data_config
        self.datasets = {}
        self.eval_datasets = {}
        self.total_step = 0
        self.stage = -1
        
        # Prepare datasets
        for stage in self.config:
            for subset in stage['subsets']:
                dataset_name, _ = subset
                if dataset_name in self.datasets:
                    continue

                self.datasets[dataset_name] = DATASET[dataset_name](self.config, self.data_config)
                self.eval_datasets[dataset_name] = DATASET[dataset_name](self.config, self.data_config, train=False)
                self.total_step += len(self.datasets[dataset_name]) // self.data_config['batch_size']

        self.task_datasets = []
        for stage in self.config:
            subsets = []
            for dataset_name, subset_name in stage['subsets']:
                subsets.append(self.datasets[dataset_name].subsets[subset_name])
            dataset = ConcatDataset(subsets)
            self.task_datasets.append(dataset)

    def __len__(self):
        return self.total_step

class CIFAR100(torchvision.datasets.CIFAR100):
    name = 'cifar100'
    num_classes = 100

    def __init__(self, config, data_config, train=True):
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor()])
        torchvision.datasets.CIFAR100.__init__(self, root=os.path.join(data_config['data_root'], 'cifar100'),
                                               train=train, transform=transform, download=True)
        
        self.org_targets = deepcopy(self.targets)
        self.subsets

        if train:
            if data_config['superclass_noise']:
                # symmetric noise within superclass
                super_classes = [["beaver", "dolphin", "otter", "seal", "whale"],
                                 ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
                                 ["orchid", "poppy", "rose", "sunflower", "tulip"],
                                 ["bottle", "bowl", "can", "cup", "plate"],
                                 ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
                                 ["clock", "keyboard", "lamp", "telephone", "television"],
                                 ["bed", "chair", "couch", "table", "wardrobe"],
                                 ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
                                 ["bear", "leopard", "lion", "tiger", "wolf"],
                                 ["bridge", "castle", "house", "road", "skyscraper"],
                                 ["cloud", "forest", "mountain", "plain", "sea"],
                                 ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
                                 ["fox", "porcupine", "possum", "raccoon", "skunk"],
                                 ["crab", "lobster", "snail", "spider", "worm"],
                                 ["baby", "boy", "girl", "man", "woman"],
                                 ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
                                 ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
                                 ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
                                 ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
                                 ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],]
                for super_cls in super_classes:
                    cls_idx = [self.class_to_idx[c] for c in super_cls]
                    print(cls_idx)

    def __getitem__(self, idx):
        x, y = torchvision.datasets.CIFAR100.__getitem__(self, idx)

        return x, y


DATASET = {
    CIFAR100.name: CIFAR100
}

parser = ArgumentParser()
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--config', '-c', default='configs/cifar100.yaml')
parser.add_argument('--log-dir', '-l', default='./data')
parser.add_argument('--override', default='')
parser.add_argument('--dataset', default='CIFAR100')
parser.add_argument('--data_path', type=str, default='./data/CIFAR100/')
parser.add_argument('--savepath', default='./data/')

args = parser.parse_args()

config_path = args.config
config = yaml.load(open(config_path), Loader=yaml.FullLoader)

random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

data_config = {
    'data_root': args.data_path,
    'batch_size': 10,
    'superclass_noise': True,
    'corruption_percent': 0.4 # may be not use this
}

data_loader = DataLoader(config, data_config)