import os
import random
import argparse
import numpy as np

import torch
from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', default='CIFAR100')
parser.add_argument('--label_ratio', type=float, default=0.2, help="Labeled data ratio")
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--root', type=str, default='./data/')
parser.add_argument('--mode', type=str, default='super')

args = parser.parse_args()

class CIFAR10(datasets.CIFAR10):
    name = 'CIFAR10'
    num_classes = 10

    def __init__(self, args, train=True, semi=False):
        pass

class CIFAR100(datasets.CIFAR100):
    name = 'CIFAR100'
    num_classes = 100

    def __init__(self, args, train=True, semi=False, lab=True):
        self.subsets = dict()
        transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor()])
        datasets.CIFAR100.__init__(self, root=os.path.join(args.root, args.dataset), train=train, transform=transform, download=True)
        
        if args.mode == 'super':
            self.super_data = []
            self.super_target = []

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
                sup_subset = []
                cls_idx = [self.class_to_idx[c] for c in super_cls]
                
                for y in cls_idx:
                    sup_classes = torch.nonzero(torch.Tensor(self.targets) == y)

                    if train and semi:
                        split_point = int(len(sup_classes)*args.label_ratio)
                        if lab: sup_subset.append(sup_classes[:split_point])
                        else: sup_subset.append(sup_classes[split_point:])
                    else:
                        sup_subset.append(sup_classes)

                sup_subset = torch.cat(sup_subset).squeeze(1).tolist()
            
                self.super_data = [self.data[loc] for loc in sup_subset]
                self.super_target = [self.targets[loc] for loc in sup_subset]


    def __getitem__(self, idx):
        if args.mode == 'super':
            x, y = self.super_data[idx], self.super_target[idx]
        else:
            x, y = self.data[idx], self.targets[idx]

        return x, y

DATASET = {
    CIFAR10.name: CIFAR10,
    CIFAR100.name: CIFAR100
}

def CIFAR100_Generator():
    pass

def CIFAR100_Generator():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('Train Labeled Dataset Loading...')
    train_datasets = DATASET[args.dataset](args, semi=True)
    print('Train Unlabeled Dataset Loading...')
    train_datasets = DATASET[args.dataset](args, semi=True, lab=False)
    print('Test Dataset Loading...')
    test_datasets = DATASET[args.dataset](args, train=False)

if args.dataset == 'CIFAR10': CIFAR100_Generator()
elif args.dataset == 'CIFAR100': CIFAR100_Generator()