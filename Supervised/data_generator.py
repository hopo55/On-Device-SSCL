import os
import random
import numpy as np

import torch
from torchvision import datasets, transforms


class CIFAR10(datasets.CIFAR10):
    name = 'CIFAR10'
    num_classes = 10

    def __init__(self, args, train=True):
        self.split_data = []
        self.split_target = []
        self.root = os.path.join(args.root, args.dataset)

        transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor()])

        datasets.CIFAR10.__init__(self, root=self.root, train=train, transform=transform, download=True)

        if train:
            save_path = self.root + '/train'
            if not os.path.exists(save_path): os.mkdir(save_path)
        else:
            save_path = self.root + '/test'
            if not os.path.exists(save_path): os.mkdir(save_path)

        for y in range(self.num_classes):
            cls_idx = torch.nonzero(torch.Tensor(self.targets) == y)
            self.split_data = [self.data[loc] for loc in cls_idx]
            self.split_target = [self.targets[loc] for loc in cls_idx]
        
            np.save(os.path.join(save_path, args.dataset + '_Class' + str(y)), np.array(self.split_data))
            np.save(os.path.join(save_path, args.dataset + '_Labels' + str(y)), np.array(self.split_target))

    def __getitem__(self, index):
        x, y = self.split_data[index], self.split_target[index]
        
        return x, y


class CIFAR100(datasets.CIFAR100):
    name = 'CIFAR100'
    num_classes = 100

    def __init__(self, args, train=True):
        self.root = os.path.join(args.root, args.dataset)

        transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor()])

        datasets.CIFAR100.__init__(self, root=self.root, train=train, transform=transform, download=True)

        if train:
            save_path = self.root + '/train'
            if not os.path.exists(save_path): os.mkdir(save_path)
        else:
            save_path = self.root + '/test'
            if not os.path.exists(save_path): os.mkdir(save_path)

        for y in range(self.num_classes):
            cls_idx = torch.nonzero(torch.Tensor(self.targets) == y)
            self.split_data = [self.data[loc] for loc in cls_idx]
            self.split_target = [self.targets[loc] for loc in cls_idx]
        
            np.save(os.path.join(save_path, args.dataset + '_Class' + str(y)), np.array(self.split_data))
            np.save(os.path.join(save_path, args.dataset + '_Labels' + str(y)), np.array(self.split_target))

    def __getitem__(self, index):
        x, y = self.split_data[index], self.split_target[index]
        
        return x, y

class HAR():
    name = 'HAR'
    num_classes = 6

    def __init__(self, args, train=True):
        self.split_data = []
        self.split_target = []
        self.root = os.path.join(args.root, args.dataset)
        INPUT_SIGNAL_TYPES = ["body_acc_x_", "body_acc_y_", "body_acc_z_",
                              "body_gyro_x_", "body_gyro_y_", "body_gyro_z_"]
        if train:
            data_file = [os.path.join(self.root, 'train', 'Inertial Signals', item + 'train.txt') for item in INPUT_SIGNAL_TYPES]
            target_file = os.path.join(self.root, 'train','y_train.txt')
            save_path = self.root + '/train'

        else:
            data_file = [os.path.join(self.root, 'test', 'Inertial Signals', item + 'test.txt') for item in INPUT_SIGNAL_TYPES]
            target_file = os.path.join(self.root, 'test', 'y_test.txt')
            save_path = self.root + '/test'

        data = self.format_data_x(data_file)
        data = np.transpose(data, [0,2,1])
        targets = self.format_data_y(target_file)

        for y in range(self.num_classes):
            cls_idx = torch.nonzero(torch.Tensor(targets) == y)
            self.split_data = [data[loc] for loc in cls_idx]
            self.split_target = [targets[loc] for loc in cls_idx]
        
            np.save(os.path.join(save_path, args.dataset + '_Class' + str(y)), np.array(self.split_data))
            np.save(os.path.join(save_path, args.dataset + '_Labels' + str(y)), np.array(self.split_target))

    def format_data_x(self, fns):
        x_data = None
        for item in fns:
            item_data = np.loadtxt(item, dtype=np.float32)
            if x_data is None:
                x_data = np.zeros((len(item_data), 1))
            x_data = np.hstack((x_data, item_data))
        x_data = x_data[:, 1:]

        X = None
        for i in range(len(x_data)):
            row = np.asarray(x_data[i, :], dtype=np.float32)
            row = row.reshape(6, 128).T
            if X is None:
                X = np.zeros((len(x_data), 128, 6))
            X[i] = row

        return X

    def format_data_y(self, datafile):
        data = np.loadtxt(datafile, dtype=np.int) - 1
        return data

    def __getitem__(self, index):
        x, y = self.split_data[index], self.split_target[index]
        
        return x, y

DATASET = {
    CIFAR10.name: CIFAR10,
    CIFAR100.name: CIFAR100,
    HAR.name: HAR
}

def Generator(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    DATASET[args.dataset](args)
    print('Train Dataset Save Done!')
    DATASET[args.dataset](args, train=False)
    print('Test Dataset Save Done!')

