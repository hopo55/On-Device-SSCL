import os
import random
import autoaugment
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from collections import Counter

dataset_stats = {
    'CIFAR10' : {'mean': (0.49139967861519607, 0.48215840839460783, 0.44653091444546567),
                 'std' : (0.2470322324632819, 0.24348512800005573, 0.26158784172796434),
                 'size' : 32},
    'CIFAR100': {'mean': (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                 'std' : (0.2673342858792409, 0.25643846291708816, 0.2761504713256834),
                 'size' : 32}
}

def get_transform(dataset_name='CIFAR100', aug_type='none'):

    if aug_type == 'weak':
        transform_weak = transforms.Compose(
            [
                transforms.RandomCrop(dataset_stats[dataset_name]['size'], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(dataset_stats[dataset_name]['mean'], dataset_stats[dataset_name]['std']),
            ]
        )
        return transform_weak
    elif aug_type == 'strong':
        transform_strong = transforms.Compose(
            [
                autoaugment.RandomAugment(),
                transforms.ToTensor(),
                transforms.Normalize(dataset_stats[dataset_name]['mean'], dataset_stats[dataset_name]['std']),
                autoaugment.Cutout()
            ]
        )
        return transform_strong
    else:
        transform_none = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(dataset_stats[dataset_name]['mean'], dataset_stats[dataset_name]['std']),
            ]
        )
        return transform_none

class dataset(Dataset):
    def __init__(self, args, task, train=True, lab=True, buffer=None):
        self.root = os.path.join(args.root, args.dataset)
        self.args = args
        self.train = train
        self.lab = lab
        self.transform = {
            "labeled": [
                        get_transform(args.dataset, 'weak')
                       ],
            "unlabeled": [
                          get_transform(args.dataset, 'weak'),
                          get_transform(args.dataset, 'strong')
                         ],
            "test": [get_transform(args.dataset, 'none')]
        }

        if self.train:
            if self.lab:
                # load labeled data & label
                labeled_image_file = self.root + '/Train/Labeled/' + args.dataset + '_Images_Task' + str(task) + '_' + args.mode + '.npy'
                labeled_file = self.root + '/Train/Labeled/' + args.dataset + '_Labels_Task' + str(task) + '_' + args.mode + '.npy'

                train_xl = np.squeeze(np.load(labeled_image_file))
                train_yl = np.squeeze(np.load(labeled_file))
                self.train_xl = train_xl
                self.train_yl = train_yl

                if buffer is not None:
                    buffer_size, remainder = divmod(args.buffer_size, (task + 1))
                    if buffer_size > len(self.train_xl):
                        buffer_size, remainder = divmod(len(self.train_xl), (task + 1))

                    sample_list = list(range(len(self.train_xl)))
                    sample_list = random.sample(sample_list, buffer_size)

                    if task == 0:
                        self.buffer_x, self.buffer_y = self.train_xl[sample_list], self.train_yl[sample_list]
                    else:
                        # Update buffer
                        temp_buffer_x = []
                        temp_buffer_y = []
                        pre_size = self.args.buffer_size // (task)

                        for k in range(task + 1):
                            if k == task:
                                if remainder != 0:
                                    sample_list = list(range(len(self.train_xl)))
                                    sample_list = random.sample(sample_list, buffer_size+remainder)
                                temp_buffer_x.extend(self.train_xl[sample_list])
                                temp_buffer_y.extend(self.train_yl[sample_list])
                            else:
                                step = (pre_size*k)+buffer_size
                                temp_buffer_x.extend(buffer[0][pre_size*k:step])
                                temp_buffer_y.extend(buffer[1][pre_size*k:step])
                        
                        temp_buffer_x = np.array(temp_buffer_x)
                        temp_buffer_y = np.array(temp_buffer_y)

                        self.buffer_x, self.buffer_y = temp_buffer_x, temp_buffer_y

                        self.train_xl = np.concatenate((self.train_xl, buffer[0]), axis=0)
                        self.train_yl = np.concatenate((self.train_yl, buffer[1]), axis=0)

            else:
                # load unlabeled data & label
                unlabeled_image_file = self.root + '/Train/Unlabeled/' + args.dataset + '_Images_Task' + str(task) + '_' + args.mode + '.npy'
                unlabeled_file = self.root + '/Train/Unlabeled/' + args.dataset + '_Labels_Task' + str(task) + '_' + args.mode + '.npy'

                train_xul = np.squeeze(np.load(unlabeled_image_file))
                train_yul = np.squeeze(np.load(unlabeled_file))
                self.train_xul = train_xul
                self.train_yul = train_yul

        else:
            # load test data & label
            self.test_x = []
            self.test_y = []

            for task_idx in range(task+1):
                test_image_file = self.root + '/Test/' + args.dataset + '_Images_Task' + str(task_idx) + '_' + args.mode + '.npy'
                test_label_file = self.root + '/Test/' + args.dataset + '_Labels_Task' + str(task_idx) + '_' + args.mode + '.npy'

                test_x = np.squeeze(np.load(test_image_file))
                test_y = np.squeeze(np.load(test_label_file))
                self.test_x.extend(test_x)
                self.test_y.extend(test_y)

            self.test_x = np.array(self.test_x)
            self.test_y = np.array(self.test_y)

    def __len__(self):
        if self.train:
            if self.lab: return len(self.train_xl)
            else: return len(self.train_xul)
        else:
            return len(self.test_x)

    def __getitem__(self, index):
        if self.train:
            if self.lab:
                img, target = 255*self.train_xl[index], self.train_yl[index]
                img = img.astype(np.uint8)
                img = Image.fromarray(img)
                img = self.transform["labeled"][0](img)
                return img, target
            else:
                img, target = 255*self.train_xul[index], self.train_yul[index] 
                img = img.astype(np.uint8)
                img = Image.fromarray(img)
                weak = self.transform["unlabeled"][0](img)
                strong = self.transform["unlabeled"][1](img)
                return weak, strong, target
        else:
            img, target = 255*self.test_x[index], self.test_y[index]
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            img = self.transform['test'][0](img)            
            return img, target


class dataloader():
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset
        self.split_size = 0
        self.buffer_x = []
        self.buffer_y = []

    def load(self, task, train=True):
        if train:
            labeled_dataset = dataset(self.args, task, train, lab=True, buffer=(self.buffer_x, self.buffer_y))
            unlabeled_dataset = dataset(self.args, task, train, lab=False, buffer=False)
            mu = int(unlabeled_dataset.__len__() / labeled_dataset.__len__())
            if mu == 0: mu = 1

            self.buffer_x = labeled_dataset.buffer_x
            self.buffer_y = labeled_dataset.buffer_y

            labeled_trainloader = DataLoader(labeled_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
            unlabeled_trainloader = DataLoader(unlabeled_dataset, batch_size=self.args.batch_size*mu, shuffle=True, num_workers=self.args.num_workers)

            return labeled_trainloader, unlabeled_trainloader

        else:
            test_dataset = dataset(self.args, task, train)
            test_loader = DataLoader(test_dataset, batch_size=self.args.test_size, shuffle=False, num_workers=self.args.num_workers)

            return test_loader