import os
import autoaugment
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

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
    elif aug_type == 'hard':
        transform_strong = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                autoaugment.__dict__[dataset_name + 'Policy'](),
                transforms.ToTensor(),
                transforms.Normalize(dataset_stats[dataset_name]['mean'], dataset_stats[dataset_name]['std']),
            ]
        )
    else:
        transform_none = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(dataset_stats[dataset_name]['mean'], dataset_stats[dataset_name]['std']),
            ]
        )

class dataset(Dataset):
    def __init__(self, args, task, transform, train=True):
        self.root = os.path.join(args.root, args.dataset)
        self.transform = transform

        if train:
            labeled_image_file = self.root + '/Train/Labeled/' + args.dataset + '_Images_Task' + str(task) + '_' + args.mode + '.npy'
            labeled_file = self.root + '/Train/Labeled/' + args.dataset + '_Labels_Task' + str(task) + '_' + args.mode + '.npy'

            unlabeled_image_file = self.root + '/Train/Unlabeled/' + args.dataset + '_Images_Task' + str(task) + '_' + args.mode + '.npy'
            unlabeled_file = self.root + '/Train/Unlabeled/' + args.dataset + '_Labels_Task' + str(task) + '_' + args.mode + '.npy'

            train_xl = np.squeeze(np.load(labeled_image_file))
            train_yl = np.squeeze(np.load(labeled_file))

            self.train_xl = train_xl
            self.train_yl = train_yl
            print(self.train_yl)

            train_xul = np.squeeze(np.load(unlabeled_image_file))
            train_yul = np.squeeze(np.load(unlabeled_file))

            self.train_xul = train_xul
            self.train_yul = train_yul

        else:
            test_image_file = self.root + '/Test/' + args.dataset + '_Images_Task' + str(task) + '_' + args.mode + '.npy'
            test_label_file = self.root + '/Test/' + args.dataset + '_Labels_Task' + str(task) + '_' + args.mode + '.npy'
            test_x = np.squeeze(np.load(test_image_file))
            test_y = np.squeeze(np.load(test_label_file))

            self.test_x = test_x
            self.test_y = test_y


class dataloader():
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset
        self.transforms = get_transform(self.dataset_name)
        self.transforms_test = get_transform(self.dataset_name)

    def load(self, task, train=True):
        if train:
            labeled_dataset = dataset(self.args, task, self.transforms, train)
            unlabeled_dataset = dataset(self.args, train)

            labeled_trainloader = DataLoader(labeled_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True)
            unlabeled_trainloader = DataLoader(unlabeled_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True)

            return labeled_trainloader, unlabeled_trainloader

        else:
            test_dataset = dataset(self.args, task, self.transforms_test, train)
            test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

            return test_loader