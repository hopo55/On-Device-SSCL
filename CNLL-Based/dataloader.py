import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import autoaugment

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
    def __init__(self, dataset_name, train=True):

        if train:
            pass
        else:
            pass

class dataloader():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.transforms = get_transform(self.dataset_name)
        self.transforms_test = get_transform(self.dataset_name)
