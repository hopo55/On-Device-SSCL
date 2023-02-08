import os
import numpy as np
from PIL import Image
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

def get_transform(dataset_name='CIFAR100'):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(dataset_stats[dataset_name]['mean'], dataset_stats[dataset_name]['std']),
        ]
    )
    return transform

class dataset(Dataset):
    def __init__(self, args, task, train=True):
        self.root = os.path.join(args.root, args.dataset)
        self.train = train
        self.transform = get_transform(args.dataset)

        if self.train:
            # load train data & label
            image_file = self.root + '/Train/' + args.dataset + '_Images_Class' + str(task) + '.npy'
            labeled_file = self.root + '/Train/' + args.dataset + '_Labels_Class' + str(task) + '.npy'

            train_x = np.squeeze(np.load(image_file))
            train_y = np.squeeze(np.load(labeled_file))
            self.train_x = train_x
            self.train_y = train_y

        else:
            # load test data & label
            self.test_x = []
            self.test_y = []

            for task_idx in range(task+1):
                test_image_file = self.root + '/Test/' + args.dataset + '_Images_Class' + str(task_idx) + '.npy'
                test_label_file = self.root + '/Test/' + args.dataset + '_Labels_Class' + str(task_idx) + '.npy'

                test_x = np.squeeze(np.load(test_image_file))
                test_y = np.squeeze(np.load(test_label_file))
                self.test_x.extend(test_x)
                self.test_y.extend(test_y)

            self.test_x = np.array(self.test_x)
            self.test_y = np.array(self.test_y)

    def __len__(self):
        if self.train:
            return len(self.train_x)
        else:
            return len(self.test_x)

    def __getitem__(self, index):
        if self.train:
            img, target = 255*self.train_x[index], self.train_y[index]
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target
        else:
            img, target = 255*self.test_x[index], self.test_y[index]
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target


class dataloader():
    def __init__(self, args):
        self.args = args

    def load(self, task, train=True):
        if train:
            train_dataset = dataset(self.args, task, train)
            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

            return train_loader

        else:
            test_dataset = dataset(self.args, task, train)
            test_loader = DataLoader(test_dataset, batch_size=self.args.test_size, shuffle=False, num_workers=self.args.num_workers)

            return test_loader