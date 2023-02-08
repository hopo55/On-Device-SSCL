from torchvision.datasets import CIFAR10, CIFAR100

def get_cifar10_dataset(dataset_root):
    train_set = CIFAR10(dataset_root, train=True, download=True)
    test_set = CIFAR10(dataset_root, train=False, download=True)

    return train_set, test_set

def get_cifar100_dataset(dataset_root):
    train_set = CIFAR100(dataset_root, train=True, download=True)
    test_set = CIFAR100(dataset_root, train=False, download=True)

    return train_set, test_set
