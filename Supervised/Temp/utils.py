import os
import json
import numpy as np

import torch

def get_feature_size(arch):
    if arch == 'resnet18':
        c = 512
    elif arch == 'mobilenet_v3_small':
        c = 576
    elif arch == 'mobilenet_v3_large':
        c = 960
    else:
        raise ValueError('arch not found: ' + arch)
    return c

def bool_flag(s):
    if s == '1' or s == 'True' or s == 'true':
        return True
    elif s == '0' or s == 'False' or s == 'false':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0/1 or True/False or true/false)'
    raise ValueError(msg % s)

def remap_classes(num_classes, seed):
    # get different class permutations

    np.random.seed(seed)
    ix = np.arange(num_classes)
    np.random.shuffle(ix)
    d = {}
    for i, v in enumerate(ix):
        d[i] = v
    return d

def accuracy(output, target, topk=(1,), output_has_class_ids=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    output = output.cpu()
    target = target.cpu()
    if not output_has_class_ids:
        output = torch.Tensor(output)
    else:
        output = torch.LongTensor(output)
    target = torch.LongTensor(target)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = output.shape[0]
        if not output_has_class_ids:
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
        else:
            pred = output[:, :maxk].t()
        correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_accuracies(accuracies, min_class_trained, max_class_trained, save_path, suffix=''):
    name = 'accuracies_min_trained_' + str(min_class_trained) + '_max_trained_' + str(
        max_class_trained) + suffix + '.json'
    json.dump(accuracies, open(os.path.join(save_path, name), 'w'))

def save_predictions(y_pred, min_class_trained, max_class_trained, save_path, suffix=''):
    name = 'preds_min_trained_' + str(min_class_trained) + '_max_trained_' + str(max_class_trained) + suffix
    torch.save(y_pred, save_path + '/' + name + '.pth')

class CMA(object):
    """
    A continual moving average for tracking loss updates.
    """

    def __init__(self):
        self.N = 0
        self.avg = 0.0

    def update(self, X):
        self.avg = (X + self.N * self.avg) / (self.N + 1)
        self.N = self.N + 1


def randint(max_val, num_samples):
    """
    return num_samples random integers in the range(max_val)
    """
    rand_vals = {}
    _num_samples = min(max_val, num_samples)
    while True:
        _rand_vals = np.random.randint(0, max_val, num_samples)
        for r in _rand_vals:
            rand_vals[r] = r
            if len(rand_vals) >= _num_samples:
                break

        if len(rand_vals) >= _num_samples:
            break
    return rand_vals.keys()