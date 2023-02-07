import numpy as np

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