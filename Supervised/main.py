import random
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser()
# General Settings
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=int, default=0)

args = parser.parse_args()

def main():
    ## GPU Setup
    torch.cuda.set_device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

if __name__ == '__main__':
    main()