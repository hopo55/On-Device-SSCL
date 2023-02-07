import random
import argparse
import numpy as np
from utils import *

import torch
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser()
# General Settings
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=int, default=0)
# Model Settings
parser.add_argument('--arch', type=str, choices=['resnet18', 'mobilenet_v3_small', 'mobilenet_v3_large'])

args = parser.parse_args()

args.input_feature_size = get_feature_size(args.arch)

def main():
    ## GPU Setup
    torch.cuda.set_device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # setup continual model
    print('\nUsing the %s continual learning model.' % args.cl_model)
    if args.cl_model == 'slda':
        classifier = StreamingLDA(args.input_feature_size, args.num_classes,
                                  shrinkage_param=args.shrinkage, streaming_update_sigma=args.streaming_update_sigma)
    elif args.cl_model == 'fine_tune':
        classifier = StreamingSoftmax(args.input_feature_size, args.num_classes, use_replay=False,
                                      lr=args.lr, weight_decay=args.wd)
        if args.ckpt is not None:
            classifier.load_model(args.ckpt)
    elif args.cl_model == 'replay': # using this for replay
        classifier = StreamingSoftmax(args.input_feature_size, args.num_classes, use_replay=True,
                                      lr=args.lr, weight_decay=args.wd, replay_samples=args.replay_size,
                                      max_buffer_size=args.buffer_size)
        if args.ckpt is not None:
            classifier.load_model(args.ckpt)
    elif args.cl_model == 'ncm':
        classifier = NearestClassMean(args.input_feature_size, args.num_classes)
        if args.ckpt is not None:
            classifier.load_model(args.ckpt)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()