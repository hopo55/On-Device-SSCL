import os
import sys
import math
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import dataloader
import data_generator
from models import resnet, NCM, DeepNCM, SLDA, Softmax
from metric import AverageMeter, SSCL_logger
from utils import get_feature_size

parser = argparse.ArgumentParser()
# General Settings
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--device_name', type=str, default='cal_05')
# Dataset Settings
parser.add_argument('--root', type=str, default='./data/')
parser.add_argument('--dataset', default='CIFAR10', choices=['MNIST', 'CIFAR10', 'CIFAR100', 'HAR'])
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--test_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--mode', type=str, default='vanilla', choices=['vanilla', 'super'])
# Model Settings
parser.add_argument('--model_name', type=str, default='ResNet18', choices=['ResNet18', 'ImageNet_ResNet', 'mobilenet_v3_small', 'mobilenet_v3_large'])
parser.add_argument("--pre_trained", default=False, action='store_true')
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--lr', '--learning_rate', type=float, default=0.1)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--buffer_size', type=int, default=500, help="size of buffer for replay")
# CL Settings
parser.add_argument('--cl_mode', type=str, default='FC', choices=['FC', 'Fine-tuning', 'SLDA', 'NCM', 'DeepNCM', 'Replay'])
parser.add_argument('--class_increment', type=int, default=1)
# NCM Settings

args = parser.parse_args()


def train(epoch, model, train_loader, criterion, optimizer, classifier=None):
    model.train()

    acc = AverageMeter()
    losses = AverageMeter()

    num_iter = math.ceil(len(train_loader.dataset) / args.batch_size)

    for batch_idx, (x, y) in enumerate(train_loader):
        y = y.type(torch.LongTensor)
        x, y = x.to(args.device), y.to(args.device)

        if classifier is None and args.pre_trained is False:
            logits = model(x) # FC
        
        elif args.cl_mode == 'NCM' or args.cl_mode == 'SLDA':
            feature = model(x)
            classifier.train_(feature, y)
        
        elif args.cl_mode == 'DeepNCM':
            feature = model.forward(x)
            logits = model.predict(feature)

            model.update_means(feature, y)
        
        else:
            classifier.to(args.device) # Fine-tuning
            classifier.train()

            feature = model(x) # Fine-tuning using FC
            logits = classifier(feature)

        if args.cl_mode != 'NCM' and args.cl_mode != 'SLDA':
            loss = criterion(logits, y)
            _, predicted = torch.max(logits, dim=1)

            # Compute Gradient and do SGD step
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            losses.update(loss)

            correct = predicted.eq(y).cpu().sum().item()
            acc.update(correct, len(y))

            sys.stdout.write('\r')
            sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.2f Accuracy: %.2f' % (args.dataset, epoch+1, args.epoch, batch_idx+1, num_iter, loss, acc.avg*100))
            sys.stdout.flush()

    if args.cl_mode == 'NCM' or args.cl_mode == 'SLDA':
        return 0, 0
    else:
        return loss.item(), acc.avg*100

def test(task, model, test_loader, classifier=None):
    acc = AverageMeter()
    sys.stdout.write('\n')

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(args.device), y.to(args.device)

            if classifier is None and args.pre_trained is False:
                output = model(x) # FC

            elif args.cl_mode == 'NCM' or args.cl_mode == 'SLDA':
                feature = model(x)
                output = classifier.evaluate_(feature)
                
            else:
                classifier.to(args.device) # Fine-tuning
                classifier.eval()

                feature = model(x) # Fine-tuning using FC
                output = classifier(feature)

            _, predicted = torch.max(output, dim=1)

            correct = predicted.eq(y).cpu().sum().item()
            acc.update(correct, len(y))

            sys.stdout.write('\r')
            sys.stdout.write("Test | Accuracy (Test Dataset Up to Task-%d): %.2f%%" % (task+1, acc.avg*100))
            sys.stdout.flush()

    return acc.avg*100

def main():
    ## GPU Setup
    torch.cuda.set_device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # Dataset Generator
    if 'CIFAR' in args.dataset:
        data_generator.__dict__['CIFAR_Generator'](args)
        if args.dataset == 'CIFAR10': args.num_classes = 10
        else: args.num_classes = 100

    # Create Model
    model_name = args.model_name
    if args.pre_trained:
        model = resnet.__dict__[args.model_name](device=args.device)
    elif 'ResNet' in model_name:
        model = resnet.__dict__[args.model_name](args.num_classes)
    model.to(args.device)

    # Optimizer and Scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=2e-4)
    criterion = nn.CrossEntropyLoss()

    feature_size = get_feature_size(model_name)

    # Select Classifier
    classifier_name = args.cl_mode
    if classifier_name == 'NCM':
        classifier = NCM.NearestClassMean(feature_size, args.num_classes, device=args.device)
    elif classifier_name == 'DeepNCM':
        model = resnet.ResNet18_NCM(args.num_classes)
        model.to(args.device)
    elif classifier_name == 'SLDA':
        classifier = SLDA.StreamingLDA(feature_size, args.num_classes, device=args.device)
    elif classifier_name == 'Fine-tuning' and args.pre_trained:  
        classifier = Softmax.SoftmaxLayer(feature_size, args.num_classes) # fine-tuning
        optimizer = optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=2e-4)
    else:
        classifier = None # full training

    # For plotting the logs
    logger = SSCL_logger('logs/' + args.dataset + '/' + args.device_name, args.cl_mode)
    log_t = 1
    
    data_loader = dataloader.dataloader(args)
    last_test_acc = 0

    for idx in range(0, args.num_classes, args.class_increment):
        task = [k for k in range(idx, idx+args.class_increment)]
        print('\nTask : ', task)

        train_loader = data_loader.load(task)
        test_loader = data_loader.load(task, train=False)

        best_acc = 0
        for epoch in range(args.epoch):
            loss, train_acc = train(epoch, model, train_loader, criterion, optimizer, classifier)

            if train_acc > best_acc:
                best_acc = train_acc
                logger.result('SSCL Train Epoch Loss/Labeled', loss, epoch)

        logger.result('SSCL Train Accuracy', best_acc, log_t)

        test_acc = test(idx, model, test_loader, classifier)
        logger.result('SSCL Test Accuracy', test_acc, log_t)
        last_test_acc = test_acc

        if args.cl_mode != 'NCM' and args.cl_mode != 'SLDA':
            scheduler.step()
        log_t += 1

    logger.result('Final Test Accuracy', last_test_acc, 1)
    # the average test accuracy over all tasks
    print("\n\nAverage Test Accuracy : %.2f%%" % last_test_acc)

    metric_dict = {'metric': last_test_acc}
    logger.config(config=args, metric_dict=metric_dict)


if __name__ == '__main__':
    main()