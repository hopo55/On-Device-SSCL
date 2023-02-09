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
from models import resnet
from metric import AverageMeter, SSCL_logger

parser = argparse.ArgumentParser()
# General Settings
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--device_name', type=str, default='cal_05')
# Dataset Settings
parser.add_argument('--root', type=str, default='./data/')
parser.add_argument('--dataset', default='CIFAR10', choices=['MNIST', 'CIFAR10', 'CIFAR100'])
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--test_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--mode', type=str, default='vanilla', choices=['vanilla', 'super'])
# Model Settings
parser.add_argument('--model_name', type=str, default='ResNet18', choices=['ResNet18', 'Reduced_ResNet18', 'mobilenet_v3_small', 'mobilenet_v3_large'])
parser.add_argument('--classifier', type=str, default='FC', choices=['FC', 'NCM', 'SLDA'])
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--lr', '--learning_rate', type=float, default=0.1)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--buffer_size', type=int, default=500, help="size of buffer for replay")
# CL Settings
parser.add_argument('--cl_mode', type=str, default='Fine-tune', choices=['Fine-tune', 'SLDA', 'NCM', 'Replay'])
parser.add_argument('--class_increment', type=int, default=1)
# NCM Settings

args = parser.parse_args()

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()

    acc = AverageMeter()
    losses = AverageMeter()

    num_iter = math.ceil(len(train_loader.dataset) / args.batch_size)

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(args.device), y.to(args.device)

        # Labeled samples training
        logits = model(x)
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

    return loss.item(), acc.avg*100

def test(task, model, test_loader):
    acc = AverageMeter()
    sys.stdout.write('\n')

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(args.device), y.to(args.device)

            output = model(x)
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

    root = os.path.join(args.root, args.dataset)

    # Create Model
    model_name = args.model_name
    if 'ResNet' in model_name:
        model = resnet.__dict__[args.model_name](args.num_classes)
    model.to(args.device)

    classifier_name = args.classifier
    if classifier_name == 'NCM':
        pass

    # Optimizer and Scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=2e-4)

    criterion = nn.CrossEntropyLoss()

    # For plotting the logs
    logger = SSCL_logger('logs/' + args.dataset + '/' + args.device_name, args.cl_mode)

    logger.config(config=args)
    log_t = 1
    
    data_loader = dataloader.dataloader(args)
    avg_test_acc = AverageMeter()

    for idx in range(0, args.num_classes, args.class_increment):
        task = [k for k in range(idx, idx+args.class_increment)]
        print('\nTask : ', task)

        train_loader = data_loader.load(task)
        test_loader = data_loader.load(task, train=False)

        best_acc = 0
        for epoch in range(args.epoch):
            loss, train_acc = train(epoch, model, train_loader, criterion, optimizer)

            if train_acc > best_acc:
                best_acc = train_acc
                logger.result('SSCL Train Epoch Loss/Labeled', loss, epoch)

        logger.result('SSCL Train Accuracy', best_acc, log_t)

        test_acc = test(idx, model, test_loader)
        avg_test_acc.update(test_acc)
        logger.result('SSCL Test Accuracy', test_acc, log_t)

        scheduler.step()
        log_t += 1

    logger.result('SSCL Average Test Accuracy', avg_test_acc.avg, 1)
    # the average test accuracy over all tasks
    print("\n\nAverage Test Accuracy : %.2f%%" % avg_test_acc.avg)


if __name__ == '__main__':
    main()