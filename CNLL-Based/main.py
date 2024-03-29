import os
import sys
import math
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import sampling
import dataloader
import data_generator
from models import resnet
from losses import SupConLoss
from metric import AverageMeter, SSCL_logger

parser = argparse.ArgumentParser()
# General Settings
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--device_name', type=str, default='cal_05')
# Dataset Settings
parser.add_argument('--root', type=str, default='./data/')
# parser.add_argument('--dataset', default='CIFAR10')
parser.add_argument('--dataset', default='CIFAR100')
# parser.add_argument('--mode', type=str, default='vanilla', help="vanilla|super")
parser.add_argument('--mode', type=str, default='super', help="vanilla|super")
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--label_ratio', type=float, default=0.2, help="Labeled data ratio")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=0)
# Model Settings
# parser.add_argument('--model_name', type=str, default='ResNet18')
# parser.add_argument('--model_name', type=str, default='Reduced_ResNet18')
parser.add_argument('--model_name', type=str, default='ResNet18_NCM')
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--lr', '--learning_rate', type=float, default=0.1)
parser.add_argument('--lambda_u', type=float, default=1., help='penalize the unlabeled loss')
# parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--num_class', type=int, default=100)
parser.add_argument('--sampling', type=str, default='Random')
parser.add_argument('--buffer_size', type=int, default=1000, help="size of buffer for replay")
# NCM Settings

args = parser.parse_args()

def train(epoch, task, model, labeled_trainloader, unlabeled_trainloader, criterion, optimizer):
    model.train()

    acc = AverageMeter()
    losses = AverageMeter()
    losses_ul = AverageMeter()
    losses_total = AverageMeter()

    num_iter = math.ceil(len(labeled_trainloader.dataset) / args.batch_size)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    for batch_idx, (xl, y) in enumerate(labeled_trainloader):
        xl, y = xl.to(args.device), y.to(args.device)

        x_weak, x_strong, _ = unlabeled_train_iter.next()
        x_weak, x_strong = x_weak.to(args.device), x_strong.to(args.device)

        # Labeled samples training
        l_logits = model(xl, y)
        l_loss = criterion(l_logits, y)
        _, predicted = torch.max(l_logits, dim=1)

        if 'NCM' in args.model_name:
            # Pseudo-Label
            weak_feature = model.features(x_weak)
            weak_logits = model.ncm_logits(weak_feature)
            prob_ul = torch.softmax(weak_logits, dim=1)
            max_probs, pseudo = torch.max(prob_ul, dim=1)
            mask = max_probs.ge(0.5).float()
            
            # Unlabeled samples training
            strong_feature = model.features(x_strong)
            strong_logits = model.ncm_logits(strong_feature)

        else:
            # Pseudo-Label
            weak_feature = model.features(x_weak)
            weak_logits = model.logits(weak_feature)
            prob_ul = torch.softmax(weak_logits, dim=1)
            max_probs, pseudo = torch.max(prob_ul, dim=1)
            mask = max_probs.ge(0.5).float()

            # Unlabeled samples training
            strong_feature = model.features(x_strong)
            strong_logits = model.logits(strong_feature)

        ul_loss = F.cross_entropy(strong_logits, pseudo, reduction='none') * mask
        ul_loss = ul_loss.mean()

        total_loss = l_loss + (args.lambda_u * ul_loss)

        # Compute Gradient and do SGD step
        optimizer.zero_grad()
        total_loss.requires_grad_(True)
        total_loss.backward()
        optimizer.step()

        losses.update(l_loss)
        losses_ul.update(ul_loss)
        losses_total.update(total_loss)

        correct = predicted.eq(y).cpu().sum().item()
        acc.update(correct, len(y))

        sys.stdout.write('\r')
        sys.stdout.write('%s-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f Total Loss: %.4f  Accuracy: %.2f' % (args.dataset, args.mode, epoch+1, args.epoch, batch_idx+1, num_iter, l_loss, ul_loss, total_loss, acc.avg*100))
        sys.stdout.flush()

    if 'NCM' in args.model_name:
        model.init_ncm(task, epoch+1, args.epoch)

    return l_loss.item(), ul_loss.item(), total_loss.item(), acc.avg*100

def test(task, model, test_loader):
    acc = AverageMeter()
    sys.stdout.write('\n')

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(args.device), y.to(args.device)

            output = model(x, y)
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
    cudnn.benchmark = True

    # Dataset Generator
    if 'CIFAR' in args.dataset:
        data_generator.__dict__['CIFAR_Generator'](args)
    root = os.path.join(args.root, args.dataset)

    # Create Model
    model_name = args.model_name
    if 'NCM' in model_name:
        model = resnet.__dict__[args.model_name](args.num_class, device=args.device)
    else:
        model = resnet.__dict__[args.model_name](args.num_class)
    model.to(args.device)

    criterion = nn.CrossEntropyLoss()

    # Optimizer and Scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=2e-4)

    # For plotting the logs
    sscl_logger = SSCL_logger('logs/' + args.dataset + '/sscl_logs_' + args.device_name + '_')

    sscl_logger.config(config=args)

    if args.dataset == 'CIFAR10':
        task_mode_list = ['Task-1', 'Task-2', 'Task-3', 'Task-4', 'Task-5']
    elif args.dataset == 'CIFAR100':
        task_mode_list = ['Task-1', 'Task-2', 'Task-3', 'Task-4', 'Task-5', 'Task-6', 'Task-7', 'Task-8', 'Task-9', 'Task-10', 'Task-11', 'Task-12', 'Task-13', 'Task-14', 'Task-15', 'Task-16', 'Task-17', 'Task-18', 'Task-19', 'Task-20']
    
    data_loader = dataloader.dataloader(args)
    
    avg_test_acc = AverageMeter()

    for t, task_mode in enumerate(task_mode_list):
        labeled_trainloader, unlabeled_trainloader = data_loader.load(t)
        test_loader = data_loader.load(t, train=False)

        label_file = root + '/Train/Labeled/' + args.dataset + '_Labels_Task' + str(t) + '_' + args.mode + '.npy'
        train_label = np.squeeze(np.load(label_file))
        class_name  = np.unique(train_label)
        num_samples = np.shape(train_label)[0]

        print('\n', task_mode)
        print("Number of Samples:", num_samples, class_name)

        weight = torch.zeros(args.num_class)
        weight[class_name] = 1
        weight = weight.cuda()
        criterion = nn.CrossEntropyLoss(weight = weight)

        best_acc = 0
        for epoch in range(args.epoch):
            l_loss, ul_loss, total_loss, train_acc = train(epoch, t, model, labeled_trainloader, unlabeled_trainloader, criterion, optimizer)

            if train_acc > best_acc:
                best_acc = train_acc
                sscl_logger.result('SSCL Train Epoch Loss/Labeled', l_loss, epoch)
                sscl_logger.result('SSCL Train Epoch Loss/Unlabeled', ul_loss, epoch)
                sscl_logger.result('SSCL Train Epoch Loss/Total', total_loss, epoch)

        sscl_logger.result('SSCL Train Accuracy', best_acc, t+1)

        test_acc = test(t, model, test_loader)
        avg_test_acc.update(test_acc)
        sscl_logger.result('SSCL Test Accuracy', test_acc, t+1)

        scheduler.step()

    sscl_logger.result('SSCL Average Test Accuracy', avg_test_acc.avg, 1)
    # the average test accuracy over all tasks
    print("\n\nAverage Test Accuracy : %.2f%%" % avg_test_acc.avg)


if __name__ == '__main__':
    main()