"""
Code adapted from https://github.com/facebookresearch/GradientEpisodicMemory
                    &
                  https://github.com/kuangliu/pytorch-cifar
                    &
Supervised Contrastive Replay: Revisiting the Nearest Class Mean Classifier
in Online Class-Incremental Continual Learning
                  https://github.com/RaptorMai/online-continual-learning
"""
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.num_classes = num_classes
        self.feature_size = nf * 8 * block.expansion
        
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)

        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.last = nn.Linear(self.feature_size, self.num_classes, bias=bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def logits(self, x):
        '''Apply the last FC linear mapping to get logits'''
        x = self.last(x)
        return x

    def forward(self, x):
        out = self.features(x)
        logits = self.logits(out)

        return logits
 

'''
See https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''
# Reduced ResNet18 as in GEM MIR(note that nf=20).
def Reduced_ResNet18(out_dim=10, nf=20, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], out_dim, nf, bias)

def ResNet18(out_dim=10, nf=64, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], out_dim, nf, bias)

def ResNet34(out_dim=10, nf=64, bias=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], out_dim, nf, bias)

def ResNet50(out_dim=10, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], out_dim, nf, bias)

def ResNet101(out_dim=10, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], out_dim, nf, bias)

def ResNet152(out_dim=10, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], out_dim, nf, bias)