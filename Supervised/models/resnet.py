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
    def __init__(self, block, num_blocks, num_classes, nf, bias, ncm=False, device=0):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.ncm = ncm

        self.device = device
        self.num_classes = num_classes
        self.feature_size = nf * 8 * block.expansion
        
        self.prev_muK = torch.zeros((self.num_classes, self.feature_size)).to(self.device)
        self.prev_cK = torch.zeros(self.num_classes).to(self.device)
        self.prev_num_updates = 0

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        if self.ncm:
            self.ncm_classifier = NearestClassMean(self.feature_size, self.num_classes, self.device)
        else:
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

    def ncm_logits(self, x):
        '''Apply the last FC linear mapping to get logits'''
        x = self.ncm_classifier.predict(x)
        return x

    def init_ncm(self, task, start, end):
        if task == 0 and start != end:
            self.ncm_classifier.muK = torch.zeros((self.num_classes, self.feature_size)).to(self.device)
            self.ncm_classifier.cK = torch.zeros(self.num_classes).to(self.device)
            self.ncm_classifier.num_updates = 0
        elif start == end:
            self.prev_muK = self.ncm_classifier.muK
            self.prev_cK = self.ncm_classifier.cK
            self.prev_num_updates = self.ncm_classifier.num_updates
        else:
            self.ncm_classifier.muK = self.prev_muK
            self.ncm_classifier.cK = self.prev_cK
            self.ncm_classifier.num_updates = self.prev_num_updates

    def forward(self, x, y):
        out = self.features(x)
        if self.ncm:
            self.ncm_classifier.fit_batch(out, y)
            logits = self.ncm_classifier.predict(out)
        else:
            logits = self.logits(out)

        return logits
  
class NearestClassMean(nn.Module):
    def __init__(self, input_shape, num_classes, device='cuda'):
        super(NearestClassMean, self).__init__()
        # NCM parameters
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes

        # setup weights for NCM
        self.muK = torch.zeros((num_classes, input_shape)).to(self.device)
        self.cK = torch.zeros(num_classes).to(self.device)
        self.num_updates = 0

    @torch.no_grad()
    def fit(self, x, y):
        x = x.to(self.device)
        y = y.long().to(self.device)

        # make sure things are the right shape
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)

        # update class means
        self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
        self.cK[y] += 1
        self.num_updates += 1

    @torch.no_grad()
    def find_dists(self, A, B):
        """
        Given a matrix of points A, return the indices of the closest points in A to B using L2 distance.
        :param A: N x d matrix of points
        :param B: M x d matrix of points for predictions
        :return: indices of closest points in A
        """
        M, d = B.shape
        with torch.no_grad():
            B = torch.reshape(B, (M, 1, d))  # reshaping for broadcasting
            square_sub = torch.mul(A - B, A - B)  # square all elements
            dist = torch.sum(square_sub, dim=2)
        return -dist # why use minus?

    @torch.no_grad()
    def predict(self, X, probas=False):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead of predictions returned
        :return: the test predictions or probabilities
        """
        X = X.to(self.device)
        scores = self.find_dists(self.muK, X)

        # mask off predictions for unseen classes
        not_visited_ix = torch.where(self.cK == 0)[0]
        min_col = torch.min(scores, dim=1)[0].unsqueeze(0) - 1 # ????
        scores[:, not_visited_ix] = min_col.tile(len(not_visited_ix)).reshape(len(not_visited_ix), len(X)).transpose(1, 0)  # mask off scores for unseen classes

        if not probas:
            return scores
        else:
            return torch.softmax(scores, dim=1)

    @torch.no_grad()
    def fit_batch(self, batch_x, batch_y):
        # fit NCM one example at a time
        for x, y in zip(batch_x, batch_y):
            self.fit(x.cpu(), y.view(1, ))

    @torch.no_grad()
    def predict_batch(self, batch_x, batch_y):
        num_samples = len(batch_x)
        probabilities = torch.empty((num_samples, self.num_classes))
        labels = torch.empty(num_samples).long()
        start = 0

        for batch_x_feat, batch_target in zip(batch_x, batch_y):
            probas = self.predict(batch_x_feat, probas=True)
            end = start + probas.shape[0]
            probabilities[start:end] = probas
            labels[start:end] = batch_target.squeeze()
            start = end

        return probabilities, labels

    @torch.no_grad()
    def train_(self, train_loader):
        for batch_x, batch_y, batch_ix in train_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(batch_x.to(self.device))
            else:
                batch_x_feat = batch_x.to(self.device)

            self.fit_batch(batch_x_feat, batch_y, batch_ix)

    @torch.no_grad()
    def evaluate_(self, test_loader):
        print('\nTesting on %d images.' % len(test_loader.dataset))

        num_samples = len(test_loader.dataset)
        probabilities = torch.empty((num_samples, self.num_classes))
        labels = torch.empty(num_samples).long()
        start = 0
        for test_x, test_y in test_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(test_x.to(self.device))
            else:
                batch_x_feat = test_x.to(self.device)
            probas = self.predict(batch_x_feat, return_probas=True)
            end = start + probas.shape[0]
            probabilities[start:end] = probas
            labels[start:end] = test_y.squeeze()
            start = end
        return probabilities, labels

    def save_model(self, save_path, save_name):
        # grab parameters for saving
        d = dict()
        d['muK'] = self.muK.cpu()
        d['cK'] = self.cK.cpu()
        d['num_updates'] = self.num_updates

        # save model out
        torch.save(d, os.path.join(save_path, save_name + '.pth'))

    def load_model(self, save_file):
        # load parameters
        print('\nloading ckpt from: %s' % save_file)
        d = torch.load(os.path.join(save_file))
        self.muK = d['muK'].to(self.device)
        self.cK = d['cK'].to(self.device)
        self.num_updates = d['num_updates']



'''
See https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''
def ResNet18_NCM(out_dim=10, nf=64, bias=True, device=0):
    return ResNet(BasicBlock, [2, 2, 2, 2], out_dim, nf, bias, ncm=True, device=device)

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