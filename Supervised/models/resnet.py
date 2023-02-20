"""
Code adapted from https://github.com/facebookresearch/GradientEpisodicMemory
                    &
                  https://github.com/kuangliu/pytorch-cifar
                    &
Supervised Contrastive Replay: Revisiting the Nearest Class Mean Classifier
in Online Class-Incremental Continual Learning
                  https://github.com/RaptorMai/online-continual-learning
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.functional import relu, avg_pool2d

from models import DeepNCM

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

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

class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1d, self).__init__()
        self.conv1 = conv1x1(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
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

class Bottleneck1d(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck1d, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
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
        self.input_channel = 3
        
        self.conv1 = conv3x3(self.input_channel, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)

        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.classifier = nn.Linear(self.feature_size, self.num_classes, bias=bias)

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
        x = self.classifier(x)
        return x

    def forward(self, x):
        out = self.features(x)
        logits = self.logits(out)

        return logits

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias):
        super(ResNet1D, self).__init__()
        self.in_planes = nf
        self.num_classes = num_classes
        self.feature_size = nf * 8 * block.expansion
        self.input_channel = 6
        
        self.conv1 = conv1x1(self.input_channel, nf * 1)
        self.bn1 = nn.BatchNorm1d(nf * 1)

        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.classifier = nn.Linear(self.feature_size, self.num_classes, bias=bias)

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
        x = self.classifier(x)
        return x

    def forward(self, x):
        out = self.features(x)
        logits = self.logits(out)

        return logits

class FeatureExtractorBackbone(nn.Module):
    """
    This PyTorch module allows us to extract features from a backbone network
    given a layer name.
    """

    def __init__(self, model, output_layer_name):
        super(FeatureExtractorBackbone, self).__init__()
        self.model = model
        self.output_layer_name = output_layer_name
        self.output = None  # this will store the layer output
        self.add_hooks(self.model)

    def forward(self, x):
        self.model(x)
        return self.output

    def get_name_to_module(self, model):
        name_to_module = {}
        for m in model.named_modules():
            name_to_module[m[0]] = m[1]
        return name_to_module

    def get_activation(self):
        def hook(model, input, output):
            self.output = output.detach()

        return hook

    def add_hooks(self, model):
        """
        :param model:
        :param outputs: Outputs from layers specified in `output_layer_names`
        will be stored in `output` variable
        :param output_layer_names:
        :return:
        """
        name_to_module = self.get_name_to_module(model)
        name_to_module[self.output_layer_name].register_forward_hook(
            self.get_activation()
        ) 

class ImageNet_ResNet(nn.Module):
    """
    This is a model wrapper to reproduce experiments from the original
    paper of Deep Streaming Linear Discriminant Analysis by using
    a pretrained ResNet model.
    """

    def __init__(self, arch="resnet18", output_layer_name="layer4.1", imagenet_pretrained=True, device="cpu"):
        """Init.
        :param arch: backbone architecture. Default is resnet-18, but others
            can be used by modifying layer for
            feature extraction in ``self.feature_extraction_wrapper``.
        :param imagenet_pretrained: True if initializing backbone with imagenet
            pre-trained weights else False
        :param output_layer_name: name of the layer from feature extractor
        :param device: cpu, gpu or other device
        """
        super(ImageNet_ResNet, self).__init__()

        feat_extractor = (models.__dict__[arch](pretrained=imagenet_pretrained).to(device).eval())
        self.feature_extraction_wrapper = FeatureExtractorBackbone(feat_extractor, output_layer_name).eval()

    @staticmethod
    def pool_feat(features):
        feat_size = features.shape[-1]
        num_channels = features.shape[1]
        features2 = features.permute(0, 2, 3, 1)  # 1 x feat_size x feat_size x
        # num_channels
        features3 = torch.reshape(
            features2, (features.shape[0], feat_size * feat_size, num_channels)
        )
        feat = features3.mean(1)  # mb x num_channels
        return feat

    def forward(self, x):
        """
        :param x: raw x data
        """
        feat = self.feature_extraction_wrapper(x)
        feat = ImageNet_ResNet.pool_feat(feat)
        return feat


class ResNet_NCM(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_NCM, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = DeepNCM.DeepNearestClassMean(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def update_means(self, x,y):
        self.linear.update_means(x,y)

    def predict(self, x):
        out = self.linear(x)
        return out


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

# ResNet-NCM
def ResNet18_DeepNCM(out_dim=10):
    return ResNet_NCM(BasicBlock, [2,2,2,2], out_dim)

# ResNet-HAR
def ResNet18_HAR(out_dim=10, nf=64, bias=True):
    return ResNet1D(BasicBlock1d, [2,2,2,2], out_dim, nf, bias)