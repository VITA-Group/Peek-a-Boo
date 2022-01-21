'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .seed_conv import SeedConv2d
from masked_layers import layers

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class SeedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, sign_grouped_dim=(), init_method='standard', hidden_act='none', scaling_input=False):
        super(SeedBasicBlock, self).__init__()
        self.conv1 = SeedConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                                sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SeedConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
                                sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                SeedConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False,
                           sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SeedBasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, sign_grouped_dim=(), init_method='standard', hidden_act='none', scaling_input=False):
        super(SeedBasicBlock2, self).__init__()
        self.conv1 = SeedConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                                sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SeedConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
                                sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2,  ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SeedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, sign_grouped_dim=(), init_method='standard', hidden_act='none', scaling_input=False):
        super(SeedBottleneck, self).__init__()
        self.conv1 = SeedConv2d(in_planes, planes, kernel_size=1, bias=False,
                                sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SeedConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                                sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SeedConv2d(planes, self.expansion*planes, kernel_size=1, bias=False,
                                sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                SeedConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False,
                           sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SeedResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, sign_grouped_dim=(), init_method='standard', hidden_act='none', scaling_input=False):
        super(SeedResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = SeedConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False,
                                sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,
                                       sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                                       sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                                       sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                                       sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, sign_grouped_dim, init_method, hidden_act, scaling_input):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, sign_grouped_dim, init_method, hidden_act, scaling_input=scaling_input))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class SeedResNetCifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, sign_grouped_dim=(), init_method='standard', hidden_act='none', scaling_input=False):
        super(SeedResNetCifar, self).__init__()
        self.in_planes = 16

        self.conv1 = SeedConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False,
                                sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1,
                                       sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2,
                                       sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2,
                                       sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, sign_grouped_dim, init_method, hidden_act, scaling_input):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, sign_grouped_dim, init_method, hidden_act, scaling_input=scaling_input))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SeedResNet18(sign_grouped_dim=(), init_method='standard', hidden_act='none', scaling_input=False, num_classes=10):
    return SeedResNet(SeedBasicBlock, [2,2,2,2], sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input, num_classes=num_classes)

def SeedResNet20(sign_grouped_dim=(), init_method='standard', hidden_act='none', scaling_input=False, num_classes=10):
    return SeedResNetCifar(SeedBasicBlock2, [3, 3, 3], sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input, num_classes=num_classes)

def SeedResNet34(sign_grouped_dim=(), init_method='standard', hidden_act='none', scaling_input=False, num_classes=10):
    return SeedResNet(SeedBasicBlock, [3,4,6,3], sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input, num_classes=num_classes)

def SeedResNet50(sign_grouped_dim=(), init_method='standard', hidden_act='none', scaling_input=False, num_classes=10):
    return SeedResNet(SeedBottleneck, [3,4,6,3], sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input, num_classes=num_classes)

def SeedResNet101(sign_grouped_dim=(), init_method='standard', hidden_act='none', scaling_input=False, num_classes=10):
    return SeedResNet(SeedBottleneck, [3,4,23,3], sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input, num_classes=num_classes)

def SeedResNet152(sign_grouped_dim=(), init_method='standard', hidden_act='none', scaling_input=False, num_classes=10):
    return SeedResNet(SeedBottleneck, [3,8,36,3], sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act, scaling_input=scaling_input, num_classes=num_classes)


def test():
    net = SeedResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
