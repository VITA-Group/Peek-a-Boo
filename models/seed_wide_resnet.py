import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from .seed_conv import SeedConv2d
import sys
import numpy as np
import math

def conv3x3(in_planes, out_planes, stride=1):
    return SeedConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = SeedConv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = SeedConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                SeedConv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class SeedWide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate=0.0, num_classes=10, init_method='standard'):
        super(SeedWide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        #print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

        self.reset_conv_parameters(init_method)

    def reset_parameters(self, module, init_method="kaiming_uniform") -> None:
        if init_method == "kaiming_constant_signed":
            fan = nn.init._calculate_correct_fan(module.weight, "fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            with torch.no_grad():
                module.weight.data = module.weight.data.sign() * std
        elif init_method == "kaiming_constant_unsigned":
            fan = nn.init._calculate_correct_fan(module.weight, "fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            with torch.no_grad():
                module.weight.data = torch.ones_like(module.weight.data) * std
        elif init_method == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        elif init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
        elif init_method == "kaiming_laplace":
            fan = nn.init._calculate_correct_fan(module.weight, "fan_in")
            gain = nn.init.calculate_gain("relu")
            scale = gain / math.sqrt(2.0 * fan)
            with torch.no_grad():
                new_weight = np.random.laplace(loc=0.0, scale=scale, size=module.weight.shape)
                module.weight.data = module.weight.data.new_tensor(torch.from_numpy(new_weight).clone().detach())
        elif init_method == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        elif init_method == "xavier_constant":
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            with torch.no_grad():
                module.weight.data = module.weight.data.sign() * std
        elif init_method == "standard":
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        else:
            raise ValueError(f"{init_method} is not an initialization option!")

    def reset_conv_parameters(self, init_method="standard") -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self.reset_parameters(m, init_method)
    
    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

def SeedWideResNet28_10(dropout_rate=0, num_classes=10, init_method='standard'):
    return SeedWide_ResNet(28, 10, dropout_rate=dropout_rate, num_classes=num_classes, init_method=init_method)

if __name__ == '__main__':
    net=SeedWide_ResNet(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1,3,32,32)))

    #print(y.size())
