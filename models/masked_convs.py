'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from masked_layers import layers


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layers.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layers.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Conv4(nn.Module):
    def __init__(self, num_classes=10, init_method='standard'):
        super(Conv4, self).__init__()
        self.convs = nn.Sequential(
            conv3x3(3, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            conv3x3(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            conv3x3(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            conv1x1(32 * 32 * 8, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            conv1x1(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            conv1x1(256, num_classes),
        )

        self.reset_conv_parameters(init_method)
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
    
    def get_bop_params(self):
        bop_params = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                bop_params += list(m.parameters())
        return bop_params

    def get_bop_param_masks(self):
        bop_param_masks = []
        for m in self.modules():
            if isinstance(m, layers.Conv2d):
                bop_param_masks.append(m.weight_mask)
        return bop_param_masks

    def get_non_bop_params(self):
        non_bop_params = []
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.BatchNorm2d,)):
                non_bop_params += list(m.parameters())
        return non_bop_params

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 8192, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv6(nn.Module):
    def __init__(self, num_classes=10, init_method='standard'):
        super(Conv6, self).__init__()
        self.convs = nn.Sequential(
            conv3x3(3, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            conv3x3(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            conv3x3(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            conv3x3(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            conv1x1(256 * 4 * 4, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            conv1x1(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            conv1x1(256, num_classes),
        )

        self.reset_conv_parameters(init_method)
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
    
    def get_bop_params(self):
        bop_params = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                bop_params += list(m.parameters())
        return bop_params

    def get_bop_param_masks(self):
        bop_param_masks = []
        for m in self.modules():
            if isinstance(m, layers.Conv2d):
                bop_param_masks.append(m.weight_mask)
        return bop_param_masks

    def get_non_bop_params(self):
        non_bop_params = []
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.BatchNorm2d,)):
                non_bop_params += list(m.parameters())
        return non_bop_params


    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 256 * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = conv3x3(in_planes, planes, stride=stride)  #, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = conv3x3(planes, planes, stride=1)  #, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                conv1x1(in_planes, self.expansion*planes, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = conv3x3(planes, planes, stride=stride)  #, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.conv3 = conv1x1(planes, self.expansion*planes)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                conv1x1(in_planes, self.expansion*planes, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    # def __init__(self, block, num_blocks, num_classes=10, init_method='standard'):
    def __init__(self, block, num_blocks, in_planes=64, num_classes=10, init_method='standard'):
        super(ResNet, self).__init__()
        self.in_planes = in_planes

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = conv3x3(3, self.in_planes, stride=1)  #, padding=1)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        if self.in_planes ==  64:
            # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512*block.expansion, num_classes)
            #self.linear = layers.Linear(512*block.expansion, num_classes)
        elif self.in_planes == 16:
            self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
            self.layer4 = None
            self.linear = nn.Linear(64, num_classes)
            #self.linear = layers.Linear(64, num_classes)

        self.reset_conv_parameters(init_method)
        self.init_linear = torch.clone(self.linear.weight.data).detach()
        #self.init_conv_signs = [m.weight.data.sign() if isinstance(m, nn.Conv2d) for m in self.modules]

    def reset_linear(self) -> None:
        self.linear.weight.data.copy_(self.init_linear)

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
    
    def get_bop_params(self):
        bop_params = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                bop_params += list(m.parameters())
        return bop_params

    def get_bop_param_masks(self):
        bop_param_masks = []
        for m in self.modules():
            if isinstance(m, layers.Conv2d):
                bop_param_masks.append(m.weight_mask)
        return bop_param_masks

    def get_non_bop_params(self):
        non_bop_params = []
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.BatchNorm2d,)):
                non_bop_params += list(m.parameters())
        return non_bop_params

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.layer4 is not None:
            out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet20(num_classes=10, init_method='standard'):
    return ResNet(BasicBlock, [3,3,3], in_planes=16, num_classes=num_classes, init_method=init_method)

def ResNet110(num_classes=10, init_method='standard'):
    return ResNet(BasicBlock, [18,18,18], in_planes=16, num_classes=num_classes, init_method=init_method)

def ResNet18(num_classes=10, init_method='standard'):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, init_method=init_method)

# def ResNet34(input_shape, num_classes, dense_classifier, pretrained, init_method='standard'):
def ResNet34(num_classes=10, init_method='standard'):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, init_method=init_method)

# def ResNet50(input_shape, num_classes, dense_classifier, pretrained, init_method='standard'):
#     return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes,
#                   init_method=init_method)
def ResNet50(num_classes=10, init_method='standard'):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, init_method=init_method)

# def ResNet101(input_shape, num_classes, dense_classifier, pretrained, init_method='standard'):
#     return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes,
#                   init_method=init_method)
def ResNet101(num_classes=10, init_method='standard'):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, init_method=init_method)

# def ResNet152(input_shape, num_classes, dense_classifier, pretrained, init_method='standard'):
#     return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes,
#                   init_method=init_method)
def ResNet152(num_classes=10, init_method='standard'):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, init_method=init_method)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
