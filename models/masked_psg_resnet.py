'''ResNet using PSG in PyTorch.

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

from models.masked_psg_conv import PredictiveConv2d
from masked_layers.layers import Conv2d


# Fixed
NUM_BITS = 32
NUM_BITS_WEIGHT = 32
NUM_BITS_GRAD = None

BIPRECISION = False
PREDICTIVE_FORWARD = False

WRITER = None
WRITER_PREFIX_COUNTER = 0

# Tunable
PREDICTIVE_BACKWARD = True

MSB_BITS = 4
MSB_BITS_WEIGHT = 4
MSB_BITS_GRAD = 8

THRESHOLD = 0.0
SPARSIFY = False
SIGN = True



def conv1x1(in_planes, out_planes, stride=1, input_signed=True, predictive_forward=True, writer_prefix=""):
    "1x1 convolution with no padding"
    predictive_forward = PREDICTIVE_FORWARD and predictive_forward
    return PredictiveConv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False,
        num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD,
        biprecision=BIPRECISION, input_signed=input_signed,
        predictive_forward=predictive_forward, predictive_backward=PREDICTIVE_BACKWARD,
        msb_bits=MSB_BITS, msb_bits_weight=MSB_BITS_WEIGHT, msb_bits_grad=MSB_BITS_GRAD,
        threshold=THRESHOLD, sparsify=SPARSIFY, sign=SIGN,
        writer=WRITER, writer_prefix=writer_prefix)


def conv3x3(in_planes, out_planes, stride=1, input_signed=False, predictive_forward=True, writer_prefix=""):
    "3x3 convolution with padding"
    predictive_forward = PREDICTIVE_FORWARD and predictive_forward
    return PredictiveConv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
        num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD,
        biprecision=BIPRECISION, input_signed=input_signed,
        predictive_forward=predictive_forward, predictive_backward=PREDICTIVE_BACKWARD,
        msb_bits=MSB_BITS, msb_bits_weight=MSB_BITS_WEIGHT, msb_bits_grad=MSB_BITS_GRAD,
        threshold=THRESHOLD, sparsify=SPARSIFY, sign=SIGN,
        writer=WRITER, writer_prefix=writer_prefix)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride=stride, input_signed=False, predictive_forward=False, writer_prefix=None)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1, input_signed=False, predictive_forward=False, writer_prefix=None)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                conv1x1(in_planes, self.expansion*planes, stride=stride, input_signed=False, predictive_forward=False, writer_prefix=None),
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
        self.conv1 = conv1x1(in_planes, planes, stride=1, input_signed=False, predictive_forward=False, writer_prefix=None)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = conv3x3(planes, planes, stride=stride, input_signed=False, predictive_forward=False, writer_prefix=None)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.conv3 = conv1x1(planes, self.expansion*planes, stride=1, input_signed=False, predictive_forward=False, writer_prefix=None)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                conv1x1(in_planes, self.expansion*planes, stride=stride, input_signed=False, predictive_forward=False, writer_prefix=None),
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
    def __init__(self, block, num_blocks, in_planes=64, num_classes=10, init_method='standard'):
        super(ResNet, self).__init__()
        self.in_planes = in_planes

        self.conv1 = conv3x3(3, self.in_planes, stride=1, input_signed=True, predictive_forward=False, writer_prefix=None)
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

    def get_bop_params(self):
        bop_params = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                bop_params += list(m.parameters())
        return bop_params

    def get_bop_param_masks(self):
        bop_param_masks = []
        for m in self.modules():
            if isinstance(m, Conv2d):
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


def PsgResNet20(
    num_classes=10,
    init_method='standard',
    predictive_backward=True,
    msb_bits=4,
    msb_bits_weight=4,
    msb_bits_grad=8,
    threshold=0.0,
    sparsify=False,
    sign=True
):
    global PREDICTIVE_BACKWARD, MSB_BITS, MSB_BITS_WEIGHT, MSB_BITS_GRAD, THRESHOLD, SPARSIFY, SIGN
    PREDICTIVE_BACKWARD = predictive_backward
    MSB_BITS = msb_bits
    MSB_BITS_WEIGHT = msb_bits_weight
    MSB_BITS_GRAD = msb_bits_grad
    THRESHOLD = threshold
    SPARSIFY = sparsify
    SIGN = sign
    return ResNet(BasicBlock, [3,3,3], in_planes=16, num_classes=num_classes, init_method=init_method)


def PsgResNet110(
    num_classes=10,
    init_method='standard',
    predictive_backward=True,
    msb_bits=4,
    msb_bits_weight=4,
    msb_bits_grad=8,
    threshold=0.0,
    sparsify=False,
    sign=True
):
    global PREDICTIVE_BACKWARD, MSB_BITS, MSB_BITS_WEIGHT, MSB_BITS_GRAD, THRESHOLD, SPARSIFY, SIGN
    PREDICTIVE_BACKWARD = predictive_backward
    MSB_BITS = msb_bits
    MSB_BITS_WEIGHT = msb_bits_weight
    MSB_BITS_GRAD = msb_bits_grad
    THRESHOLD = threshold
    SPARSIFY = sparsify
    SIGN = sign
    return ResNet(BasicBlock, [18,18,18], in_planes=16, num_classes=num_classes, init_method=init_method)


def PsgResNet18(
    num_classes=10,
    init_method='standard',
    predictive_backward=True,
    msb_bits=4,
    msb_bits_weight=4,
    msb_bits_grad=8,
    threshold=0.0,
    sparsify=False,
    sign=True
):
    global PREDICTIVE_BACKWARD, MSB_BITS, MSB_BITS_WEIGHT, MSB_BITS_GRAD, THRESHOLD, SPARSIFY, SIGN
    PREDICTIVE_BACKWARD = predictive_backward
    MSB_BITS = msb_bits
    MSB_BITS_WEIGHT = msb_bits_weight
    MSB_BITS_GRAD = msb_bits_grad
    THRESHOLD = threshold
    SPARSIFY = sparsify
    SIGN = sign
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, init_method=init_method)

def PsgResNet34(
    num_classes=10,
    init_method='standard',
    predictive_backward=True,
    msb_bits=4,
    msb_bits_weight=4,
    msb_bits_grad=8,
    threshold=0.0,
    sparsify=False,
    sign=True
):
    global PREDICTIVE_BACKWARD, MSB_BITS, MSB_BITS_WEIGHT, MSB_BITS_GRAD, THRESHOLD, SPARSIFY, SIGN
    PREDICTIVE_BACKWARD = predictive_backward
    MSB_BITS = msb_bits
    MSB_BITS_WEIGHT = msb_bits_weight
    MSB_BITS_GRAD = msb_bits_grad
    THRESHOLD = threshold
    SPARSIFY = sparsify
    SIGN = sign
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, init_method=init_method)

def PsgResNet50(
    num_classes=10,
    init_method='standard',
    predictive_backward=True,
    msb_bits=4,
    msb_bits_weight=4,
    msb_bits_grad=8,
    threshold=0.0,
    sparsify=False,
    sign=True
):
    global PREDICTIVE_BACKWARD, MSB_BITS, MSB_BITS_WEIGHT, MSB_BITS_GRAD, THRESHOLD, SPARSIFY, SIGN
    PREDICTIVE_BACKWARD = predictive_backward
    MSB_BITS = msb_bits
    MSB_BITS_WEIGHT = msb_bits_weight
    MSB_BITS_GRAD = msb_bits_grad
    THRESHOLD = threshold
    SPARSIFY = sparsify
    SIGN = sign
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, init_method=init_method)

def PsgResNet101(
    num_classes=10,
    init_method='standard',
    predictive_backward=True,
    msb_bits=4,
    msb_bits_weight=4,
    msb_bits_grad=8,
    threshold=0.0,
    sparsify=False,
    sign=True
):
    global PREDICTIVE_BACKWARD, MSB_BITS, MSB_BITS_WEIGHT, MSB_BITS_GRAD, THRESHOLD, SPARSIFY, SIGN
    PREDICTIVE_BACKWARD = predictive_backward
    MSB_BITS = msb_bits
    MSB_BITS_WEIGHT = msb_bits_weight
    MSB_BITS_GRAD = msb_bits_grad
    THRESHOLD = threshold
    SPARSIFY = sparsify
    SIGN = sign
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, init_method=init_method)

def PsgResNet152(
    num_classes=10,
    init_method='standard',
    predictive_backward=True,
    msb_bits=4,
    msb_bits_weight=4,
    msb_bits_grad=8,
    threshold=0.0,
    sparsify=False,
    sign=True
):
    global PREDICTIVE_BACKWARD, MSB_BITS, MSB_BITS_WEIGHT, MSB_BITS_GRAD, THRESHOLD, SPARSIFY, SIGN
    PREDICTIVE_BACKWARD = predictive_backward
    MSB_BITS = msb_bits
    MSB_BITS_WEIGHT = msb_bits_weight
    MSB_BITS_GRAD = msb_bits_grad
    THRESHOLD = threshold
    SPARSIFY = sparsify
    SIGN = sign
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, init_method=init_method)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
