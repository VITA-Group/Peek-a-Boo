'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import math

# from models.masked_psg_conv import PredictiveConv2d
from models.masked_psg_seed_conv import PredictiveSeedConv2d
from masked_layers.layers import Conv2d, Linear


cfg = {
    'PsgSeedVGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'PsgSeedVGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'PsgSeedVGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'PsgSeedVGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


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
    return PredictiveSeedConv2d(
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
    return PredictiveSeedConv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
        num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD,
        biprecision=BIPRECISION, input_signed=input_signed,
        predictive_forward=predictive_forward, predictive_backward=PREDICTIVE_BACKWARD,
        msb_bits=MSB_BITS, msb_bits_weight=MSB_BITS_WEIGHT, msb_bits_grad=MSB_BITS_GRAD,
        threshold=THRESHOLD, sparsify=SPARSIFY, sign=SIGN,
        writer=WRITER, writer_prefix=writer_prefix)


class PsgSeedVGG(nn.Module):
    def __init__(
        self,
        vgg_name,
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
        # Configure PSG
        global PREDICTIVE_BACKWARD, MSB_BITS, MSB_BITS_WEIGHT, MSB_BITS_GRAD, THRESHOLD, SPARSIFY, SIGN
        PREDICTIVE_BACKWARD = predictive_backward
        MSB_BITS = msb_bits
        MSB_BITS_WEIGHT = msb_bits_weight
        MSB_BITS_GRAD = msb_bits_grad
        THRESHOLD = threshold
        SPARSIFY = sparsify
        SIGN = sign

        super(PsgSeedVGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = Linear(512, num_classes)
        self.reset_conv_parameters(init_method)

    def reset_parameters(self, module, init_method="kaiming_uniform") -> None:
        # Bias initialization
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(module.bias, -bound, bound)

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

    def get_non_bop_params(self):
        non_bop_params = []
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.BatchNorm2d,)):
                non_bop_params += list(m.parameters())
        return non_bop_params

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        input_layer = True
        layers = []
        in_channels = 3
        for x in cfg:
            if input_layer:
                input_signed = True
                input_layer = False
            else:
                input_signed = False
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    # Conv2d(in_channels, x, kernel_size=3, padding=1),
                    conv3x3(in_channels, x, stride=1,
                            input_signed = input_signed,
                            predictive_forward = False,
                            writer_prefix = None),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()

