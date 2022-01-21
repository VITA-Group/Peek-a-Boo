"""
author: Xiaohan Chen
date: 2020-08-26
"""

import math
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


def ste_sign(x):
    detach = x.detach()
    return x - detach + torch.sign(detach)


def ste_onesided_sign(x):
    detach = x.detach()
    return x - detach + torch.sign(F.relu(detach))


def ste_relu(x):
    detach = x.detach()
    return x - detach + F.relu(detach)


def ste_ternery_with_lo_hi(x, lo, hi):
    return ste_sign(ste_relu(x - hi)) - ste_sign(ste_relu(lo - x))


def ste_ternery_with_lo_hi_ver2(x, lo, hi):
    return ste_sign(F.relu(x - hi)) - ste_sign(F.relu(lo - x))


def all_ones(x):
    return torch.ones_like(x)


HIDDEN_ACT_DICT = {
    'pruning': ste_onesided_sign,
    'flipping': ste_sign,
    'ternary': ste_ternery_with_lo_hi_ver2,
    'none': all_ones
}


class SeedConv2d(nn.Conv2d):
    """docstring for SeedConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 sign_grouped_dim=(), init_method='standard', hidden_act='none',
                 scaling_input=False):
        super(SeedConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.hidden_act = hidden_act
        self.hidden_act_func = HIDDEN_ACT_DICT[self.hidden_act]
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

        # self.sign_grouped_dim = sign_grouped_dim
        sign_dim = list(self.weight.shape)
        for idx in sign_grouped_dim:
            sign_dim[idx] = 1
        
        if self.hidden_act in ['pruning', 'flipping']:
            self.t = nn.Parameter(torch.rand(sign_dim, dtype=torch.float32) * 0.09 + 0.01)
        elif self.hidden_act == 'ternary':
            self.t = nn.Parameter(torch.rand(sign_dim, dtype=torch.float32) * 2.0 - 1.0)
        else:
            self.t = nn.Parameter(torch.rand(sign_dim, dtype=torch.float32))

        if 'ternary' in self.hidden_act:
            self.lo = nn.Parameter(torch.ones(1, dtype=torch.float32) * (-0.1))
            self.hi = nn.Parameter(torch.ones(1, dtype=torch.float32) * 0.1)
        
        self.scaling_input = scaling_input
        if self.scaling_input:
            self.scale = nn.Parameter(torch.ones((1, in_channels, 1, 1), dtype=torch.float32))

        # Initialize parameters
        self.reset_parameters(init_method)

        # Disable the learning of the weight and bias
        self.disable_weight_grad()

    def reset_parameters(self, init_method="kaiming_uniform") -> None:
        # Bias initialization
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

        if init_method == "kaiming_constant_signed":
            fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            with torch.no_grad():
                self.weight.data = self.weight.data.sign() * std
        elif init_method == "kaiming_constant_unsigned":
            fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            with torch.no_grad():
                self.weight.data = torch.ones_like(self.weight.data) * std
        elif init_method == "kaiming_normal":
            nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        elif init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.weight, mode="fan_in", nonlinearity="relu")
        elif init_method == "kaiming_laplace":
            fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
            gain = nn.init.calculate_gain("relu")
            scale = gain / math.sqrt(2.0 * fan)
            with torch.no_grad():
                new_weight = np.random.laplace(loc=0.0, scale=scale, size=self.weight.shape)
                self.weight.data = self.weight.data.new_tensor(torch.from_numpy(new_weight).clone().detach())
        elif init_method == "xavier_normal":
            nn.init.xavier_normal_(self.weight)
        elif init_method == "xavier_constant":
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            with torch.no_grad():
                self.weight.data = self.weight.data.sign() * std
        elif init_method == "standard":
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            raise ValueError(f"{init_method} is not an initialization option!")

    def disable_weight_grad(self):
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def enable_weight_grad(self):
        self.weight.requires_grad_(True)
        if self.bias is not None:
            self.bias.requires_grad_(True)

    def forward(self, input):
        temp_input = (self.t, self.lo, self.hi,) if 'ternary' in self.hidden_act else (self.t,)
        mask = self.hidden_act_func(*temp_input)
        masked_weight = mask * self.weight

        if self.scaling_input:
            input = input * self.scale

        # pruning mask
        W = self.weight_mask * masked_weight
        if self.bias is not None:
            b = self.bias_mask * self.bias
        else:
            b = self.bias

        output = F.conv2d(input, W, b, self.stride,
                          self.padding, self.dilation, self.groups)
        return output


# class QLinear(nn.Linear):
#     """docstring for QConv2d."""

#     def __init__(self, in_features, out_features, bias=True,
#                  num_bits=8, num_bits_weight=8):
#         super(QLinear, self).__init__(in_features, out_features, bias)
#         self.num_bits = num_bits
#         self.num_bits_weight = num_bits_weight or num_bits
#         self.quantize_input = QuantMeasure(self.num_bits)

#     def forward(self, input, ret_qinput=False):
#         qinput = self.quantize_input(input)
#         weight_qparams = calculate_qparams(
#             self.weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
#         qweight = quantize(self.weight, qparams=weight_qparams)
#         if self.bias is not None:
#             qbias = quantize(
#                 self.bias, num_bits=self.num_bits_weight,
#                 flatten_dims=(0, -1))
#         else:
#             qbias = None

#         output = F.linear(qinput, qweight, qbias)
#         if ret_qinput:
#             return output, dict(type='fc', data=qinput.detach(),
#                                 in_features=self.in_features,
#                                 out_features=self.out_features)
#         else:
#             return output
