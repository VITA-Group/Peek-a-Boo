'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import math

from masked_layers.layers import Conv2d, Linear

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, init_method='standard'):
        super(VGG, self).__init__()
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

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
