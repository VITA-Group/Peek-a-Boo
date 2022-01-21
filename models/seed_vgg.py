'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

from .seed_conv import SeedConv2d
from masked_layers import layers

cfg = {
    'SeedVGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'SeedVGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'SeedVGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'SeedVGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class SeedVGG(nn.Module):
    def __init__(self, vgg_name, sign_grouped_dim=(), init_method='standard',
        hidden_act='none', scaling_input=False, num_classes=10):

        super(SeedVGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], sign_grouped_dim, init_method, hidden_act, scaling_input)
        self.classifier = layers.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, sign_grouped_dim, init_method, hidden_act, scaling_input):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [SeedConv2d(in_channels, x, kernel_size=3, padding=1, 
                               sign_grouped_dim=sign_grouped_dim, init_method=init_method, hidden_act=hidden_act,
                               scaling_input=scaling_input),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = SeedVGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
