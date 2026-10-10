"""Artifact-resistant resize-convolution pix2pix generator."""

import torch
from torch import nn


class Down(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(input_channels, output_channels, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.InstanceNorm2d(output_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dropout: bool = False):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_channels, output_channels, 3, bias=False),
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.35))
        self.net = nn.Sequential(*layers)

    def forward(self, x, skip):
        return torch.cat([self.net(x), skip], dim=1)


class Pix2PixGenerator(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.d1 = Down(3, width, False)
        self.d2 = Down(width, width * 2)
        self.d3 = Down(width * 2, width * 4)
        self.d4 = Down(width * 4, width * 8)
        self.d5 = Down(width * 8, width * 8)
        self.d6 = Down(width * 8, width * 8)
        self.d7 = Down(width * 8, width * 8)
        self.d8 = Down(width * 8, width * 8, False)
        self.u1 = Up(width * 8, width * 8, True)
        self.u2 = Up(width * 16, width * 8, True)
        self.u3 = Up(width * 16, width * 8, True)
        self.u4 = Up(width * 16, width * 8)
        self.u5 = Up(width * 16, width * 4)
        self.u6 = Up(width * 8, width * 2)
        self.u7 = Up(width * 4, width)
        self.out = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.ReflectionPad2d(3),
            nn.Conv2d(width * 2, 3, 7),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)
        u1 = self.u1(d8, d7)
        u2 = self.u2(u1, d6)
        u3 = self.u3(u2, d5)
        u4 = self.u4(u3, d4)
        u5 = self.u5(u4, d3)
        u6 = self.u6(u5, d2)
        u7 = self.u7(u6, d1)
        return self.out(u7)
