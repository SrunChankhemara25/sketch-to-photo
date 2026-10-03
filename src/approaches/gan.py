"""Approach B: residual generator plus conditional PatchGAN; no U-Net skips.

Uses adversarial, paired reconstruction and perceptual losses in the notebook.
Inspired by the pix2pix/CycleGAN project; this is a separate PyTorch implementation.
"""
from torch import nn


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels, affine=True), nn.ReLU(), nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3), nn.InstanceNorm2d(channels, affine=True))

    def forward(self, x):
        return x + self.layers(x)


class ResnetGenerator(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, width, 7),
                  nn.InstanceNorm2d(width, affine=True), nn.ReLU()]
        for a, b in ((width, width*2), (width*2, width*4)):
            layers += [nn.Conv2d(a, b, 3, stride=2, padding=1),
                       nn.InstanceNorm2d(b, affine=True), nn.ReLU()]
        layers += [Residual(width*4) for _ in range(9)]
        for a, b in ((width*4, width*2), (width*2, width)):
            layers += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                       nn.Conv2d(a, b, 3, padding=1), nn.InstanceNorm2d(b, affine=True), nn.ReLU()]
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(width, 3, 7), nn.Tanh()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        layers = [nn.utils.spectral_norm(nn.Conv2d(6, width, 4, 2, 1)), nn.LeakyReLU(.2)]
        for a, b, stride in ((width, width*2, 2), (width*2, width*4, 2), (width*4, width*8, 1)):
            layers += [nn.utils.spectral_norm(nn.Conv2d(a, b, 4, stride, 1)), nn.LeakyReLU(.2)]
        layers += [nn.utils.spectral_norm(nn.Conv2d(width*8, 1, 4, 1, 1))]
        self.layers = nn.Sequential(*layers)

    def forward(self, paired_images):
        return self.layers(paired_images)
