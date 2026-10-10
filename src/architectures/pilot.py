"""Architectures for the trained photo-to-pencil pilot checkpoints."""

import torch
from torch import nn
from torch.nn import functional as F


def _unet_block(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, padding=1),
        nn.GroupNorm(8, output_channels),
        nn.SiLU(),
        nn.Conv2d(output_channels, output_channels, 3, padding=1),
        nn.GroupNorm(8, output_channels),
        nn.SiLU(),
    )


class PencilUNet(nn.Module):
    def __init__(self, width: int = 64, output_mode: str = "residual"):
        super().__init__()
        self.output_mode = output_mode
        self.enc = nn.ModuleList([
            _unet_block(3, width),
            _unet_block(width, width * 2),
            _unet_block(width * 2, width * 4),
            _unet_block(width * 4, width * 8),
        ])
        self.mid = _unet_block(width * 8, width * 8)
        self.dec = nn.ModuleList([
            _unet_block(width * 16, width * 4),
            _unet_block(width * 8, width * 2),
            _unet_block(width * 4, width),
            _unet_block(width * 2, width),
        ])
        self.out = nn.Conv2d(width, 3, 1)

    def forward(self, x):
        source = x
        skips = []
        for layer in self.enc:
            x = layer(x)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.mid(x)
        for layer, skip in zip(self.dec, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = layer(torch.cat([x, skip], dim=1))
        prediction = self.out(x)
        return (source + prediction).clamp(-1, 1) if self.output_mode == "residual" else torch.tanh(prediction)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels, affine=True), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.layers(x)


class ResnetGenerator(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3), nn.Conv2d(3, width, 7),
            nn.InstanceNorm2d(width, affine=True), nn.ReLU(),
        ]
        for input_channels, output_channels in ((width, width * 2), (width * 2, width * 4)):
            layers.extend([
                nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(output_channels, affine=True), nn.ReLU(),
            ])
        layers.extend(ResidualBlock(width * 4) for _ in range(9))
        for input_channels, output_channels in ((width * 4, width * 2), (width * 2, width)):
            layers.extend([
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(input_channels, output_channels, 3, padding=1),
                nn.InstanceNorm2d(output_channels, affine=True), nn.ReLU(),
            ])
        layers.extend([nn.ReflectionPad2d(3), nn.Conv2d(width, 3, 7), nn.Tanh()])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
