"""Approach A: skip-connected U-Net trained with paired reconstruction losses."""
import torch
from torch import nn
from torch.nn import functional as F


def block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU(),
                         nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU())


class PencilUNet(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.enc = nn.ModuleList([block(3, width), block(width, width*2),
                                 block(width*2, width*4), block(width*4, width*8)])
        self.mid = block(width*8, width*8)
        self.dec = nn.ModuleList([block(width*16, width*4), block(width*8, width*2),
                                 block(width*4, width), block(width*2, width)])
        self.out = nn.Conv2d(width, 3, 1)
        # Begin as an exact identity mapping.  Paired translation then learns only the
        # change required by the target instead of reconstructing the subject from
        # random features.  This materially improves structure/identity preservation
        # in short pilots and remains useful for both translation directions.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        source = x
        skips = []
        for layer in self.enc:
            x = layer(x); skips.append(x); x = F.avg_pool2d(x, 2)
        x = self.mid(x)
        for layer, skip in zip(self.dec, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = layer(torch.cat([x, skip], dim=1))
        return (source + self.out(x)).clamp(-1, 1)
