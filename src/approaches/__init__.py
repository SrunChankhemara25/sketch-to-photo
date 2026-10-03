"""Architectures for the new, paired 512px comparison (not legacy weights)."""

from .unet import PencilUNet
from .gan import ResnetGenerator, PatchDiscriminator


def make_generator(approach, width=64):
    if approach == 'unet':
        return PencilUNet(width)
    if approach == 'gan':
        return ResnetGenerator(width)
    raise ValueError(approach)
