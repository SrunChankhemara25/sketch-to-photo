"""Architectures for the new, paired 512px comparison (not legacy weights)."""

from .unet import PencilUNet
from .gan import ResnetGenerator, PatchDiscriminator


def make_generator(approach, width=64, output_mode="residual"):
    if approach == 'unet':
        return PencilUNet(width, output_mode=output_mode)
    if approach == 'gan':
        return ResnetGenerator(width)
    raise ValueError(approach)
