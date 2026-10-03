"""
src/models.py — checkpoint-compatible architecture definitions from the training notebook.

These MUST match the notebook byte-for-byte (or at least layer-for-layer), because
a checkpoint's state_dict only loads correctly if the module names/shapes line up
exactly with what it was saved from. Don't edit this file unless you also retrain
and re-export new checkpoints.
"""
import torch
import torch.nn as nn


# ── Model A: AutoEncoder — strided-conv encoder, transposed-conv decoder, NO skips ─
class ConvDownBlock(nn.Module):
    """Stride-2 Conv -> BN -> ReLU. Halves spatial size; no separate pooling layer."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class ConvUpBlock(nn.Module):
    """Transposed-conv (stride 2) -> BN -> ReLU. Doubles spatial size."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class SimpleAutoEncoder(nn.Module):
    """
    Approach A — plain autoencoder, NO skip connections.
    Strided-conv encoder 1->64->128->256->512 (bottleneck), transposed-conv decoder
    512->256->128->64->3 symmetrically back up to the input resolution.
    """
    def __init__(self, in_ch=1, out_ch=3):
        super().__init__()
        self.enc1 = ConvDownBlock(in_ch, 64)   # H    -> H/2
        self.enc2 = ConvDownBlock(64,   128)   # H/2  -> H/4
        self.enc3 = ConvDownBlock(128,  256)   # H/4  -> H/8
        self.enc4 = ConvDownBlock(256,  512)   # H/8  -> H/16 (bottleneck)

        self.dec4 = ConvUpBlock(512, 256)      # H/16 -> H/8
        self.dec3 = ConvUpBlock(256, 128)      # H/8  -> H/4
        self.dec2 = ConvUpBlock(128, 64)       # H/4  -> H/2
        self.dec1 = ConvUpBlock(64,  32)       # H/2  -> H
        self.out_conv = nn.Conv2d(32, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.enc1(x); x = self.enc2(x); x = self.enc3(x); x = self.enc4(x)  # bottleneck, no skips kept
        x = self.dec4(x); x = self.dec3(x); x = self.dec2(x); x = self.dec1(x)
        return torch.sigmoid(self.out_conv(x))


# ── Model B: U-Net — DoubleConv + max-pool encoder, transposed-conv decoder, CONCAT skips ─
class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU layers (the classic U-Net building block)."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c,  out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Approach B — U-Net, WITH concatenation skip connections from every encoder level.
    Encoder: DoubleConv blocks + max-pool downsampling, 1->64->128->256->512.
    Bottleneck: 512->1024.
    Decoder: transposed-conv upsampling + concatenation skips, mirrored back to 3 channels.
    """
    def __init__(self, in_ch=1, out_ch=3):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)

        self.up4  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)   # 512 (up) + 512 (skip)
        self.up3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


# ── Model C: ResNet-ED — residual encoder/decoder, NO skips between them ────
class ResidualBlock(nn.Module):
    """
    Standard 2-conv residual block. stride>1 downsamples via the first conv, with a
    matching 1x1-conv projection on the shortcut path so the addition stays valid.
    """
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResidualUpBlock(nn.Module):
    """Nearest-neighbor x2 upsample -> residual block (channel reduction)."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up  = nn.Upsample(scale_factor=2, mode='nearest')
        self.res = ResidualBlock(in_c, out_c, stride=1)

    def forward(self, x):
        return self.res(self.up(x))


class ResNetED(nn.Module):
    """
    Approach C — ResNet-style encoder-decoder, NO skip connections between encoder
    and decoder (this is what keeps it architecturally distinct from Approach B,
    which does use skip connections).
    Encoder: 7x7 stride-2 stem, then residual blocks with stride-2 downsampling,
    channels 64 -> 128 -> 256 -> 512.
    Decoder: residual blocks with nearest-neighbor x2 upsampling, mirrored back
    down to 3 output channels.
    """
    def __init__(self, in_ch=1, out_ch=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )                                        # H    -> H/2   (64ch)
        self.stage1 = ResidualBlock(64,  128, stride=2)  # H/2  -> H/4   (128ch)
        self.stage2 = ResidualBlock(128, 256, stride=2)  # H/4  -> H/8   (256ch)
        self.stage3 = ResidualBlock(256, 512, stride=2)  # H/8  -> H/16  (512ch), bottleneck

        self.up3 = ResidualUpBlock(512, 256)    # H/16 -> H/8
        self.up2 = ResidualUpBlock(256, 128)    # H/8  -> H/4
        self.up1 = ResidualUpBlock(128, 64)     # H/4  -> H/2
        self.up0 = ResidualUpBlock(64,  32)     # H/2  -> H   (undoes the stem's stride-2)
        self.out_conv = nn.Conv2d(32, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x)  # bottleneck, no skips kept
        x = self.up3(x); x = self.up2(x); x = self.up1(x); x = self.up0(x)
        return torch.sigmoid(self.out_conv(x))
