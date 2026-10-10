"""Inference for the demo pix2pix v2 portrait and general-scene experts."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn


FORMAT = "sketch2photo-demo-pix2pix-experts-v2-resizeconv"
EXPERTS = {"portrait", "general"}


class Down(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, output_channels, 4, 2, 1, bias=not norm)
        ]
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
    """Layer-for-layer match for the v2 Colab training notebook generator."""

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


def rgb_on_white(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    return Image.alpha_composite(background, rgba).convert("RGB")


def fit_canvas(image: Image.Image, size: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Match the notebook's aspect-preserving white-canvas preprocessing."""
    image = rgb_on_white(image)
    scale = min(size / image.width, size / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    left, top = (size - width) // 2, (size - height) // 2
    canvas.paste(resized, (left, top))
    return canvas, (left, top, left + width, top + height)


def available_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ExpertEditor:
    """Load one expert at a time and generate an unmodified sketch-to-photo result."""

    def __init__(self, bundle_dir: str | Path, device: str | None = None):
        self.root = Path(bundle_dir)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"Expected a {FORMAT} bundle")
        self.device = device or available_device()
        self.model: Pix2PixGenerator | None = None
        self.expert: str | None = None
        self.state: dict | None = None

    def _clear(self) -> None:
        self.model = None
        self.expert = None
        self.state = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _load(self, expert: str) -> None:
        if expert == self.expert:
            return
        if expert not in EXPERTS:
            raise ValueError(f"Unknown expert: {expert}")
        self._clear()
        checkpoint = self.root / self.manifest["experts"][expert]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing {expert} checkpoint: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("expert") != expert or state.get("direction") != "sketch_to_photo":
            raise RuntimeError(f"The {expert} checkpoint metadata is invalid")
        model = Pix2PixGenerator(int(state.get("width", 64)))
        model.load_state_dict(state["generator"], strict=True)
        self.model = model.to(self.device).eval()
        self.expert = expert
        self.state = state

    def predict(self, image: Image.Image, expert: str) -> tuple[Image.Image, Image.Image, dict]:
        if image is None:
            raise ValueError("Upload a pencil sketch first")
        self._load(expert)
        assert self.model is not None and self.state is not None
        resolution = int(self.state.get("resolution", self.manifest.get("resolution", 256)))
        canvas, box = fit_canvas(image, resolution)
        model_input = canvas.convert("L").convert("RGB")
        array = np.asarray(model_input, dtype=np.float32).copy().transpose(2, 0, 1)
        source = torch.from_numpy(array / 127.5 - 1.0)[None].to(self.device)
        with torch.inference_mode():
            output = self.model(source)[0].float().cpu().clamp(-1, 1)
        result = Image.fromarray(
            ((output.permute(1, 2, 0).numpy() + 1.0) * 127.5)
            .round()
            .astype("uint8")
        ).crop(box)
        validation = dict(self.state.get("validation", {}))
        details = {
            "expert": expert,
            "step": int(self.state.get("step", 0)),
            "resolution": resolution,
            "device": self.device,
            **validation,
        }
        return result, model_input, details
