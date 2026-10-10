"""Inference service for the trained photo-to-pencil U-Net and residual GAN."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..architectures import PencilUNet, ResnetGenerator
from ..image_processing import fit_canvas
from .device import preferred_device, release_memory


class PhotoToPencilService:
    def __init__(self, root: str | Path, device: str | None = None):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") != "sketch2photo-photo-to-pencil-v1":
            raise ValueError("Unsupported photo-to-pencil checkpoint bundle")
        self.device = device or preferred_device()
        self.model = None
        self.model_key = None

    def _load(self, model_key: str) -> dict:
        if model_key == self.model_key:
            return self.manifest["models"][model_key]
        release_memory()
        info = self.manifest["models"][model_key]
        state = torch.load(self.root / info["path"], map_location="cpu", weights_only=True)
        width = int(state.get("width", info.get("width", 64)))
        if model_key == "unet":
            model = PencilUNet(width, output_mode=state.get("output_mode", "residual"))
        elif model_key == "gan":
            model = ResnetGenerator(width)
        else:
            raise ValueError(f"Unknown photo-to-pencil model: {model_key}")
        model.load_state_dict(state["generator"], strict=True)
        self.model = model.to(self.device).eval()
        self.model_key = model_key
        return info

    def predict(self, image: Image.Image, model_key: str) -> tuple[Image.Image, dict]:
        info = self._load(model_key)
        resolution = int(info["resolution"])
        canvas, box = fit_canvas(image, resolution)
        array = np.asarray(canvas, dtype=np.float32).copy().transpose(2, 0, 1)
        source = torch.from_numpy(array / 127.5 - 1.0)[None].to(self.device)
        with torch.inference_mode():
            output = self.model(source)[0].float().cpu().clamp(-1, 1)
        result = Image.fromarray(
            np.rint((output.permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
        ).crop(box).convert("L").convert("RGB")
        return result, {
            "model": model_key,
            "resolution": resolution,
            "step": info["selected_step"],
            "device": self.device,
            "validation_lpips": info["validation_lpips"],
        }
