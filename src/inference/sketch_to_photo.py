"""Inference service for the 256px portrait and general-scene pix2pix experts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..architectures import Pix2PixGenerator
from ..image_processing import enhance_generated_photo, fit_canvas
from .device import preferred_device, release_memory


class SketchToPhotoService:
    def __init__(self, root: str | Path, device: str | None = None):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") != "sketch2photo-sketch-to-photo-v1":
            raise ValueError("Unsupported sketch-to-photo checkpoint bundle")
        self.device = device or preferred_device()
        self.model = None
        self.model_key = None
        self.state = None

    def _load(self, model_key: str) -> dict:
        if model_key == self.model_key:
            return self.manifest["models"][model_key]
        release_memory()
        info = self.manifest["models"][model_key]
        state = torch.load(self.root / info["path"], map_location="cpu", weights_only=True)
        model = Pix2PixGenerator(int(state.get("width", 64)))
        model.load_state_dict(state["generator"], strict=True)
        self.model = model.to(self.device).eval()
        self.model_key = model_key
        self.state = state
        return info

    def predict(
        self,
        image: Image.Image,
        model_key: str,
        detail_strength: float = 0.0,
    ) -> tuple[Image.Image, Image.Image, dict]:
        info = self._load(model_key)
        resolution = int(info["resolution"])
        canvas, box = fit_canvas(image, resolution)
        technical_input = canvas.convert("L").convert("RGB")
        array = np.asarray(technical_input, dtype=np.float32).copy().transpose(2, 0, 1)
        source = torch.from_numpy(array / 127.5 - 1.0)[None].to(self.device)
        with torch.inference_mode():
            output = self.model(source)[0].float().cpu().clamp(-1, 1)
        raw = Image.fromarray(
            np.rint((output.permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
        ).crop(box)
        result = enhance_generated_photo(raw, image, detail_strength)
        return result, technical_input, {
            "model": model_key,
            "resolution": resolution,
            "step": info["selected_step"],
            "device": self.device,
            "detail_strength": detail_strength,
            **info["validation"],
        }
