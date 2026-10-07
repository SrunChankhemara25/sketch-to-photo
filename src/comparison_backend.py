"""Inference for the fair pilot U-Net versus residual-GAN comparison bundle."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .approaches import make_generator


FORMAT = "sketch2photo-two-model-comparison-v1"
DIRECTIONS = {"photo_to_pencil", "sketch_to_photo"}
APPROACHES = {"unet", "gan"}


def rgb_on_white(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and composite transparency against white."""
    image = ImageOps.exif_transpose(image)
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    return Image.alpha_composite(background, rgba).convert("RGB")


def fit_canvas(image: Image.Image, size: int) -> tuple[Image.Image, tuple[int, ...]]:
    """Use the aspect-preserving white canvas employed by the pilot."""
    image = rgb_on_white(image)
    scale = size / max(image.size)
    width, height = [max(1, min(size, round(value * scale))) for value in image.size]
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    left, top = (size - width) // 2, (size - height) // 2
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(resized, (left, top))
    return canvas, (left, top, left + width, top + height)


class ComparisonEditor:
    """Load one pilot generator at a time and return its unmodified prediction."""

    def __init__(self, bundle_dir: str | Path, device: str | None = None):
        self.root = Path(bundle_dir)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"Expected a {FORMAT} bundle")
        # CPU is the conservative local default on macOS. These pilot generators
        # complete quickly on CPU, and avoiding MPS prevents backend-specific numerical
        # differences from contaminating a teacher-facing model comparison.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.key = None

    def _clear(self) -> None:
        self.model = None
        self.key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _load(self, direction: str, approach: str) -> None:
        key = (direction, approach)
        if key == self.key:
            return
        self._clear()
        info = self.manifest["models"][direction][approach]
        checkpoint = self.root / info["path"]
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing {approach}/{direction} checkpoint: {checkpoint}"
            )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("direction") != direction or state.get("approach") != approach:
            raise RuntimeError("Checkpoint metadata does not match the requested model")
        if int(state.get("step", 0)) <= 0:
            raise RuntimeError("A zero-step checkpoint cannot be presented as trained")
        model = make_generator(
            approach,
            int(state.get("width", 64)),
            output_mode=state.get("output_mode", "residual"),
        )
        model.load_state_dict(state["generator"], strict=True)
        self.model = model.to(self.device).eval()
        self.key = key

    def predict(self, image: Image.Image, direction: str, approach: str) -> Image.Image:
        if image is None:
            raise ValueError("Upload an image first")
        if direction not in DIRECTIONS:
            raise ValueError(f"Unsupported direction: {direction}")
        if approach not in APPROACHES:
            raise ValueError(f"Unsupported approach: {approach}")
        self._load(direction, approach)
        info = self.manifest["models"][direction][approach]
        canvas, box = fit_canvas(image, int(info["resolution"]))
        if direction == "sketch_to_photo":
            canvas = canvas.convert("L").convert("RGB")
        array = np.asarray(canvas, dtype=np.float32).copy().transpose(2, 0, 1)
        source = torch.from_numpy(array / 127.5 - 1.0)[None].to(self.device)
        with torch.inference_mode():
            output = self.model(source)[0].float().cpu().clamp(-1, 1)
        result = Image.fromarray(
            ((output.permute(1, 2, 0).numpy() + 1) * 127.5)
            .round()
            .astype("uint8")
        ).crop(box)
        if direction == "photo_to_pencil":
            result = result.convert("L").convert("RGB")
        return result
