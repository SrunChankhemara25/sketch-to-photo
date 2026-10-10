"""Shared, deterministic image preprocessing and display enhancement."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .config import MAX_OUTPUT_EDGE


def rgb_on_white(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and composite transparent uploads onto white."""
    image = ImageOps.exif_transpose(image)
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    return Image.alpha_composite(background, rgba).convert("RGB")


def fit_canvas(image: Image.Image, size: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Fit an image on a square white canvas without changing its aspect ratio."""
    image = rgb_on_white(image)
    scale = min(size / image.width, size / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    left, top = (size - width) // 2, (size - height) // 2
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(resized, (left, top))
    return canvas, (left, top, left + width, top + height)


def bounded_size(size: tuple[int, int], max_edge: int = MAX_OUTPUT_EDGE) -> tuple[int, int]:
    """Preserve aspect ratio while avoiding unexpectedly huge generated files."""
    width, height = size
    scale = min(1.0, max_edge / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def enhance_generated_photo(
    prediction: Image.Image,
    sketch: Image.Image,
    strength: float,
) -> Image.Image:
    """Improve presentation sharpness without changing the underlying model result.

    The prediction is resized to the bounded upload size. At non-zero strength, a
    conservative unsharp mask and a small amount of dark-line guidance from the input
    sketch are applied. This makes boundaries easier to read but cannot reconstruct
    texture or identity that the checkpoint did not learn.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    target_size = bounded_size(rgb_on_white(sketch).size)
    output = prediction.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
    if strength == 0.0:
        return output

    output = output.filter(ImageFilter.UnsharpMask(
        radius=0.8 + 0.8 * strength,
        percent=round(70 + 100 * strength),
        threshold=3,
    ))
    output = ImageEnhance.Contrast(output).enhance(1.0 + 0.06 * strength)

    rgb = np.asarray(output, dtype=np.float32) / 255.0
    gray = np.asarray(
        rgb_on_white(sketch).convert("L").resize(target_size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    line_mask = np.clip((205.0 - gray) / 205.0, 0.0, 1.0)
    rgb *= 1.0 - (0.08 * strength * line_mask[..., None])
    return Image.fromarray(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8))


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
