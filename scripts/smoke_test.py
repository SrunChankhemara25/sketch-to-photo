"""Load every deployed checkpoint on CPU and run a minimal inference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import PHOTO_TO_PENCIL_ROOT, SKETCH_TO_PHOTO_ROOT
from src.inference import PhotoToPencilService, SketchToPhotoService


def sample_sketch() -> Image.Image:
    image = Image.new("RGB", (320, 240), "white")
    drawing = ImageDraw.Draw(image)
    drawing.ellipse((85, 20, 235, 215), outline="black", width=5)
    drawing.ellipse((125, 85, 140, 100), fill="black")
    drawing.ellipse((180, 85, 195, 100), fill="black")
    drawing.arc((125, 110, 195, 165), 0, 180, fill="black", width=4)
    return image


def validate_image(name: str, image: Image.Image) -> None:
    array = np.asarray(image)
    if image.mode != "RGB" or array.size == 0 or not np.isfinite(array).all():
        raise AssertionError(f"{name} produced an invalid image")
    print(f"PASS {name:24} {image.width}x{image.height}")


def main() -> None:
    source = sample_sketch()

    photo_service = PhotoToPencilService(PHOTO_TO_PENCIL_ROOT, device="cpu")
    for model_key in ("unet", "gan"):
        result, _ = photo_service.predict(source, model_key)
        validate_image(f"photo_to_pencil/{model_key}", result)

    sketch_service = SketchToPhotoService(SKETCH_TO_PHOTO_ROOT, device="cpu")
    for model_key in ("general", "portrait"):
        result, _, _ = sketch_service.predict(source, model_key, detail_strength=0.35)
        validate_image(f"sketch_to_photo/{model_key}", result)

if __name__ == "__main__":
    main()
