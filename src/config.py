"""Application paths and deployment constants."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"

PHOTO_TO_PENCIL_ROOT = CHECKPOINT_ROOT / "photo_to_pencil"
SKETCH_TO_PHOTO_ROOT = CHECKPOINT_ROOT / "sketch_to_photo"

SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]
MAX_OUTPUT_EDGE = 2048
