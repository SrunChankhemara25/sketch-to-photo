"""Verify deployed checkpoint files against their manifest metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = (
    ROOT / "checkpoints/photo_to_pencil/manifest.json",
    ROOT / "checkpoints/sketch_to_photo/manifest.json",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for block in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_models(manifest: dict) -> list[dict]:
    models = manifest.get("models")
    return list(models.values()) if models is not None else [manifest["model"]]


def main() -> None:
    for manifest_path in MANIFESTS:
        manifest = json.loads(manifest_path.read_text())
        for model in manifest_models(manifest):
            checkpoint_path = manifest_path.parent / model["path"]
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            if checkpoint_path.stat().st_size != model["size_bytes"]:
                raise ValueError(f"Size mismatch: {checkpoint_path}")
            if file_sha256(checkpoint_path) != model["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {checkpoint_path}")
            print(f"PASS {checkpoint_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
