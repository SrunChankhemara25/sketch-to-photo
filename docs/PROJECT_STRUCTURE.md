# Project structure and ownership

The runtime follows a small layered design. Dependencies point inward from the UI
to inference services, then to architectures and shared image utilities.

```text
app.py
  └── src/ui/app.py
        ├── src/inference/photo_to_pencil.py
        ├── src/inference/sketch_to_photo.py
        ├── src/image_processing.py
        └── src/config.py
              └── checkpoints/**/manifest.json
```

## Module responsibilities

- `app.py`: starts Streamlit and delegates rendering. It contains no model logic.
- `src/ui/`: widgets, session state, user messages and downloads.
- `src/inference/`: checkpoint loading, tensor conversion and prediction.
- `src/architectures/`: layer-for-layer checkpoint-compatible PyTorch modules.
- `src/image_processing.py`: EXIF/transparency handling, aspect-preserving canvas
  fitting, bounded resizing and optional display enhancement.
- `src/config.py`: stable project paths and application constants.
- `checkpoints/`: deployed model weights, manifests and visual reports.
- `notebooks/`: current reproducible training workflows.
- `archive/`: superseded experiments that are intentionally excluded at runtime.

## Design rules

1. Never import Streamlit from architecture or inference modules.
2. Never put checkpoint-specific layer definitions in the UI.
3. Treat manifests as the deployment contract; avoid hard-coded metrics in UI code.
4. Keep preprocessing deterministic and shared between smoke tests and production.
5. Preserve raw output mode (`Detail enhancement = 0`) for honest evaluation.
6. Do not rename or reverse a checkpoint to represent another training direction.

## Adding a model

1. Add its architecture under `src/architectures/`.
2. Add or extend a direction-specific inference service.
3. Store the weight below its direction folder in `checkpoints/`.
4. Add the model and validation metadata to the direction manifest.
5. Extend `scripts/smoke_test.py` and run it before exposing the model in the UI.

Use `scripts/verify_checkpoints.py` after downloading or moving weights to verify
their file sizes and SHA-256 checksums against the manifests.
