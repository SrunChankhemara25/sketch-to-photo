# Training and export guide

## Current workflows

- `notebooks/sketch2photo_master_training.ipynb` contains the broader bidirectional
  training and evaluation record.
- `notebooks/sketch_to_photo_demo_pix2pix_experts_v2.ipynb` trains the deployed
  portrait and general-scene resize-convolution experts.

Superseded attempts are preserved under `archive/notebooks/` for research history
and are not recommended starting points.

## Required evaluation

Image translation does not have a useful single “accuracy” value. At minimum,
record the following on a held-out split:

- LPIPS for perceptual distance (lower is better);
- MAE for pixel reconstruction error (lower is better);
- fixed visual gates covering portraits and general scenes; and
- the selected checkpoint step and selection score.

Never select a deployment checkpoint from training loss alone. Keep the test set
separate from checkpoint selection.

## Export contract

A deployable export must contain:

```text
manifest.json
models/<model-name>/best.pt
reports/learning_curves.png
reports/<model-name>_final_gate.png
```

Each manifest entry must define:

- stable model key and user-facing label;
- checkpoint path relative to the manifest;
- architecture identifier;
- input resolution;
- validation metrics; and
- selected training step.

After installing an export, update `src/config.py` only if its root folder changes.
Then run `python scripts/smoke_test.py` before starting Streamlit.

## Resume safety

Save resumable training state separately from `best.pt`. A resumable state may
include optimizers, schedulers and gradient scalers; a deployment checkpoint should
contain only the generator state and the metadata needed for inference.

## Quality expectations

Monochrome sketches discard colour and texture information. Higher resolution,
better hand-drawn/photo pairing and perceptual/adversarial objectives can improve
plausibility, but no deterministic model can recover the exact original photograph
from every sketch.
