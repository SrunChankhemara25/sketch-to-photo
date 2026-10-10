# Sketch2Photo Studio

A local PyTorch and Streamlit application for two image-to-image directions:

- **Photo → Pencil** using a trained 512px U-Net or residual GAN.
- **Pencil → Colour Photo** using dedicated 256px general-scene and portrait
  pix2pix experts.

The application loads validation-selected checkpoints, displays their saved
metrics and exports exactly the image shown in the interface.

## Quick start

Requirements: Python 3.11+ and approximately 2 GB of free memory for CPU
inference. CUDA and Apple Metal are detected automatically when available.

```bash
cd /Users/khemara/Downloads/sketch2photo_test_ui
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open the local URL printed by Streamlit, normally
`http://localhost:8501`. You can also start the app with:

```bash
streamlit run app.py
```

## Using the application

1. Choose **Photo → Pencil** or **Pencil → Colour Photo**.
2. Select a checkpoint from **Trained model**.
3. Upload a PNG, JPEG or WebP image.
4. For sketch-to-photo, optionally adjust **Detail enhancement**.
5. Generate and download the PNG result.

### Model selection

| Direction | Model | Training resolution | Validation |
|---|---|---:|---:|
| Photo → Pencil | U-Net (recommended) | 512px | LPIPS 0.2600 |
| Photo → Pencil | Residual GAN | 512px | LPIPS 0.3204 |
| Pencil → Photo | General scene (recommended) | 256px | LPIPS 0.2040, MAE 0.0531 |
| Pencil → Photo | Portrait / face | 256px | LPIPS 0.4324, MAE 0.2256 |

Lower LPIPS and MAE are better. Metrics are not “accuracy” and do not replace
visual inspection.

## Reducing soft output

The sketch-to-photo checkpoints are low-resolution generative models, so some
softness is learned into their predictions. The application now:

1. restores the model crop to the upload’s aspect ratio and bounded display size;
2. applies a conservative unsharp mask; and
3. transfers a small amount of line detail from the input sketch.

Set **Detail enhancement** to `0` for the raw checkpoint prediction. The default
`0.35` improves visible boundaries without aggressively drawing over the result.
This post-processing cannot recreate texture, identity or true colours that are
absent from the sketch. Material quality improvements require retraining at a
higher resolution with stronger paired data.

## Repository structure

```text
sketch2photo_test_ui/
├── app.py                         # minimal Streamlit entry point
├── src/
│   ├── config.py                  # paths and deployment constants
│   ├── image_processing.py        # shared preprocessing/enhancement
│   ├── architectures/             # checkpoint-compatible networks
│   ├── inference/                 # one service per model direction
│   └── ui/                        # Streamlit presentation layer
├── checkpoints/
│   ├── README.md                  # checkpoint management rules
│   ├── photo_to_pencil/           # manifest + deployed U-Net/GAN
│   └── sketch_to_photo/           # manifest + deployed experts/reports
├── notebooks/                     # current training notebooks
├── reports/                       # historical evaluation artifacts
├── docs/                          # developer and training documentation
├── .vscode/                       # shared editor defaults and extensions
└── archive/                       # superseded experiments; not imported
```

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for module ownership
and [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for export requirements.

## Checkpoint contracts

The `.pt` files are architecture-specific. A checkpoint cannot be reversed or
loaded into another architecture by renaming it. Every deployment bundle has a
`manifest.json` containing the relative path, architecture, training resolution,
selected step and validation metrics. Keep the manifest and weight together.

Photo-to-pencil checkpoints are stored under `checkpoints/photo_to_pencil/models/`;
sketch-to-photo checkpoints are stored under `checkpoints/sketch_to_photo/models/`.

## Development checks

Compile all application modules:

```bash
python -m compileall -q app.py src
```

Run the model smoke test:

```bash
python scripts/smoke_test.py
```

The smoke test loads every deployed checkpoint on CPU and runs one inference.

Verify checkpoint size and SHA-256 integrity after copying or downloading weights:

```bash
python scripts/verify_checkpoints.py
```

## Known limitations

- A monochrome drawing does not uniquely encode the source colours.
- Portrait identity is not reliable; the portrait expert is experimental.
- General-scene output preserves structure better than fine photographic texture.
- Detail enhancement changes presentation only, not validation metrics or learned weights.
- Checkpoints use Git LFS; install Git LFS before cloning and maintain a separate backup.

## License and data

Only use training images and checkpoints whose licenses permit your intended use.
Dataset licenses are separate from this application’s source code.
