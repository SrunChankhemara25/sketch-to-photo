# Sketch2Photo

Student: Srun Chankhemara (verify spelling for submission).

## New training experiment — use this notebook

Use [sketch2photo_master_training.ipynb](notebooks/sketch2photo_master_training.ipynb)
in Colab. It trains **both directions**, comparing a U-Net, conditional GAN and
pretrained diffusion fine-tuning on identical splits. It includes LR/weight-decay
tuning, shared evaluation, failure analysis and a compatible model export.
See the [complete training and run guide](docs/DUAL_TRAINING.md).

The design has been revised; **better output is not yet demonstrated**. New full
training has not run. Default data combines COCO synthetic pairs and FS2K real
portrait drawings, with explicit dataset setup and limitations in the notebook.

There is one Streamlit web launcher: `python app.py` (or `streamlit run app.py`).
The sidebar uses the original models until an export is installed at
`checkpoints/quality_study/`, then offers the validation-selected new models.
Original checkpoints are not removed. Install `requirements-diffusion.txt` when
running the portable new-model export by itself.

The local project can also load the installed fair pilot bundle at
`checkpoints/two_model_comparison/`. Its comparison view runs the trained U-Net and
residual conditional-GAN generators side by side for both directions. Large checkpoint
weights are kept out of ordinary Git; retain the Colab ZIP or publish them with Git LFS
or an external download link.

The sections below describe the **original application**, not the newly trained
study models. Its pencil renderer is image processing, not a learned pencil model.

## Original application

A Streamlit web application for two image workflows:

- **Photo → pencil drawing** — preserves facial shading, hair detail, and fabric folds with continuous graphite values while keeping the uploaded image's aspect ratio.
- **Sketch → colour photo** — runs one of the included PyTorch checkpoints to create a plausible colour interpretation of a pencil sketch.

The project includes AutoEncoder, U-Net, and ResNet encoder-decoder model definitions. The tuned U-Net is selected by default because it has the best validation result among the included exports.

## Important quality note

The supplied neural checkpoints were trained for eight epochs on COCO images at **128 × 128** with an L1 reconstruction objective. A pencil sketch does not retain the original colours, texture, or all facial/object detail, so a sketch-only input cannot be reconstructed perfectly by these checkpoints. Their output is an AI estimate, not the original photograph.

The web app displays actual renderer/model outputs. It does not substitute the uploaded
original photo for a generated result. A high-fidelity round trip must be earned by the
trained models and reported using the held-out evaluation.

## Features

- Continuous pencil shading from fine, medium, and broad image structure; a soft detail mask can lighten blurry scenery.
- Separate model-ready sketch generation that exactly follows the saved checkpoint pipeline.
- Automatic checkpoint discovery and architecture selection.
- CUDA, Apple Silicon (MPS), and CPU support.
- Visible 128 × 128 model input so preprocessing mistakes are easy to diagnose.
- Automatic polarity correction for graphite-on-white uploads, matching the dark sketch
  representation on which the checkpoints were trained.
- Preview and downloaded PNG are the same actual output.

## Requirements

- Python 3.10 or newer
- A trained checkpoint in `checkpoints/` (the project already includes example checkpoints)
- Optional: CUDA-capable NVIDIA GPU or Apple Silicon for faster inference

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
python app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`. The same app can
be hosted on a server; it is not limited to a temporary local test UI. To select a
different port, use Streamlit's command:

```bash
streamlit run app.py --server.port 8090
```

## How to use it

### 1. Create a pencil drawing

1. Open **Photo → Pencil Sketch**.
2. Upload a photo and select **Create pencil drawing**.
3. Use **Pencil darkness** to control the result. Start at `1.05`. **Fade soft scenery** (default `0.6`) lightens blurry regions; it can also lighten smooth subject areas and is not background segmentation.
4. Adjust **Pencil and paper texture** (default `0.45`) for subtle irregular graphite marks. Zero gives smooth shading; higher values make paper grain and short diagonal pencil marks stronger.
5. Download **Pencil drawing · v5**. If the app still says **v4** or **Graphite pencil drawing**, you have an older process or browser session open.

This renderer is a classical image-processing effect, not a trained photo-to-illustration network.
It retains the photo's geometry; it cannot invent artist-designed hair strokes, simplify an
entire scene semantically, or reproduce manga character proportions. The included neural
checkpoints solve the opposite direction (sketch to photo).

The main pencil preview and download use the same drawing. The dark 256 × 256 OpenCV
representation required by the neural network is hidden under **Advanced: input prepared
for the trained model**. It is a technical input rather than a second drawing result.

### 2. Convert a sketch to a photo

Choose the workflow that matches what you have:

| Input | Output mode | Result |
| --- | --- | --- |
| Finished drawing sent from Step 1 | AI estimate from sketch | Shows the same finished drawing as the input preview, while internally passing its paired training sketch to the 128 × 128 network. |
| Hand-drawn or external sketch | AI estimate from sketch | A plausible result, but less reliable because these sketches were not the exact training distribution. |

Step 1 keeps the paired drawing information in Streamlit session state so it can be
used in Step 2. The state does not survive a server restart or a new browser session.
The technical 128 × 128 input is shown only in a collapsed Advanced section.

Independent clean graphite-on-white uploads are detected and inverted automatically before inference.
This is necessary because the generated OpenCV sketches used for training are usually dark;
the **Technical network input** panel shows when this correction has been applied.

For the AI path, prefer `ModelB_UNet_Tuned_best.pt` and **Model B — U-Net**. If the image looks corrupted on MPS or CUDA, select CPU once to rule out a device/backend issue.

## Checkpoints and preprocessing

Place additional `.pt` checkpoint files in `checkpoints/` and select the matching architecture in the UI. Filenames containing `AutoEncoder`, `UNet`, or `ResNetED` are recognised automatically.

The AI inference path intentionally follows the training representation:

```text
photo → 256 × 256 cv2.pencilSketch (sigma_s=60, sigma_r=0.07)
      → 128 × 128 grayscale tensor in [0, 1]
      → trained model → 128 × 128 RGB estimate
```

Changing the architecture, input polarity, preprocessing, or resolution without retraining can make output much worse. `src/models.py` must remain layer-compatible with the checkpoint used.

## Improving the sketch-only AI model

For the new experiment, use the **master notebook linked at the top of this README**.
It is the only training notebook in this project.

Step 2 has **Colour intensity** and **Reduce cream/yellow cast** controls for display/export
post-processing. The default values are `1.5` and `0.35`. These amplify existing predicted
colours and optionally neutralize bright areas; they do not infer new object colours or
change the learned network. Use `1.0` and `0.0` for raw predictions and model evaluation.
Cast correction can alter intentional warm lighting. After adjusting a control, click
**Generate colour photo** again.

The [Colab training guide](docs/TRAINING_GUIDE.md) points to the master workflow.
The app now reads input resolution and pencil style from each checkpoint's saved config,
so new white-pencil weights and old dark-sketch weights can coexist.

To materially improve sketch-to-photo results, retrain rather than applying post-processing to the output. The most effective upgrades are:

1. Train at 256 or 512 pixels with correctly paired sketches and photos.
2. Train longer and validate with perceptual/SSIM-style losses in addition to L1.
3. Use a conditional diffusion or GAN-based image-to-image model for photorealistic texture.
4. Include the same clean hand-drawn sketch style you expect users to upload.

These changes require a new training run and new compatible checkpoints; no frontend adjustment can restore information that is absent from a sketch.

## Project structure

```text
sketch2photo_test_ui/
├── app.py              # local application launcher
├── src/
│   ├── __init__.py
│   ├── application.py  # original-model processing/inference compatibility backend
│   ├── models.py       # original checkpoint-compatible architectures
│   ├── diffusion_backend.py # pretrained-editor inference
│   ├── study_backend.py     # new study-bundle inference
│   └── approaches/          # new U-Net, residual GAN and diffusion training core
├── notebooks/
│   └── sketch2photo_master_training.ipynb # only standalone Colab training notebook
├── docs/               # project structure and Colab training guide
├── checkpoints/        # trained .pt model files
├── results/            # exported metrics and sample results
├── requirements.txt    # Python dependencies
└── README.md
```

See [Project structure](docs/PROJECT_STRUCTURE.md) for file responsibilities and GitHub
submission notes. Training stays in Colab; VS Code is for the local application.

## Sources, results and disclosure

New measured results are pending; run the master notebook's full evaluation before
making quality claims. It exports shared result tables, curves, tuning records,
hardware/time reports and failure figures. Add real weight-download links and the
required slides after training; none have been fabricated here.

Technical sources: [U-Net](https://arxiv.org/abs/1505.04597),
[pix2pix/CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix),
[LPIPS](https://github.com/richzhang/PerceptualSimilarity),
[InstructPix2Pix](https://www.timothybrooks.com/instruct-pix2pix/),
[pretrained weights](https://huggingface.co/timbrooks/instruct-pix2pix),
[COCO](https://cocodataset.org/#download), and [FS2K](https://github.com/DengPingFan/FS2K).
Dataset image rights are distinct from code licenses. Record actual counts and
distribution, real-vs-synthetic pencil supervision, portrait/style bias and possible
pretrained-data overlap. Code and experiment design are AI-assisted; the student
must understand, execute, verify and disclose that assistance under course rules.

## Troubleshooting

- **No checkpoint appears:** put a `.pt` file in `checkpoints/`, then use **Refresh checkpoints**.
- **Checkpoint architecture error:** select the model family that created the checkpoint (A, B, or C).
- **AI result is blurry:** this is expected from the supplied 128 × 128 checkpoints. Retrain with the master 512px experiment and compare the measured results.
- **A white-paper sketch gives a black result:** the app now detects and inverts this input style automatically. Confirm that the **Actual 128×128 model input** is dark with light structure; restart the app if you still have an older session open.
- **Transparent sketch uploads look black:** export them against a white background; the application also composites transparent uploads onto white automatically.
- **Every result shows “Error”:** stop the old process, start `python app.py` again, and open the fresh Streamlit URL. The app shows the underlying error and rejects unavailable MPS/CUDA devices.
