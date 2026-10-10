"""Export the completed fair U-Net versus GAN pilot from Google Drive.

Run in a fresh Colab code cell after mounting Drive:

    %run /content/two_model_comparison_export_colab.py

This script does not train, evaluate, or download datasets.  It verifies that the
four selected pilot checkpoints are non-zero trained steps, creates a transparent
two-architecture comparison app, saves it in Drive, and downloads the ZIP.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch


DRIVE_ROOT = Path("/content/drive/MyDrive/Sketch2Photo_training")
PILOT_ROOT = DRIVE_ROOT / "identity_colour_gate_pilot_v2" / "pilot"

SOURCES = {
    ("photo_to_pencil", "unet"): PILOT_ROOT / "unet/photo_to_pencil/best.pt",
    ("sketch_to_photo", "unet"): PILOT_ROOT / "unet/sketch_to_photo/best.pt",
    ("photo_to_pencil", "gan"): PILOT_ROOT / "gan/photo_to_pencil/best.pt",
    ("sketch_to_photo", "gan"): PILOT_ROOT / "gan/sketch_to_photo/best.pt",
}

if not DRIVE_ROOT.is_dir():
    raise RuntimeError("Mount Google Drive first: drive.mount('/content/drive')")

states = {}
for key, path in SOURCES.items():
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved comparison checkpoint: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    direction, approach = key
    if state.get("direction") != direction or state.get("approach") != approach:
        raise RuntimeError(
            f"Checkpoint metadata mismatch for {path}: "
            f"{state.get('direction')!r}, {state.get('approach')!r}"
        )
    if int(state.get("step", 0)) <= 0:
        raise RuntimeError(
            f"{approach}/{direction} selected step is zero and cannot be claimed as trained."
        )
    states[key] = state

print("Verified four non-zero trained checkpoints:")
for (direction, approach), state in states.items():
    print(
        f"- {approach:5s} | {direction:16s} | step {int(state['step']):5d} "
        f"| val LPIPS {float(state.get('val_lpips', float('nan'))):.6f}"
    )

stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
export_root = DRIVE_ROOT / f"two_model_comparison_{stamp}"
model_root = export_root / "models"
source_root = export_root / "src"
export_root.mkdir(parents=True, exist_ok=False)
model_root.mkdir()
source_root.mkdir()

models = {direction: {} for direction in ("photo_to_pencil", "sketch_to_photo")}
for (direction, approach), source in SOURCES.items():
    destination = model_root / direction / approach / "best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    state = states[(direction, approach)]
    models[direction][approach] = {
        "path": str(destination.relative_to(export_root)),
        "selected_step": int(state["step"]),
        "validation_lpips": float(state.get("val_lpips", float("nan"))),
        "resolution": int(state.get("resolution", 512)),
        "width": int(state.get("width", 64)),
        "output_mode": state.get("output_mode", "residual"),
        "training": state.get("training", "pilot"),
    }

defaults = {
    direction: min(
        models[direction],
        key=lambda approach: models[direction][approach]["validation_lpips"],
    )
    for direction in models
}

manifest = {
    "format": "sketch2photo-two-model-comparison-v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "experiment": "identity_colour_gate_pilot_v2",
    "purpose": "Fair comparison of two distinct trained architectures on the same pilot protocol.",
    "architectures": {
        "unet": "Skip-connected PencilUNet trained with paired reconstruction/perceptual losses.",
        "gan": "Nine-block residual generator trained with a conditional PatchGAN discriminator.",
    },
    "models": models,
    "validation_best": defaults,
    "quality_status": (
        "Experimental comparison. Validation selection does not mean production quality; "
        "known failures and visual-gate rejections must be reported."
    ),
}
(export_root / "manifest.json").write_text(json.dumps(manifest, indent=2))


MODELS_SOURCE = r'''"""The two distinct generator architectures used in the fair pilot."""
import torch
from torch import nn
from torch.nn import functional as F


def block(a, b):
    return nn.Sequential(
        nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU(),
        nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU(),
    )


class PencilUNet(nn.Module):
    def __init__(self, width=64, output_mode="residual"):
        super().__init__()
        if output_mode not in {"residual", "direct"}:
            raise ValueError(output_mode)
        self.output_mode = output_mode
        self.enc = nn.ModuleList([
            block(3, width), block(width, width * 2),
            block(width * 2, width * 4), block(width * 4, width * 8),
        ])
        self.mid = block(width * 8, width * 8)
        self.dec = nn.ModuleList([
            block(width * 16, width * 4), block(width * 8, width * 2),
            block(width * 4, width), block(width * 2, width),
        ])
        self.out = nn.Conv2d(width, 3, 1)

    def forward(self, x):
        source = x
        skips = []
        for layer in self.enc:
            x = layer(x)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.mid(x)
        for layer, skip in zip(self.dec, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = layer(torch.cat([x, skip], dim=1))
        prediction = self.out(x)
        if self.output_mode == "residual":
            return (source + prediction).clamp(-1, 1)
        return torch.tanh(prediction)


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels, affine=True), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.layers(x)


class ResnetGenerator(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3), nn.Conv2d(3, width, 7),
            nn.InstanceNorm2d(width, affine=True), nn.ReLU(),
        ]
        for a, b in ((width, width * 2), (width * 2, width * 4)):
            layers += [
                nn.Conv2d(a, b, 3, stride=2, padding=1),
                nn.InstanceNorm2d(b, affine=True), nn.ReLU(),
            ]
        layers += [Residual(width * 4) for _ in range(9)]
        for a, b in ((width * 4, width * 2), (width * 2, width)):
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(a, b, 3, padding=1),
                nn.InstanceNorm2d(b, affine=True), nn.ReLU(),
            ]
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(width, 3, 7), nn.Tanh()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def make_generator(approach, width=64, output_mode="residual"):
    if approach == "unet":
        return PencilUNet(width, output_mode=output_mode)
    if approach == "gan":
        return ResnetGenerator(width)
    raise ValueError(approach)
'''


BACKEND_SOURCE = r'''"""Safe local inference for the two-model comparison bundle."""
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .models import make_generator


def rgb_on_white(image):
    image = ImageOps.exif_transpose(image)
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    return Image.alpha_composite(background, rgba).convert("RGB")


def fit_canvas(image, size):
    image = rgb_on_white(image)
    scale = size / max(image.size)
    width, height = [max(1, min(size, round(value * scale))) for value in image.size]
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    left, top = (size - width) // 2, (size - height) // 2
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(resized, (left, top))
    return canvas, (left, top, left + width, top + height)


class ComparisonEditor:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") != "sketch2photo-two-model-comparison-v1":
            raise ValueError("Wrong bundle format")
        self.device = (
            "cuda" if torch.cuda.is_available() else
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
            "cpu"
        )
        self.model = None
        self.key = None

    def _load(self, direction, approach):
        key = (direction, approach)
        if key == self.key:
            return
        self.model = None
        self.key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        info = self.manifest["models"][direction][approach]
        state = torch.load(self.root / info["path"], map_location="cpu", weights_only=True)
        if state.get("direction") != direction or state.get("approach") != approach:
            raise RuntimeError("Checkpoint metadata does not match the requested model")
        self.model = make_generator(
            approach,
            int(state.get("width", 64)),
            state.get("output_mode", "residual"),
        )
        self.model.load_state_dict(state["generator"], strict=True)
        self.model.to(self.device).eval()
        self.key = key

    def predict(self, image, direction, approach):
        if direction not in {"photo_to_pencil", "sketch_to_photo"}:
            raise ValueError(direction)
        if approach not in {"unet", "gan"}:
            raise ValueError(approach)
        self._load(direction, approach)
        info = self.manifest["models"][direction][approach]
        canvas, box = fit_canvas(image, int(info["resolution"]))
        if direction == "sketch_to_photo":
            canvas = canvas.convert("L").convert("RGB")
        array = np.asarray(canvas, dtype=np.float32).copy().transpose(2, 0, 1)
        tensor = torch.from_numpy(array / 127.5 - 1.0)[None].to(self.device)
        with torch.inference_mode():
            output = self.model(tensor)[0].float().cpu().clamp(-1, 1)
        result = Image.fromarray(
            ((output.permute(1, 2, 0).numpy() + 1) * 127.5).round().astype("uint8")
        ).crop(box)
        if direction == "photo_to_pencil":
            result = result.convert("L").convert("RGB")
        return result
'''


APP_SOURCE = r'''"""Streamlit app for an honest U-Net versus GAN comparison."""
import io
import json
from pathlib import Path

from PIL import Image
import streamlit as st

from src.backend import ComparisonEditor


ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "manifest.json").read_text())


@st.cache_resource(show_spinner=False)
def editor():
    return ComparisonEditor(ROOT)


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


st.set_page_config(page_title="Sketch2Photo model comparison", page_icon="✏️", layout="wide")
st.title("Sketch2Photo — U-Net versus GAN")
st.warning(
    "These are actual experimental outputs. Validation-best does not mean production quality, "
    "and known blur, colour uncertainty, identity drift, and scene failures must be reported."
)

direction_label = st.radio(
    "Direction", ["Photo → Pencil", "Pencil → Colour Photo"], horizontal=True
)
direction = "photo_to_pencil" if direction_label == "Photo → Pencil" else "sketch_to_photo"
uploaded = st.file_uploader("Upload a JPG, PNG, or WebP image", type=["jpg", "jpeg", "png", "webp"])

if st.button("Compare both trained models", type="primary", use_container_width=True):
    if uploaded is None:
        st.error("Upload an image first.")
    else:
        with Image.open(uploaded) as opened:
            source = opened.copy()
        with st.spinner("Running U-Net and GAN…"):
            unet = editor().predict(source, direction, "unet")
            gan = editor().predict(source, direction, "gan")
        st.session_state["comparison"] = (source, unet, gan, direction)

comparison = st.session_state.get("comparison")
if comparison is not None:
    source, unet, gan, generated_direction = comparison
    columns = st.columns(3)
    columns[0].image(source, caption="Input", use_container_width=True)
    columns[1].image(unet, caption="Approach A — U-Net", use_container_width=True)
    columns[2].image(gan, caption="Approach B — Residual GAN", use_container_width=True)
    extension = "pencil" if generated_direction == "photo_to_pencil" else "photo"
    first, second = st.columns(2)
    first.download_button(
        "Download U-Net output", png_bytes(unet), f"unet_{extension}.png", "image/png",
        use_container_width=True,
    )
    second.download_button(
        "Download GAN output", png_bytes(gan), f"gan_{extension}.png", "image/png",
        use_container_width=True,
    )

st.divider()
st.subheader("Saved validation selection")
for approach in ("unet", "gan"):
    info = MANIFEST["models"][direction][approach]
    st.write(
        f"**{approach.upper()}** — selected step {info['selected_step']}; "
        f"validation LPIPS {info['validation_lpips']:.6f}"
    )
st.caption("Lower LPIPS is better, but metrics do not replace visual inspection.")
'''

(source_root / "__init__.py").write_text('"""Two-model comparison package."""\n')
(source_root / "models.py").write_text(MODELS_SOURCE)
(source_root / "backend.py").write_text(BACKEND_SOURCE)
(export_root / "app.py").write_text(APP_SOURCE)
(export_root / "requirements.txt").write_text(
    "torch==2.8.0\nPillow==11.3.0\nnumpy==2.2.6\nstreamlit==1.65.0\n"
)
(export_root / "README.md").write_text(
    "# Sketch2Photo two-model comparison\n\n"
    "This bundle contains the completed fair pilot checkpoints for two distinct trained "
    "approaches in both directions: a skip-connected U-Net and a residual conditional GAN.\n\n"
    "Run locally with:\n\n"
    "```bash\npython -m pip install -r requirements.txt\nstreamlit run app.py\n```\n\n"
    "The outputs are experimental and include documented failures. Do not describe them as "
    "production quality or claim that a monochrome sketch uniquely determines its source photo.\n"
)

archive = Path(shutil.make_archive(str(export_root), "zip", root_dir=export_root))
print("Comparison folder:", export_root)
print("Comparison ZIP:", archive)
print("Defaults selected by pilot validation LPIPS:", defaults)

try:
    from google.colab import files

    files.download(str(archive))
except ImportError:
    print("Not running in Colab; download the ZIP manually.")
