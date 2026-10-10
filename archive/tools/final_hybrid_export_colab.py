"""Build the final domain-safe Streamlit bundle from completed Colab checkpoints.

Upload this file to the active Colab session and execute it with:
    %run /content/final_hybrid_export_colab.py

Google Drive must already be mounted at /content/drive. No dataset is read and no
training is performed. The script copies only the selected checkpoints and reports.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import torch


DRIVE_ROOT = Path("/content/drive/MyDrive/Sketch2Photo_training")
GENERAL_ROOT = DRIVE_ROOT / "general_20k_bidirectional_v1" / "general_rescue"
GENERAL_SELECTION = GENERAL_ROOT / "selected_models.json"


def resolve_saved_path(value: str | Path) -> Path:
    """Resolve absolute paths saved by a different Colab/Drive account."""
    candidate = Path(value)
    if candidate.is_file():
        return candidate
    parts = candidate.parts
    if "Sketch2Photo_training" in parts:
        index = parts.index("Sketch2Photo_training")
        remapped = DRIVE_ROOT.joinpath(*parts[index + 1 :])
        if remapped.is_file():
            return remapped
    raise FileNotFoundError(f"Checkpoint is unavailable: {candidate}")


assert DRIVE_ROOT.is_dir(), f"Drive folder is unavailable: {DRIVE_ROOT}"
assert GENERAL_SELECTION.is_file(), f"Run completed Section 8 first: {GENERAL_SELECTION}"

selection = json.loads(GENERAL_SELECTION.read_text())
photo_info = selection["photo_to_pencil"]["unet"]
sketch_info = selection["sketch_to_photo"]["unet"]

photo_general = resolve_saved_path(photo_info["path"])
photo_portrait = resolve_saved_path(photo_info["warm_started_from"])
sketch_general = resolve_saved_path(sketch_info["path"])
sketch_portrait = resolve_saved_path(sketch_info["warm_started_from"])

states = {
    "photo_general": torch.load(photo_general, map_location="cpu", weights_only=True),
    "photo_portrait": torch.load(photo_portrait, map_location="cpu", weights_only=True),
    "sketch_general": torch.load(sketch_general, map_location="cpu", weights_only=True),
    "sketch_portrait": torch.load(sketch_portrait, map_location="cpu", weights_only=True),
}

assert states["photo_general"]["direction"] == "photo_to_pencil"
assert states["photo_portrait"]["direction"] == "photo_to_pencil"
assert states["sketch_general"]["direction"] == "sketch_to_photo"
assert states["sketch_portrait"]["direction"] == "sketch_to_photo"
assert states["photo_general"]["step"] == 21694, "Expected approved photo checkpoint at step 21694."
assert states["photo_portrait"]["step"] == 2560, "Expected portrait photo checkpoint at step 2560."
assert states["sketch_general"]["step"] == 18000, "Expected general-scene sketch checkpoint at step 18000."

stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
export_root = DRIVE_ROOT / f"final_hybrid_streamlit_{stamp}"
model_root = export_root / "models"
report_root = export_root / "reports"
source_root = export_root / "src" / "approaches"
for folder in (model_root, report_root, source_root):
    folder.mkdir(parents=True, exist_ok=False)

photo_destination = model_root / "photo_to_pencil_general_best.pt"
photo_portrait_destination = model_root / "photo_to_pencil_portrait_best.pt"
sketch_general_destination = model_root / "sketch_to_photo_general_best.pt"
sketch_portrait_destination = model_root / "sketch_to_photo_portrait_fallback.pt"
shutil.copy2(photo_general, photo_destination)
shutil.copy2(photo_portrait, photo_portrait_destination)
shutil.copy2(sketch_general, sketch_general_destination)
shutil.copy2(sketch_portrait, sketch_portrait_destination)

for name in (
    "photo_to_pencil_general_coco_visual_gate.png",
    "photo_to_pencil_general_fs2k_visual_gate.png",
    "sketch_to_photo_general_coco_visual_gate.png",
    "sketch_to_photo_general_fs2k_visual_gate.png",
):
    source = GENERAL_ROOT / name
    if source.is_file():
        shutil.copy2(source, report_root / name)

for direction in ("photo_to_pencil", "sketch_to_photo"):
    history = GENERAL_ROOT / "unet_multiscale_patchgan" / direction / "history.json"
    if history.is_file():
        shutil.copy2(history, report_root / f"{direction}_history.json")

manifest = {
    "format": "sketch2photo-hybrid-v1",
    "resolution": 512,
    "routes": {
        "photo_to_pencil": {
            "general": "models/photo_to_pencil_general_best.pt",
            "portrait": "models/photo_to_pencil_portrait_best.pt",
        },
        "sketch_to_photo": {
            "general": "models/sketch_to_photo_general_best.pt",
            "portrait": "models/sketch_to_photo_portrait_fallback.pt",
        },
    },
    "checkpoint_steps": {
        "photo_to_pencil_general": states["photo_general"]["step"],
        "photo_to_pencil_portrait": states["photo_portrait"]["step"],
        "sketch_to_photo_general": states["sketch_general"]["step"],
        "sketch_to_photo_portrait": states["sketch_portrait"]["step"],
    },
    "selection_reason": {
        "photo_to_pencil_general": "General best at 21694 passed the diverse-scene visual gate.",
        "photo_to_pencil_portrait": "Earlier face specialist at 2560 avoids the over-dark portrait details from general training.",
        "sketch_to_photo_general": "Overall LPIPS-selected step 18000; final step 21694 regressed.",
        "sketch_to_photo_portrait": "Pre-general portrait fallback avoids the severe magenta FS2K failure.",
    },
    "limitations": [
        "Sketches do not uniquely determine original colours.",
        "The portrait fallback is safer but can be softer than the general-scene model.",
        "Content type is selected explicitly because automatic face detection on sketches is unreliable.",
    ],
}
(export_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

UNET_SOURCE = '''"""Checkpoint-compatible U-Net used by the final Sketch2Photo bundle."""
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
            raise ValueError(f"Unsupported output mode: {output_mode}")
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
        if output_mode == "residual":
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

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
'''

INIT_SOURCE = '''from .unet import PencilUNet


def make_generator(width=64, output_mode="residual"):
    return PencilUNet(width=width, output_mode=output_mode)
'''

BACKEND_SOURCE = '''"""Domain-safe inference for the final hybrid Sketch2Photo bundle."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .approaches import make_generator


def rgb_on_white(image):
    image = ImageOps.exif_transpose(image)
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, "white")
    return Image.alpha_composite(white, rgba).convert("RGB")


def fit_canvas(image, size=512):
    image = rgb_on_white(image)
    scale = size / max(image.size)
    width, height = [max(1, min(size, round(value * scale))) for value in image.size]
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    left, top = (size - width) // 2, (size - height) // 2
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(resized, (left, top))
    return canvas, (left, top, left + width, top + height)


class HybridEditor:
    def __init__(self, bundle_dir, device=None):
        self.root = Path(bundle_dir)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") != "sketch2photo-hybrid-v1":
            raise ValueError("This is not a final hybrid Sketch2Photo bundle.")
        self.device = device or (
            "cuda" if torch.cuda.is_available() else
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
            "cpu"
        )
        self.model = None
        self.key = None
        self.state = None

    def _load(self, direction, content_type):
        key = (direction, content_type)
        if key == self.key:
            return
        self.model = None
        self.state = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        relative = self.manifest["routes"][direction][content_type]
        state = torch.load(self.root / relative, map_location="cpu", weights_only=True)
        if state["direction"] != direction:
            raise ValueError("Checkpoint direction does not match the requested operation.")
        model = make_generator(state["width"], state.get("output_mode", "residual"))
        model.load_state_dict(state["generator"], strict=True)
        self.model = model.to(self.device).eval()
        self.state = state
        self.key = key

    def __call__(self, image, direction, content_type):
        if image is None:
            raise ValueError("Upload an image first.")
        if direction not in {"photo_to_pencil", "sketch_to_photo"}:
            raise ValueError("Choose a supported direction.")
        if content_type not in {"general", "portrait"}:
            raise ValueError("Choose General scene or Portrait.")
        self._load(direction, content_type)
        canvas, box = fit_canvas(image, self.manifest["resolution"])
        if direction == "sketch_to_photo":
            canvas = canvas.convert("L").convert("RGB")
        array = np.asarray(canvas, dtype=np.float32).copy().transpose(2, 0, 1) / 127.5 - 1.0
        tensor = torch.from_numpy(array)[None].to(self.device)
        with torch.inference_mode():
            prediction = self.model(tensor)[0].float().cpu().clamp(-1, 1)
        pixels = ((prediction.permute(1, 2, 0).numpy() + 1) * 127.5).round().astype("uint8")
        output = Image.fromarray(pixels).crop(box)
        if direction == "photo_to_pencil":
            output = output.convert("L").convert("RGB")
        return output, int(self.state.get("step", -1))
'''

APP_SOURCE = '''"""Final Streamlit UI for the domain-safe Sketch2Photo bundle."""
from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
from PIL import Image

from src.hybrid_backend import HybridEditor


ROOT = Path(__file__).resolve().parent


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@st.cache_resource(show_spinner=False)
def editor():
    return HybridEditor(ROOT)


st.set_page_config(page_title="Sketch2Photo Studio", page_icon="✏️", layout="wide")
st.title("✏️ Sketch2Photo Studio")
st.caption("Domain-safe routing between the final validated checkpoints")

direction_label = st.radio(
    "Direction", ["Photo → Pencil", "Pencil → Colour Photo"], horizontal=True
)
content_label = st.radio("Content type", ["General scene", "Portrait"], horizontal=True)
direction = "photo_to_pencil" if direction_label == "Photo → Pencil" else "sketch_to_photo"
content_type = "general" if content_label == "General scene" else "portrait"

if direction == "sketch_to_photo" and content_type == "portrait":
    st.info("Portrait-safe fallback is selected to prevent the magenta failure found in final validation.")

uploaded = st.file_uploader("Upload a JPG, PNG, or WebP image", type=["jpg", "jpeg", "png", "webp"])
if st.button("Generate", type="primary", use_container_width=True):
    if uploaded is None:
        st.error("Upload an image first.")
    else:
        with Image.open(uploaded) as source:
            source = source.copy()
        with st.spinner("Running the selected trained checkpoint…"):
            result, step = editor()(source, direction, content_type)
        st.session_state["source"] = source
        st.session_state["result"] = result
        st.session_state["step"] = step
        st.session_state["direction"] = direction

if "result" in st.session_state:
    left, right = st.columns(2)
    left.image(st.session_state["source"], caption="Input", use_container_width=True)
    right.image(st.session_state["result"], caption="Generated output", use_container_width=True)
    st.caption(f"Checkpoint step: {st.session_state['step']}")
    filename = "pencil_drawing.png" if st.session_state["direction"] == "photo_to_pencil" else "colour_photo.png"
    st.download_button(
        "Download generated PNG", png_bytes(st.session_state["result"]), filename,
        "image/png", use_container_width=True,
    )

st.divider()
st.caption("A grayscale sketch does not uniquely determine its original colours. Outputs are model estimates.")
'''

(source_root / "unet.py").write_text(UNET_SOURCE)
(source_root / "__init__.py").write_text(INIT_SOURCE)
(export_root / "src" / "__init__.py").write_text('"""Final Sketch2Photo inference package."""\n')
(export_root / "src" / "hybrid_backend.py").write_text(BACKEND_SOURCE)
(export_root / "app.py").write_text(APP_SOURCE)
(export_root / "requirements.txt").write_text(
    "torch>=2.2,<3\nnumpy>=1.26\nPillow>=10\nstreamlit>=1.40,<2\n"
)
(export_root / "README.md").write_text(
    "# Final Sketch2Photo hybrid app\n\n"
    "This bundle uses the validated general checkpoints and the safer portrait fallback.\n\n"
    "```bash\npython -m pip install -r requirements.txt\nstreamlit run app.py\n```\n"
)

archive = shutil.make_archive(str(export_root), "zip", root_dir=export_root)
print("Final hybrid folder:", export_root)
print("Final hybrid ZIP:", archive)
print("Selected checkpoint steps:", manifest["checkpoint_steps"])

try:
    from google.colab import files

    files.download(archive)
except Exception as error:
    print("Automatic browser download did not start:", error)
    print("Download the ZIP manually from Drive:", archive)
