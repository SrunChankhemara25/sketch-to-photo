"""Single-page Streamlit interface for all deployed checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import streamlit as st

from ..config import PHOTO_TO_PENCIL_ROOT, SKETCH_TO_PHOTO_ROOT, SUPPORTED_IMAGE_TYPES
from ..image_processing import png_bytes
from ..inference import PhotoToPencilService, SketchToPhotoService


PHOTO_TO_PENCIL = "Photo → Pencil"
SKETCH_TO_PHOTO = "Pencil → Colour Photo"


def _read_manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text())


@st.cache_resource(show_spinner=False)
def _photo_service() -> PhotoToPencilService:
    return PhotoToPencilService(PHOTO_TO_PENCIL_ROOT)


@st.cache_resource(show_spinner=False)
def _sketch_service() -> SketchToPhotoService:
    return SketchToPhotoService(SKETCH_TO_PHOTO_ROOT)


def _uploaded_image(uploaded_file) -> Image.Image | None:
    if uploaded_file is None:
        return None
    with Image.open(uploaded_file) as image:
        return image.copy()


def _model_options(direction: str) -> dict[str, str]:
    if direction == PHOTO_TO_PENCIL:
        manifest = _read_manifest(PHOTO_TO_PENCIL_ROOT)
        return {info["label"]: key for key, info in manifest["models"].items()}

    manifest = _read_manifest(SKETCH_TO_PHOTO_ROOT)
    return {info["label"]: key for key, info in manifest["models"].items()}


def _render_model_summary(direction: str, model_key: str) -> None:
    if direction == PHOTO_TO_PENCIL:
        info = _read_manifest(PHOTO_TO_PENCIL_ROOT)["models"][model_key]
        st.caption(
            f"Validation LPIPS {info['validation_lpips']:.4f} · selected step "
            f"{info['selected_step']} · {info['resolution']}px training"
        )
    else:
        info = _read_manifest(SKETCH_TO_PHOTO_ROOT)["models"][model_key]
        validation = info["validation"]
        st.caption(
            f"Validation LPIPS {validation['lpips']:.4f} · MAE {validation['mae']:.4f} · "
            f"selected step {info['selected_step']} · {info['resolution']}px training"
        )


def _run_inference(image, direction, model_key, detail_strength):
    if direction == PHOTO_TO_PENCIL:
        result, details = _photo_service().predict(image, model_key)
        return result, None, details
    return _sketch_service().predict(image, model_key, detail_strength)


def _render_result(direction: str, signature: tuple[str, str, float]) -> None:
    state = st.session_state.get("generation")
    if not state or state["signature"] != signature:
        return
    left, right = st.columns(2)
    left.image(state["input"], caption="Input", width="stretch")
    caption = "Generated pencil drawing" if direction == PHOTO_TO_PENCIL else "Generated colour photo"
    right.image(state["result"], caption=caption, width="stretch")
    filename = "generated_pencil.png" if direction == PHOTO_TO_PENCIL else "generated_photo.png"
    st.download_button(
        "Download generated PNG", png_bytes(state["result"]), filename, "image/png", width="stretch"
    )
    details = state["details"]
    st.caption(
        f"Model {details['model']} · selected step {details['step']} · "
        f"{details['resolution']}px network · device {details['device']}"
    )
    if state["technical"] is not None:
        with st.expander("Technical network input"):
            st.image(state["technical"], width=320)


def render_app() -> None:
    st.set_page_config(
        page_title="Sketch2Photo Studio", page_icon="✏️", layout="wide", initial_sidebar_state="collapsed"
    )
    st.markdown(
        """<style>
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none;}
.hero {padding: 1.35rem 1.5rem; border-radius: 18px; margin-bottom: 1.2rem;
  background: linear-gradient(120deg,#201b2c,#522b47 55%,#8a4a2d); color:white;}
.hero h1 {margin:0; font-size:2.25rem;}
.hero p {margin:.4rem 0 0; opacity:.88;}
</style><div class="hero"><h1>Sketch2Photo Studio</h1><p>Run validation-selected models for photo-to-pencil and sketch-to-photo generation.</p></div>""",
        unsafe_allow_html=True,
    )

    direction = st.segmented_control(
        "Direction", [PHOTO_TO_PENCIL, SKETCH_TO_PHOTO], default=SKETCH_TO_PHOTO, key="direction"
    )
    options = _model_options(direction)
    selected_label = st.selectbox("Trained model", list(options), key=f"model_{direction}")
    model_key = options[selected_label]
    _render_model_summary(direction, model_key)

    detail_strength = 0.0
    if direction == SKETCH_TO_PHOTO:
        detail_strength = st.slider(
            "Detail enhancement", 0.0, 1.0, 0.35, 0.05,
            help="0 shows the raw model prediction. Higher values sharpen edges and gently transfer sketch lines.",
        )
        st.caption(
            "Detail enhancement reduces visible softness but cannot recreate colours, texture or identity "
            "that are absent from the sketch or were not learned by the model."
        )

    upload_label = "Upload a colour photo" if direction == PHOTO_TO_PENCIL else "Upload a pencil sketch"
    uploaded = st.file_uploader(upload_label, type=SUPPORTED_IMAGE_TYPES, key=f"upload_{direction}")
    button_label = "Generate pencil drawing" if direction == PHOTO_TO_PENCIL else "Generate colour photo"
    if st.button(button_label, type="primary", width="stretch"):
        image = _uploaded_image(uploaded)
        if image is None:
            st.error(f"{upload_label} first.")
        else:
            try:
                with st.spinner(f"Running {selected_label}…"):
                    result, technical, details = _run_inference(
                        image, direction, model_key, detail_strength
                    )
                st.session_state.generation = {
                    "signature": (direction, model_key, detail_strength),
                    "input": image, "result": result, "technical": technical, "details": details,
                }
            except Exception as error:
                st.exception(error)

    _render_result(direction, (direction, model_key, detail_strength))
    st.divider()
    st.caption(
        "Generated images are model estimates. Lower LPIPS/MAE is better, but metrics must be "
        "considered together with visual inspection."
    )
