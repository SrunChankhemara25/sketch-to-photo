"""Streamlit web application for original checkpoints and new Colab exports."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

from PIL import Image
import streamlit as st


ROOT = Path(__file__).resolve().parent
COMPARISON_BUNDLE = ROOT / 'checkpoints' / 'two_model_comparison'
FALLBACK_BUNDLE = ROOT / 'checkpoints' / 'quality_study'
DEFAULT_BUNDLE = Path(os.getenv(
    'SKETCH2PHOTO_BUNDLE',
    COMPARISON_BUNDLE if (COMPARISON_BUNDLE / 'manifest.json').is_file() else FALLBACK_BUNDLE,
))
if (ROOT / 'manifest.json').is_file():
    DEFAULT_BUNDLE = ROOT


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def uploaded_image(uploaded) -> Image.Image | None:
    if uploaded is None:
        return None
    with Image.open(uploaded) as image:
        return image.copy()


def render_original(st):
    """Original 128px checkpoints plus the classical pencil renderer."""
    from src import application as legacy

    st.info(
        'Original mode uses the supplied 128×128 sketch-to-photo checkpoints. '
        'Photo-to-pencil is the existing image-processing renderer, not a trained model.'
    )
    checkpoints = legacy.list_checkpoints()
    with st.sidebar:
        st.subheader('Original model')
        checkpoint = st.selectbox(
            'Checkpoint', checkpoints,
            index=checkpoints.index(legacy.default_checkpoint(checkpoints)) if checkpoints else None,
            disabled=not checkpoints,
        )
        guessed = legacy.guess_arch(checkpoint)
        architecture = st.selectbox(
            'Architecture', legacy.ARCH_NAMES,
            index=legacy.ARCH_NAMES.index(guessed),
        )
        device = st.selectbox('Compute device', legacy.AVAILABLE_DEVICES,
                              index=legacy.AVAILABLE_DEVICES.index(legacy.DEFAULT_DEVICE))

    pencil_tab, colour_tab = st.tabs(['Photo → Pencil', 'Pencil → Colour Photo'])
    with pencil_tab:
        st.subheader('Create a pencil drawing')
        st.caption('The preview and downloaded PNG are exactly the same image.')
        source_file = st.file_uploader('Upload a photo', type=['png', 'jpg', 'jpeg', 'webp'], key='old_photo')
        c1, c2, c3 = st.columns(3)
        darkness = c1.slider('Pencil darkness', .5, 2.0, 1.05, .05)
        fade = c2.slider('Fade soft scenery', 0.0, 1.0, .6, .05)
        texture = c3.slider('Paper and pencil texture', 0.0, 1.0, .45, .05)
        if st.button('Create pencil drawing', type='primary', use_container_width=True):
            image = uploaded_image(source_file)
            if image is None:
                st.error('Upload a photo first.')
            else:
                with st.spinner('Rendering pencil drawing…'):
                    drawing, technical, pair = legacy.prepare_drawing_session(image, darkness, fade, texture)
                st.session_state.old_drawing = drawing
                st.session_state.old_technical = technical
                st.session_state.old_pair = pair
        drawing = st.session_state.get('old_drawing')
        if drawing is not None:
            left, right = st.columns(2)
            source = st.session_state['old_pair']['source']
            left.image(source, caption='Source photo', use_container_width=True)
            right.image(drawing, caption='Pencil drawing', use_container_width=True)
            st.download_button('Download pencil PNG', png_bytes(drawing), 'pencil_drawing.png',
                               'image/png', use_container_width=True)
            with st.expander('Technical input used by the original checkpoint'):
                st.image(st.session_state['old_technical'], width=320)

    with colour_tab:
        st.subheader('Generate a colour photo')
        use_created = st.checkbox('Use the pencil drawing created above',
                                  value=st.session_state.get('old_drawing') is not None,
                                  disabled=st.session_state.get('old_drawing') is None)
        colour_file = None if use_created else st.file_uploader(
            'Upload a white-paper pencil sketch or a photo', type=['png', 'jpg', 'jpeg', 'webp'], key='old_colour')
        input_kind = st.radio('Input type', [legacy.INPUT_PENCIL, legacy.INPUT_PHOTO], horizontal=True,
                              disabled=use_created)
        raw_colours = st.toggle('Raw model colours (recommended for evaluation)', value=True)
        if raw_colours:
            colour_strength, cast_reduction = 1.0, 0.0
        else:
            c1, c2 = st.columns(2)
            colour_strength = c1.slider('Display colour intensity', 0.0, 2.5, 1.5, .05)
            cast_reduction = c2.slider('Reduce cream/yellow cast', 0.0, 1.0, .35, .05)
        if st.button('Generate colour photo', type='primary', use_container_width=True):
            image = st.session_state.get('old_drawing') if use_created else uploaded_image(colour_file)
            if image is None:
                st.error('Create or upload a pencil drawing first.')
            elif not checkpoint:
                st.error('No original checkpoint is available.')
            else:
                try:
                    with st.spinner('Running the selected checkpoint…'):
                        result, technical, details = legacy.run_inference(
                            image, legacy.INPUT_PENCIL if use_created else input_kind,
                            legacy.OUTPUT_MODEL, checkpoint, architecture, device,
                            st.session_state.get('old_pair') if use_created else None,
                            colour_strength, cast_reduction,
                        )
                    st.session_state.old_colour_result = result
                    st.session_state.old_colour_input = technical
                    st.session_state.old_colour_details = details
                except Exception as error:
                    st.error(str(error))
        result = st.session_state.get('old_colour_result')
        if result is not None:
            left, right = st.columns(2)
            input_image = st.session_state.get('old_drawing') if use_created else uploaded_image(colour_file)
            if input_image is not None:
                left.image(input_image, caption='Pencil input', use_container_width=True)
            right.image(result, caption='Generated colour result', use_container_width=True)
            st.download_button('Download colour PNG', png_bytes(result), 'colour_result.png',
                               'image/png', use_container_width=True)
            st.caption(st.session_state.get('old_colour_details', ''))
            with st.expander('Technical network input'):
                st.image(st.session_state.get('old_colour_input'), width=320)


@st.cache_resource(show_spinner=False)
def get_study_editor(bundle: str):
    from src.study_backend import StudyEditor
    return StudyEditor(bundle)


@st.cache_resource(show_spinner=False)
def get_dual_editor(bundle: str):
    from src.diffusion_backend import DualEditor
    return DualEditor(bundle)


@st.cache_resource(show_spinner=False)
def get_comparison_editor(bundle: str):
    from src.comparison_backend import ComparisonEditor
    return ComparisonEditor(bundle)


def render_comparison(st, bundle: Path, manifest: dict):
    """Compare the two genuinely different pilot architectures side by side."""
    editor = get_comparison_editor(str(bundle))
    st.info(
        'Loaded the fair pilot comparison: **Approach A — U-Net** and '
        '**Approach B — residual generator trained with a conditional PatchGAN**. '
        'Both outputs are raw model predictions.'
    )
    st.warning(
        'These are experimental results, not production-quality claims. Known blur, '
        'colour uncertainty, identity drift and general-scene failures must be reported.'
    )
    direction_label = st.segmented_control(
        'Direction', ['Photo → Pencil', 'Pencil → Colour Photo'],
        default='Photo → Pencil', key='comparison_direction',
    )
    direction = 'photo_to_pencil' if direction_label == 'Photo → Pencil' else 'sketch_to_photo'
    uploaded = st.file_uploader(
        'Upload source image', type=['png', 'jpg', 'jpeg', 'webp'], key='comparison_source'
    )
    if st.button('Compare U-Net and GAN', type='primary', use_container_width=True):
        image = uploaded_image(uploaded)
        if image is None:
            st.error('Upload an image first.')
        else:
            try:
                with st.spinner('Running both trained models…'):
                    unet = editor.predict(image, direction, 'unet')
                    gan = editor.predict(image, direction, 'gan')
                st.session_state.comparison_result = (image, unet, gan, direction)
            except Exception as error:
                st.error(str(error))

    result = st.session_state.get('comparison_result')
    if result is not None:
        source, unet, gan, generated_direction = result
        columns = st.columns(3)
        columns[0].image(source, caption='Input', use_container_width=True)
        columns[1].image(unet, caption='Approach A — U-Net', use_container_width=True)
        columns[2].image(gan, caption='Approach B — Residual GAN', use_container_width=True)
        suffix = 'pencil' if generated_direction == 'photo_to_pencil' else 'photo'
        first, second = st.columns(2)
        first.download_button(
            'Download U-Net output', png_bytes(unet), f'unet_{suffix}.png',
            'image/png', use_container_width=True,
        )
        second.download_button(
            'Download GAN output', png_bytes(gan), f'gan_{suffix}.png',
            'image/png', use_container_width=True,
        )

    st.subheader('Saved pilot validation selection')
    for approach, label in (('unet', 'U-Net'), ('gan', 'Residual GAN')):
        info = manifest['models'][direction][approach]
        st.write(
            f"**{label}:** step {info['selected_step']}; "
            f"validation LPIPS {info['validation_lpips']:.6f}"
        )
    st.caption('Lower LPIPS is better; visual inspection is still required.')


def render_new(st, bundle: Path):
    """New three-approach study bundle or earlier two-adapter export."""
    import json
    manifest_path = bundle / 'manifest.json'
    if not manifest_path.is_file():
        st.warning(
            'No new trained export is installed yet. Run the master Colab notebook and '
            f'extract the ZIP contents into `{bundle}`.'
        )
        return
    manifest = json.loads(manifest_path.read_text())
    model_format = manifest.get('format')
    if model_format == 'sketch2photo-two-model-comparison-v1':
        render_comparison(st, bundle, manifest)
        return
    elif model_format == 'sketch2photo-study-v2':
        editor = get_study_editor(str(bundle))
        approaches = ['Validation best', 'U-Net', 'GAN', 'Diffusion']
    elif model_format == 'sketch2photo-ip2p-lora-v1':
        editor = get_dual_editor(str(bundle))
        approaches = ['Diffusion']
    else:
        st.error('The selected folder is not a recognized Sketch2Photo export.')
        return

    mode = manifest.get('mode', 'unknown')
    st.info(f'Loaded a **{mode}** export. “Validation best” means lowest held-out validation LPIPS, not perfect output.')
    direction_label = st.segmented_control(
        'Direction', ['Photo → Pencil', 'Pencil → Colour Photo'], default='Photo → Pencil')
    direction = 'photo_to_pencil' if direction_label == 'Photo → Pencil' else 'sketch_to_photo'
    uploaded = st.file_uploader('Upload source image', type=['png', 'jpg', 'jpeg', 'webp'], key='new_source')
    c1, c2 = st.columns(2)
    approach_label = c1.selectbox('Trained approach', approaches)
    seed = c2.number_input('Diffusion seed', min_value=0, max_value=2**31-1, value=42, step=1)
    description = st.text_input('Optional colour/content guidance (diffusion only)',
                                placeholder='Example: red jacket, green plants, warm daylight')
    if st.button('Generate result', type='primary', use_container_width=True):
        image = uploaded_image(uploaded)
        if image is None:
            st.error('Upload an image first.')
        else:
            mapping = {'Validation best':'validation_best', 'U-Net':'unet', 'GAN':'gan', 'Diffusion':'diffusion'}
            try:
                with st.spinner('Generating with the trained model…'):
                    if model_format == 'sketch2photo-study-v2':
                        result = editor(image, direction, mapping[approach_label], int(seed), description)
                    else:
                        result = editor(image, direction, int(seed), description)
                st.session_state.new_result = result
                st.session_state.new_input = image
                st.session_state.new_direction = direction
            except Exception as error:
                st.error(str(error))
    result = st.session_state.get('new_result')
    if result is not None:
        left, right = st.columns(2)
        left.image(st.session_state['new_input'], caption='Input', use_container_width=True)
        right.image(result, caption='Actual generated output', use_container_width=True)
        output_direction = st.session_state.get('new_direction', direction)
        filename = 'pencil_drawing.png' if output_direction == 'photo_to_pencil' else 'colour_photo.png'
        st.download_button('Download generated PNG', png_bytes(result), filename, 'image/png',
                           use_container_width=True)
        st.caption('The displayed preview and downloaded file are the same model output.')


def render():
    st.set_page_config(page_title='Sketch2Photo Studio', page_icon='✏️', layout='wide')
    st.markdown('''<style>
    .block-container {max-width: 1240px; padding-top: 2rem;}
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    .hero {padding: 1.25rem 1.4rem; border-radius: 18px; margin-bottom: 1rem;
      background: linear-gradient(120deg,#201b2c,#522b47 55%,#8a4a2d); color:white;}
    .hero h1 {margin:0; font-size:2.25rem;} .hero p {margin:.35rem 0 0; opacity:.85;}
    </style><div class="hero"><h1>Sketch2Photo Studio</h1><p>Train once in Colab. Compare models and create pencil drawings or colour photographs on the web.</p></div>''',
                unsafe_allow_html=True)
    bundle_text = st.sidebar.text_input('New model bundle', str(DEFAULT_BUNDLE))
    bundle = Path(bundle_text).expanduser().resolve()
    new_available = (bundle / 'manifest.json').is_file()
    legacy_available = (ROOT / 'src' / 'application.py').is_file() and (ROOT / 'src' / 'models.py').is_file()
    choices = ['New trained models'] + (['Original checkpoints'] if legacy_available else [])
    engine = st.sidebar.radio('Model source', choices, index=0 if new_available or not legacy_available else 1)
    st.sidebar.caption('The original weights are kept. Install a Colab export to enable the new trained models.')
    st.sidebar.divider()
    st.sidebar.caption('PyTorch · Streamlit · fair 512px comparison')
    if engine == 'New trained models':
        render_new(st, bundle)
    else:
        render_original(st)
    st.divider()
    st.caption('A grayscale sketch does not uniquely contain the original colours. Results are model estimates.')


if __name__ == '__main__':
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        running_in_streamlit = get_script_run_ctx(suppress_warning=True) is not None
    except ImportError:
        running_in_streamlit = False
    if running_in_streamlit:
        render()
    else:
        os.execv(sys.executable, [sys.executable, '-m', 'streamlit', 'run', str(Path(__file__).resolve()), *sys.argv[1:]])
