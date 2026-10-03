"""
Sketch-to-Photo — Local Testing UI (Gradio)

Loads one of your trained checkpoints (AutoEncoder / U-Net / ResNet-ED) and runs it on
the *same image representation and resolution used during training*.

This project contains a one-way learned model: pencil sketch -> colour photo.  Photo ->
pencil sketch is therefore done with OpenCV. The app offers both a high-resolution clean
line drawing for viewing/downloading and the separate dark pencil representation used to
make the model's training inputs. It is deliberately not passed through the neural network
in reverse (the checkpoint has not been trained to do that).

Drop your trained checkpoint(s) into the checkpoints/ folder before running. These are the
files the training notebook's train_model() saves — e.g. ModelA_AutoEncoder_best.pt,
ModelB_UNet_best.pt, ModelC_ResNetED_best.pt, or the *_Tuned_best.pt versions. Any filename
works as long as you pick the matching architecture in the dropdown if auto-detection guesses
wrong.
"""

import os
import glob
import time
import hashlib

import numpy as np
import torch
import cv2
from PIL import Image, ImageOps
import gradio as gr

from .models import SimpleAutoEncoder, UNet, ResNetED

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# CUDA -> Apple Silicon (MPS) -> CPU, whichever is available on this machine, used as the
# default selection below. Some PyTorch versions have had real numerical bugs on MPS
# specifically for ConvTranspose2d/Upsample (used in these decoders), so a manual CPU
# override is offered in the UI as a way to rule that out if results look wrong.
AVAILABLE_DEVICES = ['CPU']
if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
    AVAILABLE_DEVICES.insert(0, 'MPS')
if torch.cuda.is_available():
    AVAILABLE_DEVICES.insert(0, 'CUDA')
DEFAULT_DEVICE = AVAILABLE_DEVICES[0]

ARCH_MAP = {
    'Model A — AutoEncoder': SimpleAutoEncoder,
    'Model B — U-Net':       UNet,
    'Model C — ResNet-ED':   ResNetED,
}
ARCH_NAMES = list(ARCH_MAP.keys())

# Filename keywords used to auto-guess the architecture from a checkpoint's name.
ARCH_KEYWORDS = {
    'autoencoder': 'Model A — AutoEncoder',
    'unet':        'Model B — U-Net',
    'resneted':    'Model C — ResNet-ED',
}

# These values come from results/final_report.json / the training notebook configuration.
# All checkpoints in this folder were trained after first making a 256px pencil sketch and
# then resizing both the sketch and target image to 128px.  A fully-convolutional network
# can accept another size, but that is an out-of-distribution input and noticeably hurts
# results, so the UI intentionally does not offer a resolution selector.
STORED_SIZE = 256
TRAIN_RESOLUTION = 128
PENCIL_SIGMA_S = 60
PENCIL_SIGMA_R = 0.07

INPUT_PENCIL = 'Pencil sketch (already prepared)'
INPUT_PHOTO = 'Photo (make the model-ready pencil sketch automatically)'
OUTPUT_MODEL = 'AI estimate from sketch (selected checkpoint)'
OUTPUT_SOURCE = 'Restore original photo from Step 1 (uses saved photo, not AI)'

def resolve_device(device_str):
    device_key = device_str.lower()
    if device_key == 'mps' and not (
        getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()
    ):
        raise gr.Error('MPS is not available in this PyTorch installation. Select CPU and try again.')
    if device_key == 'cuda' and not torch.cuda.is_available():
        raise gr.Error('CUDA is not available in this PyTorch installation. Select CPU and try again.')
    return torch.device(device_key)


# Cache of the currently-loaded model so repeated inferences don't reload from disk every time.
# Keyed on device too, so switching the device dropdown reloads onto the new device.
_loaded = {'path': None, 'arch': None, 'device': None, 'model': None, 'config': {}}


def list_checkpoints():
    files = sorted(glob.glob(os.path.join(CHECKPOINTS_DIR, '*.pt')))
    return [os.path.basename(f) for f in files]


def default_checkpoint(files=None):
    """Prefer the strongest exported model, while still working with any user checkpoint."""
    files = list_checkpoints() if files is None else files
    # The tuned U-Net has the lowest validation L1 in this project's exported results.
    for preferred in ('ModelB_UNet_Tuned_best.pt', 'ModelB_UNet_best.pt'):
        if preferred in files:
            return preferred
    return files[0] if files else None


def guess_arch(filename):
    if not filename:
        return ARCH_NAMES[0]
    lowered = filename.lower()
    for keyword, arch_name in ARCH_KEYWORDS.items():
        if keyword in lowered:
            return arch_name
    return ARCH_NAMES[0]


def load_model(checkpoint_name, arch_name, device):
    if not checkpoint_name:
        raise gr.Error(
            'No checkpoint selected. Drop a .pt file into the checkpoints/ folder, '
            'then click "Refresh checkpoints".'
        )
    if arch_name not in ARCH_MAP:
        raise gr.Error(f'Unknown architecture: {arch_name}')

    path = os.path.join(CHECKPOINTS_DIR, checkpoint_name)
    if _loaded['path'] == path and _loaded['arch'] == arch_name and _loaded['device'] == device:
        return _loaded['model']  # already loaded on this device, avoid reloading every click

    model = ARCH_MAP[arch_name]().to(device)

    # weights_only=False: this checkpoint dict also carries optimizer/scheduler state,
    # history, and config (plain Python objects, not just tensors), and it's a file you
    # generated yourself in the training notebook, so this is safe to disable here.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise gr.Error(
            f"This checkpoint doesn't match the '{arch_name}' architecture. "
            f'Try a different option in the Architecture selector. Details: {e}'
        )

    model.eval()
    config = ckpt.get('config', {}) if isinstance(ckpt, dict) else {}
    _loaded.update({'path': path, 'arch': arch_name, 'device': device, 'model': model,
                    'config': config})
    return model


def rgb_on_white(image):
    """Convert an upload to RGB without turning transparent drawing canvases black."""
    # Phone photos often store their orientation in EXIF instead of rotating the pixels.
    # Correct it before any OpenCV processing so portrait images are not treated sideways.
    image = ImageOps.exif_transpose(image)
    if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
        background = Image.new('RGBA', image.size, 'white')
        return Image.alpha_composite(background, image.convert('RGBA')).convert('RGB')
    return image.convert('RGB')


def resize_for_training(image, size):
    """Use the exact square resize convention used by the training data pipeline."""
    arr = np.asarray(image)
    if arr.shape[:2] != (size, size):
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return arr


def make_model_ready_pencil(photo_img):
    """Reproduce training preprocessing: RGB photo -> 256px OpenCV pencil sketch."""
    photo_rgb = rgb_on_white(photo_img)
    photo_rgb = resize_for_training(photo_rgb, STORED_SIZE)
    photo_bgr = cv2.cvtColor(photo_rgb, cv2.COLOR_RGB2BGR)
    sketch, _ = cv2.pencilSketch(
        photo_bgr, sigma_s=PENCIL_SIGMA_S, sigma_r=PENCIL_SIGMA_R
    )
    return Image.fromarray(sketch)


def make_clean_line_art(photo_img, graphite_strength=1.05, background_fade=0.6, pencil_texture=0.45):
    """Render continuous pencil values without tracing binary outlines around features.

    Three dark-only Gaussian differences retain fine marks, medium structure and broad
    shadows. A continuous tone curve keeps overlapping hair/flower marks from clipping to
    solid black. This is an image-processing effect; it does not synthesize an illustration.
    """
    source = rgb_on_white(photo_img)
    # Bound filter cost for camera uploads. Spatial parameters follow the working resolution
    # so a 4K photo receives the same visual treatment as a smaller upload.
    working = source.copy()
    working.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    gray = cv2.cvtColor(np.asarray(working), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    scale = max(working.size) / 924.0
    smooth = cv2.bilateralFilter(gray, d=5, sigmaColor=0.06, sigmaSpace=max(1.0, 3*scale))

    def blur(sigma):
        return cv2.GaussianBlur(smooth, (0, 0), sigmaX=max(0.5, sigma*scale))

    fine = np.maximum(blur(2) - smooth, 0.0)
    medium = np.maximum(blur(8) - smooth, 0.0)
    broad = np.maximum(blur(28) - smooth, 0.0)
    tone = np.square(1.0 - smooth)
    ink = 2.2*fine + 1.5*medium + 0.60*broad + 0.4*tone

    # A softly varying detail map fades blurry areas while preserving focused hair/eyes.
    # This is deliberately not called background removal: smooth subject areas can also
    # fade, and a sharply focused background will keep its detail.
    activity = cv2.GaussianBlur(np.abs(smooth - blur(3)), (0, 0), sigmaX=max(0.5, 12*scale))
    focus = np.clip((activity - 0.002) / 0.013, 0.0, 1.0)
    fade = float(np.clip(background_fade, 0.0, 1.0))
    ink *= (1.0-fade) + fade*focus
    strength = float(np.clip(graphite_strength, 0.5, 2.0))
    # Graphite is deposited unevenly on paper, with short directional marks in shadows.
    # Use deterministic irregular strokes rather than periodic diagonal stripes; the fixed
    # local RNG makes previews reproducible without changing the model's random state.
    texture = float(np.clip(pencil_texture, 0.0, 1.0))
    rng = np.random.default_rng(1947)
    grain = rng.standard_normal(gray.shape).astype(np.float32)
    stroke_length = max(3, int(round(7*scale))) | 1
    kernel = np.eye(stroke_length, dtype=np.float32) / stroke_length
    strokes = cv2.filter2D(grain, -1, kernel, borderType=cv2.BORDER_REFLECT)
    strokes *= np.sqrt(stroke_length)
    # Slight smoothing softens the digital outline while retaining fine eyelashes and hair.
    softened = cv2.GaussianBlur(ink, (0, 0), sigmaX=max(0.35, 0.55*scale))
    ink = (1.0 - 0.35*texture)*ink + 0.35*texture*softened
    density = ink * np.clip(1.0 + texture*(0.20*grain + 0.20*strokes), 0.25, 1.85)
    graphite = 255.0 * (1.0 - np.exp(-2.4*strength*density))
    # Grain is strongest in medium graphite shades and diminishes on blank paper.
    tooth = texture * 9.0 * grain * np.sqrt(np.clip(graphite/255.0, 0, 1))
    paper = 255.0 - texture*np.maximum(grain, 0)*0.65
    drawing = np.rint(paper - graphite + tooth).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(drawing)
    if result.size != source.size:
        result = result.resize(source.size, Image.Resampling.LANCZOS)
    return result


def sketch_array_for_training(sketch_img, resolution=TRAIN_RESOLUTION):
    """PIL image -> the exact uint8 grayscale array used as the model input."""
    sketch_np = np.asarray(rgb_on_white(sketch_img).convert('L'))
    if sketch_np.shape[:2] != (resolution, resolution):
        # INTER_AREA matches the downsampling method used in the training notebook.
        sketch_np = cv2.resize(
            sketch_np, (resolution, resolution), interpolation=cv2.INTER_AREA
        )
    return sketch_np


def photo_input_for_checkpoint(photo, config):
    """Use the preprocessing recipe saved by the notebook, retaining legacy defaults."""
    stored_size = int(config.get('stored_size', STORED_SIZE))
    rgb = resize_for_training(rgb_on_white(photo), stored_size)
    style = config.get('sketch_style', 'opencv_dark')
    if style == 'pencil_v5_white':
        if config.get('renderer_version', 'v5') != 'v5':
            raise gr.Error('This checkpoint requires a different pencil renderer version.')
        return make_clean_line_art(
            Image.fromarray(rgb), float(config.get('graphite_strength', 1.05)),
            float(config.get('background_fade', 0.6)), float(config.get('pencil_texture', 0.45)),
        )
    if style != 'opencv_dark':
        raise gr.Error(f'Unsupported checkpoint sketch style: {style}')
    sketch, _ = cv2.pencilSketch(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        sigma_s=float(config.get('sigma_s', PENCIL_SIGMA_S)),
        sigma_r=float(config.get('sigma_r', PENCIL_SIGMA_R)),
    )
    return Image.fromarray(sketch)


def match_training_sketch_polarity(sketch_np):
    """Invert clean white-paper drawings to match the checkpoint's dark sketch domain.

    ``cv2.pencilSketch`` (used to build the training pairs) commonly represents objects as
    light structure on a dark textured field. Hand drawings and downloaded pencil art are
    usually the opposite: graphite on mostly white paper. That polarity mismatch can make
    an otherwise valid checkpoint return an almost-black image.

    Returns the corrected image and whether an inversion was applied. We require both a
    bright median and a bright lower quartile, so ordinary photographs/dark model-ready
    sketches are not accidentally flipped because of one bright background region.
    """
    median = float(np.median(sketch_np))
    lower_quartile = float(np.percentile(sketch_np, 25))
    should_invert = median >= 205.0 and lower_quartile >= 150.0
    return (255 - sketch_np if should_invert else sketch_np), should_invert


def sketch_array_to_tensor(sketch_np, device):
    """A 128px grayscale array -> the 1x1x128x128 [0,1] training tensor."""
    x = torch.from_numpy(sketch_np.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return x.to(device)


def tensor_to_pil(pred_tensor):
    """(1,3,H,W) float tensor in [0,1] -> PIL RGB image."""
    arr = pred_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def enhance_predicted_colors(image, color_strength=1.0, cast_reduction=0.0):
    """Adjust existing predicted chroma; do not paint arbitrary new semantic colours.

    Cast reduction assumes the brightest part of the prediction should be approximately
    neutral. It is adjustable because sunset scenes and coloured backgrounds violate that
    assumption. Vibrance boosts weak colours more than already saturated colours.
    """
    rgb = np.asarray(image.convert('RGB')).astype(np.float32) / 255.0
    reduction = float(np.clip(cast_reduction, 0.0, 1.0))
    strength = float(np.clip(color_strength, 0.0, 2.5))
    if reduction > 0:
        luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        bright = (luminance >= np.percentile(luminance, 85)) & (luminance > 0.12)
        if np.any(bright):
            white = np.median(rgb[bright], axis=0)
            neutral = float(np.mean(white))
            gains = np.clip(neutral / np.maximum(white, 0.05), 0.8, 1.25)
            rgb = np.clip(rgb * (1.0 + reduction*(gains-1.0)), 0, 1)
    if strength != 1.0:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        if strength >= 1:
            boost = 1.0 + (strength-1.0)*(1.0-saturation)
            hsv[:, :, 1] = np.clip(saturation*boost, 0, 1)
        else:
            hsv[:, :, 1] *= strength
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(np.rint(np.clip(rgb, 0, 1)*255).astype(np.uint8))


def image_signature(image):
    """Match a displayed/downloaded drawing to its source without matching by filename."""
    rgb = rgb_on_white(image)
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def prepare_drawing_session(photo_img, graphite_strength, background_fade, pencil_texture):
    """Store paired images in Gradio's per-session state, not in a shared global cache."""
    drawing, training_sketch = photo_to_pencil_outputs(
        photo_img, graphite_strength, background_fade, pencil_texture
    )
    if drawing is None:
        return None, None, None
    pair = {
        'drawing_signature': image_signature(drawing),
        'source': rgb_on_white(photo_img).copy(),
        'training_sketch': training_sketch.copy(),
    }
    return drawing, training_sketch, pair


def send_drawing_to_step2(drawing):
    if drawing is None:
        raise gr.Error('Create a pencil drawing in Step 1 first.')
    # The user sees precisely the finished drawing; preprocessing stays internal.
    return drawing, INPUT_PENCIL, OUTPUT_MODEL, gr.Tabs(selected=1), None, None, ''


@torch.no_grad()
def run_inference(input_img, input_kind, output_mode, checkpoint_name, arch_name, device_str,
                  drawing_session=None, color_strength=1.0, cast_reduction=0.0):
    if input_img is None:
        raise gr.Error('Please provide a photo or pencil sketch first.')

    paired_drawing = (
        input_kind == INPUT_PENCIL and drawing_session is not None
        and image_signature(input_img) == drawing_session['drawing_signature']
    )
    config = {}
    if output_mode == OUTPUT_MODEL:
        device = resolve_device(device_str)
        model = load_model(checkpoint_name, arch_name, device)
        config = _loaded['config']
    resolution = int(config.get('train_res', TRAIN_RESOLUTION))
    if resolution < 16 or resolution % 16:
        raise gr.Error('Checkpoint training resolution must be divisible by 16.')
    style = config.get('sketch_style', 'opencv_dark')
    # A photo is never fed straight to the network: its tones and texture do not have the
    # pencil-sketch distribution that the one-channel checkpoint learned from.
    if paired_drawing:
        model_sketch = photo_input_for_checkpoint(drawing_session['source'], config)
    elif input_kind == INPUT_PHOTO:
        model_sketch = photo_input_for_checkpoint(input_img, config)
    else:
        model_sketch = rgb_on_white(input_img).convert('L')
    input_array = sketch_array_for_training(model_sketch, resolution)
    polarity_fixed = False
    if input_kind == INPUT_PENCIL and not paired_drawing and style == 'opencv_dark':
        input_array, polarity_fixed = match_training_sketch_polarity(input_array)

    # A sketch does not contain the original colour, texture, or high-frequency detail.
    # When the user supplied the source photo, returning it is the only honest way to make
    # a high-fidelity photo -> drawing -> photo round trip.  It also avoids silently
    # presenting a lossy 128px neural prediction as if it were an exact reconstruction.
    if output_mode == OUTPUT_SOURCE:
        if input_kind != INPUT_PHOTO and not paired_drawing:
            raise gr.Error(
                'There is no saved original photo matching this drawing. Create the drawing '
                'in Step 1 and send it to Step 2, or provide an original Photo. '
                'A separate pencil upload cannot recover the original photo through this option.'
            )
        return (
            drawing_session['source'].copy() if paired_drawing else rgb_on_white(input_img),
            None,
            'RESTORED ORIGINAL PHOTO · returned the supplied source photo at full resolution. '
            'This is not an AI prediction from the pencil drawing.',
        )

    x = sketch_array_to_tensor(input_array, device)

    start = time.time()
    pred = model(x)
    elapsed_ms = (time.time() - start) * 1000

    polarity_note = (
        ' · white-paper sketch automatically inverted to match the training data'
        if polarity_fixed else ''
    )
    return (
        enhance_predicted_colors(tensor_to_pil(pred), color_strength, cast_reduction),
        Image.fromarray(input_array),
        f'{elapsed_ms:.1f} ms on {device_str} · AI estimate is limited to '
        f'{resolution}×{resolution} because that is this checkpoint\'s training resolution'
        f'{" · using the paired training input saved in Step 1" if paired_drawing else ""}'
        f'{polarity_note}'
        f'{" · colour enhancement applied (post-processing)" if color_strength != 1.0 or cast_reduction != 0.0 else ""}',
        # Color controls affect display/export only, never the network or evaluation metrics.
    )


def photo_to_pencil_outputs(photo_img, graphite_strength=1.05, background_fade=0.6, pencil_texture=0.45):
    """Return a clean export drawing and the distinct sketch representation used by the model."""
    if photo_img is None:
        return None, None
    return make_clean_line_art(photo_img, graphite_strength, background_fade, pencil_texture), make_model_ready_pencil(photo_img)


def refresh_checkpoints():
    files = list_checkpoints()
    default = default_checkpoint(files)
    new_arch = guess_arch(default) if default else ARCH_NAMES[0]
    return gr.update(choices=files, value=default), new_arch


with gr.Blocks(title='Sketch-to-Photo — Local Test UI') as demo:
    drawing_session = gr.State(None)
    gr.Markdown(
        '# Sketch-to-Photo — Local Test UI\n'
        'Create a detailed pencil drawing, or test the trained sketch-to-photo checkpoint. '
        'The app keeps the model pipeline at its trained **128×128** resolution and makes '
        'the high-fidelity photo round-trip option explicit when the original photo is available.'
    )

    with gr.Row():
        checkpoint_dd = gr.Dropdown(
            choices=list_checkpoints(), value=default_checkpoint(),
            label='Checkpoint (.pt file)', scale=3,
        )
        refresh_btn = gr.Button('🔄 Refresh checkpoints', scale=1)

    arch_radio = gr.Radio(
        choices=ARCH_NAMES, value=guess_arch(default_checkpoint()),
        label='Architecture (auto-guessed from the filename — change if it guessed wrong)',
    )
    with gr.Row():
        device_dd = gr.Dropdown(
            choices=AVAILABLE_DEVICES, value=DEFAULT_DEVICE, label='Compute device',
            info='If results look broken on MPS/CUDA, try CPU — rules out a backend-specific bug.',
        )

    with gr.Tabs() as tabs:
        with gr.Tab('1. Photo → Pencil Sketch', id=0):
            gr.Markdown(
                '**Pencil drawing · renderer v5** combines soft graphite shading with '
                'paper grain and short pencil marks. The preview below is the same drawing '
                'you download.'
            )
            with gr.Row():
                photo_in = gr.Image(type='pil', label='Photo')
                clean_pencil_out = gr.Image(
                    type='pil', label='Pencil drawing · v5 (preview and download)',
                    interactive=False, format='png',
                )
            graphite_strength = gr.Slider(
                minimum=0.5, maximum=2.0, value=1.05, step=0.05,
                label='Pencil darkness',
                info='Controls continuous graphite shading. Start at 1.05.',
            )
            background_fade = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.6, step=0.05,
                label='Fade soft scenery',
                info='Lightens blurry areas. It can also lighten smooth subject areas; it does not remove the background.',
            )
            pencil_texture = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.45, step=0.05,
                label='Pencil and paper texture',
                info='Adds fine grain and short pencil marks to the shading. Zero gives smooth shading.',
            )
            convert_btn = gr.Button('Create pencil drawing', variant='primary')
            send_btn = gr.Button('Send this pencil drawing to Step 2 →')
            with gr.Accordion('Advanced: input prepared for the trained model', open=False):
                gr.Markdown(
                    'This dark **256×256 training representation** is sent to Step 2. '
                    'Its appearance matches the saved model\'s training data. '
                    'For the finished pencil drawing, use the main preview above.'
                )
                model_sketch_out = gr.Image(
                    type='pil', label='Technical model input (256×256)', interactive=False,
                )

        with gr.Tab('2. Sketch → Colorful Photo', id=1):
            gr.Markdown(
                'The input preview shows your finished pencil drawing. **AI estimate** runs '
                'the selected checkpoint at its trained resolution; the existing 128×128 '
                'models may be blurry and lose facial '
                'detail. **Restore original photo** returns the photo you supplied in Step 1 '
                'with its real colours and full detail. Restoration uses the saved source '
                'photo and must not be reported as model performance.'
            )
            input_kind = gr.Radio(
                choices=[INPUT_PENCIL, INPUT_PHOTO], value=INPUT_PENCIL,
                label='What are you providing?',
            )
            output_mode = gr.Radio(
                choices=[OUTPUT_MODEL, OUTPUT_SOURCE], value=OUTPUT_MODEL,
                label='Output mode',
                info='Restoration requires a matching drawing from Step 1 or a supplied original Photo.',
            )
            with gr.Row():
                sketch_in = gr.Image(type='pil', label='Pencil sketch or photo')
                photo_pred_out = gr.Image(type='pil', label='Colour result', interactive=False, format='png')
            with gr.Row():
                color_strength = gr.Slider(
                    minimum=0.0, maximum=2.5, value=1.5, step=0.05,
                    label='Colour intensity',
                    info='Strengthens colours already predicted. 1.0 keeps the raw model colours.',
                )
                cast_reduction = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.35, step=0.05,
                    label='Reduce cream/yellow cast',
                    info='Balances bright areas toward neutral. Use zero to retain warm lighting.',
                )
            gr.Markdown(
                'These controls adjust predicted colours; they cannot recover missing red, '
                'pink or green objects. Set intensity to **1.0** and cast reduction to **0.0** '
                'to inspect the raw model output. Original-photo restoration ignores these controls.'
            )
            with gr.Accordion('Advanced: network input after preprocessing', open=False):
                gr.Markdown(
                    'This uses the resolution and pencil style saved in the checkpoint. '
                    'It is different from the finished drawing shown above.'
                )
                model_input_out = gr.Image(type='pil', label='Technical network input (checkpoint preprocessing)', interactive=False)
            gen_time = gr.Textbox(label='Result method and details', interactive=False)
            generate_btn = gr.Button('Generate photo', variant='primary')

    # Wired up after both tabs exist, so Step 1's button can hand its result to Step 2.
    checkpoint_dd.change(fn=guess_arch, inputs=checkpoint_dd, outputs=arch_radio)
    refresh_btn.click(fn=refresh_checkpoints, outputs=[checkpoint_dd, arch_radio])

    convert_btn.click(
        fn=prepare_drawing_session,
        inputs=[photo_in, graphite_strength, background_fade, pencil_texture],
        outputs=[clean_pencil_out, model_sketch_out, drawing_session],
    )
    send_btn.click(
        fn=send_drawing_to_step2,
        inputs=clean_pencil_out,
        outputs=[sketch_in, input_kind, output_mode, tabs, photo_pred_out, model_input_out, gen_time],
    )
    generate_btn.click(
        fn=run_inference,
        inputs=[sketch_in, input_kind, output_mode, checkpoint_dd, arch_radio, device_dd,
                drawing_session, color_strength, cast_reduction],
        outputs=[photo_pred_out, model_input_out, gen_time],
    )


def main():
    # 7860–7959 is often occupied by an older Gradio process.  A dedicated default avoids
    # launching a stale UI by accident; set SKETCH2PHOTO_PORT to use another free port.
    port = int(os.environ.get('SKETCH2PHOTO_PORT', '8088'))
    demo.launch(server_name='127.0.0.1', server_port=port, show_error=True)


if __name__ == '__main__':
    main()
