"""Inference for the separate Colab dual-direction diffusion export.

Dependencies are loaded only when a model is opened. Legacy CNN inference is separate.
The same source is embedded in the standalone Colab notebook and its exported bundle.
"""

import json
from pathlib import Path

BASE_MODEL = 'timbrooks/instruct-pix2pix'
BASE_REVISION = '31519b5cb02a7fd89b906d88731cd4d6a7bbf88d'
PROMPTS = {
    'photo_to_pencil': (
        'Turn this image into a detailed hand-drawn graphite pencil drawing on white paper, '
        'with fine pencil strokes, gentle shading and clean highlights. Preserve the subject and composition.'
    ),
    'sketch_to_photo': (
        'Turn this pencil drawing into a realistic full-color photograph with natural colors, '
        'realistic lighting and detailed textures. Preserve the subject and composition.'
    ),
}


def rgb_on_white(image):
    from PIL import Image, ImageOps
    image = ImageOps.exif_transpose(image)
    rgba = image.convert('RGBA')
    white = Image.new('RGBA', rgba.size, 'white')
    return Image.alpha_composite(white, rgba).convert('RGB')


def fit_canvas(image, size=512):
    """Aspect-preserving resize and padding, also used for training pairs."""
    from PIL import Image
    image = rgb_on_white(image)
    scale = size / max(image.size)
    width, height = [max(1, min(size, round(v * scale))) for v in image.size]
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    left, top = (size - width) // 2, (size - height) // 2
    canvas = Image.new('RGB', (size, size), 'white')
    canvas.paste(resized, (left, top))
    return canvas, (left, top, left + width, top + height)


def open_pipeline(adapter_dir=None, device=None, base_model=BASE_MODEL,
                  revision=BASE_REVISION, offload=False):
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline, DDIMScheduler
    from peft import PeftModel
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu')
    dtype = torch.float16 if device == 'cuda' else torch.float32
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        base_model, revision=revision, torch_dtype=dtype, use_safetensors=True,
        variant='fp16' if dtype == torch.float16 else None)
    if adapter_dir is not None:
        adapter_dir = Path(adapter_dir)
        if not (adapter_dir / 'adapter_model.safetensors').is_file():
            raise FileNotFoundError(f'No trained adapter at {adapter_dir}')
        pipe.unet = PeftModel.from_pretrained(pipe.unet, str(adapter_dir), is_trainable=False)
        pipe.unet.to(dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)
    if offload and device == 'cuda':
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def generate(pipe, image, direction, seed=42, steps=30, text_guidance=7.0,
             image_guidance=1.5, size=512, description=''):
    import torch
    if direction not in PROMPTS:
        raise ValueError(f'Unknown direction: {direction}')
    if direction == 'sketch_to_photo':
        image = rgb_on_white(image).convert('L').convert('RGB')
    canvas, box = fit_canvas(image, size)
    # CPU generator works across CPU, CUDA, and MPS and is repeatable on each backend.
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    prompt = PROMPTS[direction]
    if description.strip():
        prompt += ' ' + description.strip()
    with torch.inference_mode():
        output = pipe(prompt=prompt, image=canvas, num_inference_steps=int(steps),
                      guidance_scale=float(text_guidance),
                      image_guidance_scale=float(image_guidance), generator=generator)
    if output.nsfw_content_detected is not None and any(output.nsfw_content_detected):
        raise RuntimeError('The base model filtered this output; it is not a successful prediction.')
    result = output.images[0].crop(box)
    if direction == 'photo_to_pencil':
        result = result.convert('L').convert('RGB')
    return result


class DualEditor:
    """Load one direction at a time to limit GPU memory."""
    def __init__(self, bundle_dir, device=None):
        self.root = Path(bundle_dir)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest.get('format') != 'sketch2photo-ip2p-lora-v1':
            raise ValueError('Expected a dual-direction notebook export, not a legacy .pt checkpoint.')
        self.device = device
        self.pipe = None
        self.direction = None

    def __call__(self, image, direction, seed=42, description=''):
        import gc
        import torch
        if image is None:
            raise ValueError('Upload an image first.')
        if direction not in PROMPTS:
            raise ValueError('Choose a supported direction.')
        if self.direction != direction:
            self.pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            self.pipe = open_pipeline(self.root / direction, self.device,
                                      self.manifest['base_model'], self.manifest['base_revision'],
                                      offload=True)
            self.direction = direction
        settings = self.manifest['inference']
        return generate(self.pipe, image, direction, seed=seed, description=description, **settings)


def launch_demo(bundle_dir, port=8094):
    import gradio as gr
    editor = DualEditor(bundle_dir)
    with gr.Blocks(title='Sketch2Photo — trained diffusion adapters') as demo:
        gr.Markdown('# Sketch2Photo\nGenerate a pencil drawing or a colour photograph.')
        direction = gr.Radio(list(PROMPTS), value='photo_to_pencil', label='Direction')
        with gr.Row():
            source = gr.Image(type='pil', label='Your image')
            result = gr.Image(type='pil', label='Generated result', interactive=False, format='png')
        description = gr.Textbox(label='Optional description or desired colours')
        seed = gr.Number(value=42, precision=0, label='Seed')
        gr.Button('Generate', variant='primary').click(
            editor, [source, direction, seed, description], result, concurrency_limit=1)
        gr.Markdown('Colours are plausible estimates. Fine facial details can change. '
                    'Results are generated from the uploaded image using your selected trained adapter.')
    demo.queue(default_concurrency_limit=1).launch(
        server_name='127.0.0.1', server_port=int(port), show_error=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--port', type=int, default=8094)
    args = parser.parse_args()
    launch_demo(args.bundle, args.port)
