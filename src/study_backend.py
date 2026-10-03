"""Shared inference for the three-approach Colab study export."""
import gc
import json
from pathlib import Path
from .diffusion_backend import PROMPTS, fit_canvas, open_pipeline, generate


class StudyEditor:
    def __init__(self, bundle_dir, device=None):
        import torch
        self.root = Path(bundle_dir)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest.get('format') != 'sketch2photo-study-v2':
            raise ValueError('Expected the master notebook study export.')
        self.device = device or ('cuda' if torch.cuda.is_available() else (
            'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))
        self.model, self.key = None, None

    def __call__(self, image, direction, approach='validation_best', seed=42, description=''):
        import numpy as np
        import torch
        from PIL import Image
        from .approaches import make_generator
        if image is None:
            raise ValueError('Upload a photo or pencil sketch first.')
        if direction not in PROMPTS:
            raise ValueError('Unknown direction')
        if approach == 'validation_best':
            approach = self.manifest['validation_best'][direction]
        key = (direction, approach)
        info = self.manifest['models'][direction][approach]
        if key != self.key:
            self.model = None; self.key = None; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if approach == 'diffusion':
                self.model = open_pipeline(self.root / info['path'], self.device,
                    self.manifest['base_model'], self.manifest['base_revision'], offload=True)
            else:
                state = torch.load(self.root / info['path'], map_location='cpu', weights_only=True)
                assert state['approach'] == approach and state['direction'] == direction
                self.model = make_generator(approach, state['width'])
                self.model.load_state_dict(state['generator'], strict=True)
                self.model.to(self.device).eval()
            self.key = key
        settings = self.manifest['inference']
        if approach == 'diffusion':
            return generate(self.model, image, direction, seed=seed, description=description, **settings)
        canvas, box = fit_canvas(image, settings['size'])
        if direction == 'sketch_to_photo':
            canvas = canvas.convert('L').convert('RGB')
        x = torch.from_numpy(np.asarray(canvas, dtype=np.float32).copy().transpose(2, 0, 1) / 127.5 - 1)
        with torch.inference_mode():
            y = self.model(x[None].to(self.device))[0].float().cpu().clamp(-1, 1)
        image = Image.fromarray(((y.permute(1, 2, 0).numpy()+1)*127.5).round().astype('uint8')).crop(box)
        return image.convert('L').convert('RGB') if direction == 'photo_to_pencil' else image


def launch_demo(bundle_dir, port=8088):
    import gradio as gr
    editor = StudyEditor(bundle_dir)
    with gr.Blocks(title='Sketch2Photo — trained model comparison') as demo:
        gr.Markdown('# Sketch2Photo\nBoth directions use trained models from your Colab export. '
                    'Original colours cannot be uniquely recovered from a sketch.')
        gr.Markdown('Export mode: **' + editor.manifest['mode'] + '**. Selected defaults: `' +
                    json.dumps(editor.manifest['validation_best']) + '`. Best means lowest validation LPIPS, not perfect.')
        direction = gr.Radio(list(PROMPTS), value='photo_to_pencil', label='Direction')
        approach = gr.Dropdown(['validation_best', 'unet', 'gan', 'diffusion'], value='validation_best', label='Trained approach')
        with gr.Row():
            source = gr.Image(type='pil', label='Input photo / white-paper pencil sketch')
            result = gr.Image(type='pil', format='png', interactive=False, label='Actual generated output — preview and download')
        description = gr.Textbox(label='Optional description / desired colours (diffusion only)')
        seed = gr.Number(value=42, precision=0, label='Diffusion seed')
        gr.Button('Generate', variant='primary').click(editor, [source, direction, approach, seed, description], result, concurrency_limit=1)
        def send_to_photo(image):
            return image, 'sketch_to_photo'
        gr.Button('Use this output as sketch input').click(send_to_photo, result, [source, direction])
        gr.Markdown('No original-photo retrieval, saturation trick, or separate dark sketch is used. '
                    'Generation is at the training resolution; upscaling is not extra learned detail.')
    demo.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1', server_port=int(port), show_error=True)
