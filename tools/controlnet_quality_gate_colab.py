"""Fast, non-training quality gate for a line-art-specific ControlNet.

Run after Section 9 with:

    %run -i /content/controlnet_quality_gate_colab.py

This tests whether a line-art-conditioned pretrained model is a suitable
quality engine before spending more Colab hours.  It does not alter or replace
any trained project checkpoint and must be disclosed as a pretrained demo
enhancement if later used without fine-tuning.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)


REQUIRED = (
    "SAVE_ROOT",
    "GENERAL_COCO_SPLITS",
    "SPLITS",
    "get_general_pair",
    "PERCEPTUAL",
    "DEVICE",
    "SEED",
)
missing = [name for name in REQUIRED if name not in globals()]
if missing:
    raise RuntimeError(
        "Run the master notebook through Section 9 first. Missing: " + ", ".join(missing)
    )
if not torch.cuda.is_available():
    raise RuntimeError("Choose a Colab GPU runtime.")


BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_lineart"
OUTPUT_ROOT = Path(SAVE_ROOT) / "controlnet_lineart_quality_gate_v1"
CACHE_ROOT = Path("/content/huggingface_cache/controlnet_lineart")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

VALIDATION_RECORDS = list(GENERAL_COCO_SPLITS["val"][:4]) + list(SPLITS["val"][:4])
RESOLUTION = 512
INFERENCE_STEPS = 30

POSITIVE_GENERAL = (
    "a sharp realistic full-colour documentary photograph of this exact scene, "
    "natural believable object colours, realistic daylight, detailed photographic "
    "textures, preserve every person, object, pose, proportion and composition"
)
POSITIVE_PORTRAIT = (
    "a sharp realistic full-colour professional portrait photograph of this exact "
    "person, natural skin tone, realistic eyes, detailed hair and skin texture, "
    "natural soft lighting, preserve identity, facial geometry, pose and proportions"
)
NEGATIVE = (
    "drawing, sketch, illustration, cartoon, anime, painting, monochrome, sepia, "
    "pink skin, magenta skin, colour cast, plastic skin, deformed face, extra people, "
    "duplicate, double edges, blurry, low detail, text, watermark"
)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def metric_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(array.transpose(2, 0, 1))[None].to(DEVICE)


gc.collect()
torch.cuda.empty_cache()
print("Downloading/loading the public line-art ControlNet and SD 1.5 base.")
print("This is a quality test only; it does not download COCO or train a model.")

controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=torch.float16,
    use_safetensors=True,
    cache_dir=str(CACHE_ROOT),
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True,
    cache_dir=str(CACHE_ROOT),
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=False)

rows = []
figure, axes = plt.subplots(
    len(VALIDATION_RECORDS), 3, figsize=(12, 4 * len(VALIDATION_RECORDS)), squeeze=False
)

for index, record in enumerate(VALIDATION_RECORDS):
    photo, sketch = get_general_pair(record, RESOLUTION)
    prompt = POSITIVE_PORTRAIT if record["domain"] == "fs2k_artist" else POSITIVE_GENERAL
    generator = torch.Generator(device="cpu").manual_seed(int(SEED) + index)
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE,
        image=sketch.convert("RGB"),
        num_inference_steps=INFERENCE_STEPS,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.15,
        guess_mode=True,
        generator=generator,
        height=RESOLUTION,
        width=RESOLUTION,
    )
    if result.nsfw_content_detected is not None and any(result.nsfw_content_detected):
        raise RuntimeError(
            "The base safety checker filtered a normal validation image. "
            "No black image is accepted as a valid result."
        )
    prediction = result.images[0].convert("RGB")
    lpips_value = float(PERCEPTUAL(metric_tensor(prediction), metric_tensor(photo)).mean())
    rows.append(
        {
            "id": record["id"],
            "domain": record["domain"],
            "lpips": lpips_value,
            "seed": int(SEED) + index,
        }
    )
    for axis, image, label in zip(
        axes[index],
        (sketch, prediction, photo),
        ("Input sketch", "ControlNet output", "Real target"),
    ):
        axis.imshow(image)
        axis.set_title(label)
        axis.axis("off")

coco = float(np.mean([row["lpips"] for row in rows if row["domain"] == "coco_synthetic"]))
fs2k = float(np.mean([row["lpips"] for row in rows if row["domain"] == "fs2k_artist"]))
weighted = 0.35 * coco + 0.65 * fs2k
summary = {
    "status": "visual_review_required",
    "approach": "pretrained_controlnet_lineart_quality_gate",
    "counts_as_student_trained_approach": False,
    "base_model": BASE_MODEL,
    "controlnet_model": CONTROLNET_MODEL,
    "coco_lpips": coco,
    "fs2k_lpips": fs2k,
    "weighted_lpips": weighted,
    "rows": rows,
}

figure.suptitle(
    f"Line-art ControlNet quality gate | COCO {coco:.4f} | "
    f"FS2K {fs2k:.4f} | weighted {weighted:.4f}"
)
figure.tight_layout()
output_path = OUTPUT_ROOT / "controlnet_visual_gate.png"
figure.savefig(output_path, dpi=145)
plt.show()
plt.close(figure)
atomic_json(OUTPUT_ROOT / "quality_gate.json", summary)

print("ControlNet quality gate:", output_path)
print("This was inference only. Send the figure for review before any fine-tuning or export.")

