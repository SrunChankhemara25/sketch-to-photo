"""Quality gate for a constrained U-Net -> ControlNet photo refiner.

Run after Section 9 in the same Colab runtime:

    %run -i /content/constrained_photo_refiner_gate_colab.py

Unlike the rejected text-to-image ControlNet test, this pipeline starts from
the trained U-Net reconstruction and uses low denoising strength.  The sketch
is supplied separately as line-art control.  This is an inference-only gate;
it cannot overwrite any trained checkpoint.
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
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
)


REQUIRED = (
    "SAVE_ROOT",
    "GENERAL_COCO_SPLITS",
    "SPLITS",
    "get_general_pair",
    "load_unet_checkpoint",
    "predict_unet",
    "GENERAL_SELECTED",
    "SELECTED",
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
CACHE_ROOT = Path("/content/huggingface_cache/controlnet_lineart")
OUTPUT_ROOT = Path(SAVE_ROOT) / "constrained_photo_refiner_gate_v1"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

RESOLUTION = 512
INFERENCE_STEPS = 40
GENERAL_STRENGTH = 0.28
PORTRAIT_STRENGTH = 0.22
VALIDATION_RECORDS = list(GENERAL_COCO_SPLITS["val"][:4]) + list(SPLITS["val"][:4])

GENERAL_PROMPT = (
    "a realistic unedited documentary photograph of the same scene, natural colours, "
    "real camera texture and lighting, preserve the exact people, objects, layout, pose "
    "and proportions"
)
PORTRAIT_PROMPT = (
    "a realistic unedited portrait photograph of the same person, natural skin colour, "
    "realistic eyes, hair and skin texture, preserve exact identity, facial geometry, "
    "expression, pose and proportions"
)
NEGATIVE_PROMPT = (
    "drawing, sketch, cartoon, anime, painting, digital art, oversaturated, blue skin, "
    "pink skin, magenta skin, colour cast, plastic skin, changed identity, different "
    "person, deformed face, extra people, duplicate, double edges, blurry, text, watermark, "
    "nsfw, nude, nudity, naked, sexual content"
)
MAX_SAFETY_RETRIES = 4


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def metric_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(array.transpose(2, 0, 1))[None].to(DEVICE)


print("Stage 1/2: building faithful initial reconstructions with your trained U-Nets.")
general_model, general_state = load_unet_checkpoint(
    Path(GENERAL_SELECTED["sketch_to_photo"]["unet"]["path"])
)
portrait_model, portrait_state = load_unet_checkpoint(
    Path(SELECTED["sketch_to_photo"]["unet"]["path"])
)

if int(general_state["step"]) != 18000:
    raise RuntimeError("Expected the reviewed general sketch-to-photo checkpoint at step 18000.")

prepared = []
for record in VALIDATION_RECORDS:
    photo, sketch = get_general_pair(record, RESOLUTION)
    is_portrait = record["domain"] == "fs2k_artist"
    model = portrait_model if is_portrait else general_model
    initial = predict_unet(model, sketch, "sketch_to_photo")
    prepared.append(
        {
            "record": record,
            "photo": photo,
            "sketch": sketch,
            "initial": initial,
            "is_portrait": is_portrait,
        }
    )

del general_model, portrait_model
gc.collect()
torch.cuda.empty_cache()

print("Stage 2/2: low-strength photographic refinement with line-art control.")
print("The already-downloaded ControlNet/SD1.5 cache will be reused when available.")
controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=torch.float16,
    use_safetensors=True,
    cache_dir=str(CACHE_ROOT),
)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
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
    len(prepared), 4, figsize=(16, 4 * len(prepared)), squeeze=False
)

for index, item in enumerate(prepared):
    prompt = PORTRAIT_PROMPT if item["is_portrait"] else GENERAL_PROMPT
    strength = PORTRAIT_STRENGTH if item["is_portrait"] else GENERAL_STRENGTH
    refined = None
    safety_retries = 0
    for attempt in range(MAX_SAFETY_RETRIES):
        generator = torch.Generator(device="cpu").manual_seed(
            int(SEED) + 900 + index + 10000 * attempt
        )
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=item["initial"].convert("RGB"),
            control_image=item["sketch"].convert("RGB"),
            strength=strength,
            num_inference_steps=INFERENCE_STEPS,
            guidance_scale=5.0,
            controlnet_conditioning_scale=1.25,
            guess_mode=False,
            generator=generator,
        )
        flagged = (
            result.nsfw_content_detected is not None
            and any(result.nsfw_content_detected)
        )
        if not flagged:
            refined = result.images[0].convert("RGB")
            break
        safety_retries += 1
        print(
            "Safety filter retry",
            safety_retries,
            "of",
            MAX_SAFETY_RETRIES,
            "for",
            item["record"]["id"],
        )

    safety_filtered = refined is None
    if safety_filtered:
        # Keep the gate complete and visibly reject this row. Never substitute the
        # black safety-checker image and never disable the checker.
        refined = item["initial"].copy()
        print("All safety retries were filtered; this row is marked NOT ACCEPTED.")

    initial_lpips = float(
        PERCEPTUAL(metric_tensor(item["initial"]), metric_tensor(item["photo"])).mean()
    )
    refined_lpips = float(
        PERCEPTUAL(metric_tensor(refined), metric_tensor(item["photo"])).mean()
    )
    rows.append(
        {
            "id": item["record"]["id"],
            "domain": item["record"]["domain"],
            "strength": strength,
            "initial_lpips": initial_lpips,
            "refined_lpips": refined_lpips,
            "improved_lpips": refined_lpips < initial_lpips,
            "safety_filtered": safety_filtered,
            "safety_retries": safety_retries,
        }
    )
    refined_label = (
        "Safety-filtered: NOT ACCEPTED"
        if safety_filtered
        else "Constrained refinement"
    )
    for axis, image, label in zip(
        axes[index],
        (item["sketch"], item["initial"], refined, item["photo"]),
        ("Input sketch", "Trained U-Net", refined_label, "Real target"),
    ):
        axis.imshow(image)
        axis.set_title(label)
        axis.axis("off")

initial_coco = float(np.mean([r["initial_lpips"] for r in rows if r["domain"] == "coco_synthetic"]))
refined_coco = float(np.mean([r["refined_lpips"] for r in rows if r["domain"] == "coco_synthetic"]))
initial_fs2k = float(np.mean([r["initial_lpips"] for r in rows if r["domain"] == "fs2k_artist"]))
refined_fs2k = float(np.mean([r["refined_lpips"] for r in rows if r["domain"] == "fs2k_artist"]))
summary = {
    "status": "visual_review_required",
    "method": "trained_unet_then_low_strength_pretrained_controlnet_img2img",
    "counts_as_new_student_trained_approach": False,
    "base_model": BASE_MODEL,
    "controlnet_model": CONTROLNET_MODEL,
    "general_strength": GENERAL_STRENGTH,
    "portrait_strength": PORTRAIT_STRENGTH,
    "initial_coco_lpips": initial_coco,
    "refined_coco_lpips": refined_coco,
    "initial_fs2k_lpips": initial_fs2k,
    "refined_fs2k_lpips": refined_fs2k,
    "rows": rows,
}

figure.suptitle(
    "Constrained photo refiner | "
    f"COCO {initial_coco:.4f}->{refined_coco:.4f} | "
    f"FS2K {initial_fs2k:.4f}->{refined_fs2k:.4f}"
)
figure.tight_layout()
figure_path = OUTPUT_ROOT / "constrained_refiner_visual_gate.png"
figure.savefig(figure_path, dpi=145)
plt.show()
plt.close(figure)
atomic_json(OUTPUT_ROOT / "quality_gate.json", summary)

print("Visual gate:", figure_path)
print("No trained checkpoint was changed. Send this figure for the final accept/reject decision.")
