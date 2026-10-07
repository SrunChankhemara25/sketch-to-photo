"""Final quality rescue to run after Section 9 of the master Colab notebook.

Why this is separate from the completed U-Net/PatchGAN run
-----------------------------------------------------------
The Section 9 evidence shows that the deterministic generator learned useful
general-scene structure, but its sketch-to-photo portrait output developed a
large magenta colour cast and identity artifacts.  More updates with the same
objective are therefore not a safe fix.  This add-on fine-tunes the existing
InstructPix2Pix model with LoRA for sketch-to-photo only.  It uses the exact
20K COCO + FS2K records already present in the live notebook and writes small,
resumable checkpoints to Drive.

Run in the SAME Colab runtime, immediately after Section 9:

    from google.colab import files
    files.upload()  # select final_diffusion_rescue_colab.py
    %run -i /content/final_diffusion_rescue_colab.py

Do not run the stale Section 10/11 export cells after this.  Review the final
visual gate printed by this script first.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    StableDiffusionInstructPix2PixPipeline,
)
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# User-adjustable settings.  The defaults process every record once.
# ---------------------------------------------------------------------------

RESCUE_NAME = "final_sketch_to_photo_diffusion_v1"
TRAIN_RESOLUTION = 256
VALIDATION_RESOLUTION = 256
GRADIENT_ACCUMULATION = 4
LORA_RANK = 16
LEARNING_RATE = 1.0e-5
WEIGHT_DECAY = 1.0e-4
WARMUP_UPDATES = 100
VALIDATE_EVERY = 500
SAVE_EVERY = 100
VALIDATION_COCO = 4
VALIDATION_FS2K = 4
VALIDATION_STEPS = 20
EARLY_STOP_PATIENCE = 5
MIN_DELTA = 0.002
CONDITIONING_DROPOUT = 0.05
SNR_GAMMA = 5.0


def require_live_notebook_state() -> None:
    required = (
        "SAVE_ROOT",
        "GENERAL_ROOT",
        "GENERAL_TRAIN_RECORDS",
        "GENERAL_COCO_SPLITS",
        "GENERAL_DATA_FINGERPRINT",
        "SPLITS",
        "get_general_pair",
        "tensor_image",
        "augment_white_paper_sketch",
        "PERCEPTUAL",
        "BASE_MODEL",
        "BASE_REVISION",
        "configure_adapter",
        "noise_prediction_loss",
        "generate",
        "PROMPTS",
        "GENERAL_SELECTED",
        "SELECTED",
        "SEED",
        "DEVICE",
        "DTYPE",
    )
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError(
            "Run the master notebook through Section 9 first. Missing live variables: "
            + ", ".join(missing)
        )
    if str(DEVICE) != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Choose a Colab GPU runtime before starting the rescue.")


require_live_notebook_state()


RUN_ROOT = Path(SAVE_ROOT) / RESCUE_NAME
ADAPTER_ROOT = RUN_ROOT / "sketch_to_photo"
# Keep the large public base model on Colab's temporary disk so a 15 GB Drive is
# not filled.  Only the small LoRA/resume state is persistent in Drive.  A fresh
# runtime may download the public base again, but never the 20K project dataset.
HF_CACHE = Path("/content/huggingface_cache/instruct_pix2pix")
RUN_ROOT.mkdir(parents=True, exist_ok=True)
ADAPTER_ROOT.mkdir(parents=True, exist_ok=True)
HF_CACHE.mkdir(parents=True, exist_ok=True)


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def atomic_torch(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Use the exact already-built record order.  It contains all 20,000 unique COCO
# images plus the existing FS2K replay records.  The training loop consumes every
# item once, in gradient-accumulation windows.
TRAIN_RECORDS = list(GENERAL_TRAIN_RECORDS)
TARGET_UPDATES = math.ceil(len(TRAIN_RECORDS) / GRADIENT_ACCUMULATION)
VAL_RECORDS = (
    list(GENERAL_COCO_SPLITS["val"][:VALIDATION_COCO])
    + list(SPLITS["val"][:VALIDATION_FS2K])
)

if len(GENERAL_COCO_SPLITS["train"]) != 20000:
    raise RuntimeError("Expected the completed persistent 20K COCO training split.")
if not VAL_RECORDS:
    raise RuntimeError("The fixed validation records are unavailable.")

for record in (
    GENERAL_COCO_SPLITS["train"][0],
    GENERAL_COCO_SPLITS["train"][-1],
    GENERAL_COCO_SPLITS["val"][0],
):
    if not Path(record["photo"]).is_file():
        raise FileNotFoundError(
            "The persistent COCO folder is not visible in this runtime: " + record["photo"]
        )


class RescueDataset(Dataset):
    def __init__(self, records, training: bool):
        self.records = records
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        photo, sketch = get_general_pair(record, TRAIN_RESOLUTION)
        rng = random.Random(int(SEED) + 1_000_003 * index + 91)
        if self.training and rng.random() < 0.5:
            from PIL import ImageOps

            photo, sketch = ImageOps.mirror(photo), ImageOps.mirror(sketch)
        if self.training and rng.random() < 0.35:
            sketch = augment_white_paper_sketch(sketch, rng)
        return {
            "source": tensor_image(sketch),
            "target": tensor_image(photo),
            "domain": record["domain"],
            "record_id": record["id"],
        }


def open_training_pipeline():
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        cache_dir=str(HF_CACHE),
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)
    pipe.to(DEVICE)
    configure_adapter(pipe, LORA_RANK)
    return pipe


def prompt_embeddings(pipe):
    prompt = PROMPTS["sketch_to_photo"] + " " + (
        "Use natural daylight and a clean background. Preserve the exact objects. "
        "Avoid pink or magenta skin, global colour casts, drawn outlines and double edges."
    )
    with torch.no_grad():
        ids = pipe.tokenizer(
            [prompt, ""],
            padding="max_length",
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(DEVICE)
        embeddings = pipe.text_encoder(ids)[0].detach()
    return prompt, embeddings


def metric_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(array.transpose(2, 0, 1))[None].to(DEVICE)


@torch.no_grad()
def validate(pipe, prompt: str, step: int):
    pipe.unet.eval()
    previous_vae_dtype = next(pipe.vae.parameters()).dtype
    pipe.vae.to(dtype=torch.float16)
    rows = []
    figure, axes = plt.subplots(
        len(VAL_RECORDS), 3, figsize=(12, 4 * len(VAL_RECORDS)), squeeze=False
    )
    try:
        for index, record in enumerate(VAL_RECORDS):
            photo, sketch = get_general_pair(record, VALIDATION_RESOLUTION)
            description = (
                "Use natural daylight and a clean background. Preserve the exact objects. "
                "Avoid pink or magenta skin, global colour casts, drawn outlines and double edges. "
                "Natural human skin tones, realistic eyes and facial texture."
                if record["domain"] == "fs2k_artist"
                else "Use natural daylight and a clean background. Preserve the exact objects. "
                "Avoid global colour casts, drawn outlines and double edges. "
                "Natural scene colours and realistic object textures."
            )
            prediction = generate(
                pipe,
                sketch,
                "sketch_to_photo",
                seed=int(SEED) + index,
                steps=VALIDATION_STEPS,
                text_guidance=7.0,
                image_guidance=2.0,
                size=VALIDATION_RESOLUTION,
                description=description,
            )
            value = float(PERCEPTUAL(metric_tensor(prediction), metric_tensor(photo)).mean())
            rows.append(
                {
                    "step": step,
                    "id": record["id"],
                    "domain": record["domain"],
                    "lpips": value,
                }
            )
            for axis, image, label in zip(
                axes[index],
                (sketch, prediction, photo),
                ("Input sketch", f"Diffusion step {step}", "Real target"),
            ):
                axis.imshow(image)
                axis.set_title(label)
                axis.axis("off")
    finally:
        pipe.vae.to(dtype=previous_vae_dtype)
        pipe.unet.train()

    coco = float(np.mean([row["lpips"] for row in rows if row["domain"] == "coco_synthetic"]))
    fs2k = float(np.mean([row["lpips"] for row in rows if row["domain"] == "fs2k_artist"]))
    # Faces receive more weight because Section 9 identified them as the severe
    # failure domain.  Both domain values are still reported separately.
    score = 0.35 * coco + 0.65 * fs2k
    summary = {
        "step": step,
        "weighted_val_lpips": score,
        "coco_lpips": coco,
        "fs2k_lpips": fs2k,
        "validation_images": len(rows),
    }
    figure.suptitle(
        "Sketch-to-photo diffusion rescue | "
        f"COCO {coco:.4f} | FS2K {fs2k:.4f} | weighted {score:.4f}"
    )
    figure.tight_layout()
    figure_path = ADAPTER_ROOT / f"visual_gate_step_{step:07d}.png"
    figure.savefig(figure_path, dpi=145)
    plt.show()
    plt.close(figure)
    atomic_json(ADAPTER_ROOT / f"validation_step_{step:07d}.json", rows)
    return summary, figure_path


def save_adapter(pipe, folder: Path, metadata: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    pipe.unet.save_pretrained(folder, safe_serialization=True)
    atomic_json(folder / "selection.json", metadata)


def training_loss(pipe, scheduler, embeddings, batch):
    return noise_prediction_loss(
        pipe,
        scheduler,
        embeddings,
        batch,
        device=DEVICE,
        dtype=DTYPE,
        conditioning_dropout=CONDITIONING_DROPOUT,
        snr_gamma=SNR_GAMMA,
        training=True,
    )


seed_everything(int(SEED) + 910)
gc.collect()
torch.cuda.empty_cache()

print("Opening the pretrained diffusion base in Colab's temporary cache.")
print("The public diffusion base may download once per fresh Colab runtime.")
print("The COCO/FS2K training dataset will NOT be downloaded again.")
print(
    "Budget:", TARGET_UPDATES, "optimizer updates from", len(TRAIN_RECORDS),
    "existing training records; Drive resume is saved every", SAVE_EVERY, "updates."
)
pipe = open_training_pipeline()
PROMPT, EMBEDDINGS = prompt_embeddings(pipe)
NOISE_SCHEDULER = DDPMScheduler.from_pretrained(
    BASE_MODEL,
    revision=BASE_REVISION,
    subfolder="scheduler",
    cache_dir=str(HF_CACHE),
)
PARAMETERS = [parameter for parameter in pipe.unet.parameters() if parameter.requires_grad]
OPTIMIZER = torch.optim.AdamW(
    PARAMETERS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)


def lr_multiplier(step: int) -> float:
    warm = min(1.0, (step + 1) / max(1, WARMUP_UPDATES))
    progress = min(1.0, step / max(1, TARGET_UPDATES))
    cosine = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return warm * cosine


LR_SCHEDULER = torch.optim.lr_scheduler.LambdaLR(OPTIMIZER, lr_multiplier)
SCALER = torch.amp.GradScaler("cuda", init_scale=1024.0)

RECIPE = {
    "name": RESCUE_NAME,
    "algorithm": "instruct_pix2pix_lora_sketch_to_photo",
    "base_model": BASE_MODEL,
    "base_revision": BASE_REVISION,
    "prompt": PROMPT,
    "general_data_fingerprint": GENERAL_DATA_FINGERPRINT,
    "unique_coco_train_images": len(GENERAL_COCO_SPLITS["train"]),
    "unique_fs2k_train_pairs": len(SPLITS["train"]),
    "training_records": len(TRAIN_RECORDS),
    "target_updates": TARGET_UPDATES,
    "train_resolution": TRAIN_RESOLUTION,
    "gradient_accumulation": GRADIENT_ACCUMULATION,
    "lora_rank": LORA_RANK,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "validation": {
        "coco": VALIDATION_COCO,
        "fs2k": VALIDATION_FS2K,
        "steps": VALIDATION_STEPS,
        "selection": "0.35*COCO_LPIPS + 0.65*FS2K_LPIPS",
    },
}
SIGNATURE = hashlib.sha256(json.dumps(RECIPE, sort_keys=True).encode()).hexdigest()
recipe_path = ADAPTER_ROOT / "recipe.json"
if recipe_path.exists():
    if json.loads(recipe_path.read_text()) != RECIPE:
        raise RuntimeError("Rescue configuration changed. Use a new RESCUE_NAME.")
else:
    atomic_json(recipe_path, RECIPE)

PROGRESS = {
    "step": 0,
    "sample_offset": 0,
    "best_score": float("inf"),
    "best_step": 0,
    "stale": 0,
    "seconds": 0.0,
}
HISTORY = []
LAST_STATE = ADAPTER_ROOT / "last_training_state.pt"

if LAST_STATE.exists():
    state = torch.load(LAST_STATE, map_location="cpu", weights_only=False)
    if state["signature"] != SIGNATURE:
        raise RuntimeError("Saved rescue state does not match this recipe.")
    set_peft_model_state_dict(pipe.unet, state["adapter"])
    OPTIMIZER.load_state_dict(state["optimizer"])
    LR_SCHEDULER.load_state_dict(state["lr_scheduler"])
    SCALER.load_state_dict(state["scaler"])
    PROGRESS = state["progress"]
    HISTORY = state["history"]
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    print("Resuming diffusion rescue at update", PROGRESS["step"])
    del state
else:
    baseline, baseline_figure = validate(pipe, PROMPT, 0)
    PROGRESS["best_score"] = baseline["weighted_val_lpips"]
    HISTORY.append({**baseline, "label": "pretrained_baseline"})
    save_adapter(pipe, ADAPTER_ROOT / "best", baseline)
    atomic_json(ADAPTER_ROOT / "history.json", HISTORY)
    print("Pretrained baseline:", baseline)
    print("Baseline visual gate:", baseline_figure)


def save_resume(started: float, elapsed_before: float) -> None:
    PROGRESS["seconds"] = elapsed_before + time.perf_counter() - started
    atomic_torch(
        LAST_STATE,
        {
            "signature": SIGNATURE,
            "adapter": {
                key: value.detach().cpu().clone()
                for key, value in get_peft_model_state_dict(pipe.unet).items()
            },
            "optimizer": OPTIMIZER.state_dict(),
            "lr_scheduler": LR_SCHEDULER.state_dict(),
            "scaler": SCALER.state_dict(),
            "progress": dict(PROGRESS),
            "history": HISTORY,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        },
    )
    atomic_json(ADAPTER_ROOT / "history.json", HISTORY)
    atomic_json(ADAPTER_ROOT / "progress.json", PROGRESS)


DATASET = RescueDataset(TRAIN_RECORDS, training=True)
remaining_indices = list(range(PROGRESS["sample_offset"], len(TRAIN_RECORDS)))
LOADER = DataLoader(
    Subset(DATASET, remaining_indices),
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)
ITERATOR = iter(LOADER)
started = time.perf_counter()
elapsed_before = PROGRESS["seconds"]
bar = tqdm(total=TARGET_UPDATES, initial=PROGRESS["step"], desc="Diffusion rescue")
pipe.unet.train()

while (
    PROGRESS["sample_offset"] < len(TRAIN_RECORDS)
    and PROGRESS["step"] < TARGET_UPDATES
    and PROGRESS["stale"] < EARLY_STOP_PATIENCE
):
    count = min(
        GRADIENT_ACCUMULATION,
        len(TRAIN_RECORDS) - PROGRESS["sample_offset"],
    )
    window = [next(ITERATOR) for _ in range(count)]
    for retry in range(6):
        OPTIMIZER.zero_grad(set_to_none=True)
        mean_loss = 0.0
        for batch in window:
            loss = training_loss(pipe, NOISE_SCHEDULER, EMBEDDINGS, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite diffusion loss; the last Drive checkpoint is safe."
                )
            mean_loss += float(loss.detach())
            SCALER.scale(loss / count).backward()
        SCALER.unscale_(OPTIMIZER)
        torch.nn.utils.clip_grad_norm_(PARAMETERS, 1.0, error_if_nonfinite=True)
        old_scale = SCALER.get_scale()
        SCALER.step(OPTIMIZER)
        SCALER.update()
        if SCALER.get_scale() >= old_scale:
            break
        print(
            "Retrying this update after FP16 overflow protection reduced the scale to",
            SCALER.get_scale(),
        )
    else:
        raise FloatingPointError(
            "Repeated FP16 overflow; the last Drive checkpoint remains recoverable."
        )
    LR_SCHEDULER.step()

    PROGRESS["step"] += 1
    PROGRESS["sample_offset"] += count
    step = PROGRESS["step"]
    mean_loss /= count
    HISTORY.append(
        {
            "step": step,
            "sample_offset": PROGRESS["sample_offset"],
            "train_noise_loss": mean_loss,
            "lr": LR_SCHEDULER.get_last_lr()[0],
        }
    )
    bar.update(1)
    bar.set_postfix(loss=round(mean_loss, 4), samples=PROGRESS["sample_offset"])

    should_validate = step % VALIDATE_EVERY == 0 or step == TARGET_UPDATES
    if should_validate:
        summary, figure_path = validate(pipe, PROMPT, step)
        HISTORY.append(summary)
        improved = summary["weighted_val_lpips"] < PROGRESS["best_score"] - MIN_DELTA
        if improved:
            PROGRESS["best_score"] = summary["weighted_val_lpips"]
            PROGRESS["best_step"] = step
            PROGRESS["stale"] = 0
            save_adapter(pipe, ADAPTER_ROOT / "best", summary)
        else:
            PROGRESS["stale"] += 1
        save_adapter(pipe, ADAPTER_ROOT / f"adapter_step_{step:07d}", summary)
        print("Validation:", summary)
        print("Visual gate:", figure_path)
        pipe.unet.train()

    if step % SAVE_EVERY == 0 or should_validate:
        save_resume(started, elapsed_before)
    del window

bar.close()
save_resume(started, elapsed_before)

FINAL_SELECTION = {
    "format": "sketch2photo-final-quality-rescue-v1",
    "photo_to_pencil": {
        "general_scene": GENERAL_SELECTED["photo_to_pencil"]["unet"],
        "portrait": SELECTED["photo_to_pencil"]["unet"],
        "strategy": "route to the reviewed specialist; no risky extra GAN updates",
    },
    "sketch_to_photo": {
        "diffusion_adapter": str(ADAPTER_ROOT / "best"),
        "best_step": PROGRESS["best_step"],
        "weighted_val_lpips": PROGRESS["best_score"],
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "fallback_general_unet": GENERAL_SELECTED["sketch_to_photo"]["unet"],
        "fallback_portrait_unet": SELECTED["sketch_to_photo"]["unet"],
    },
    "status": (
        "needs_visual_review"
        if PROGRESS["best_step"] > 0
        else "fine_tuning_did_not_beat_pretrained_baseline"
    ),
}
atomic_json(RUN_ROOT / "final_selection.json", FINAL_SELECTION)

print("\nRESCUE TRAINING COMPLETE")
print("Best fine-tuned step:", PROGRESS["best_step"])
print("Best weighted validation LPIPS:", PROGRESS["best_score"])
print("Selection file:", RUN_ROOT / "final_selection.json")
if PROGRESS["best_step"] == 0:
    print(
        "WARNING: fine-tuning did not beat the pretrained baseline. Do not export yet; "
        "send the step-0 and latest visual gates for review."
    )
else:
    print(
        "Send the best/latest visual-gate PNGs for final review. Do not run the old "
        "Section 10/11 export cells."
    )
