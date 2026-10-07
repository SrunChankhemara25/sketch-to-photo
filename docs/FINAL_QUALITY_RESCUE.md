# Final quality rescue after Section 9

The completed Section 9 run should not be continued with more U-Net/PatchGAN
updates. Its evidence shows two different outcomes:

- **Photo to pencil:** retain the general-scene best checkpoint and the earlier
  face-specialist checkpoint. Routing between these checkpoints is safer than
  continuing the adversarial run.
- **Sketch to photo:** the general checkpoint improves scene layout but creates
  severe portrait colour/identity artifacts. Use the diffusion-LoRA rescue below.

## Run once after Section 9

Keep the current Colab runtime connected. Do not rerun or delete the completed
general training folders.

1. Upload `tools/final_diffusion_rescue_colab.py` to Colab.
2. In a new code cell run:

   ```python
   %run -i /content/final_diffusion_rescue_colab.py
   ```

The `-i` is required because the rescue intentionally reuses the verified
Section 9 records and functions already in memory.

The script does **not** redownload COCO or repeat the GAN run. A fresh Colab
runtime may need to download the public InstructPix2Pix base model. The base is
kept on temporary Colab disk; only small LoRA and resume files are written to:

```text
MyDrive/Sketch2Photo_training/final_sketch_to_photo_diffusion_v1/
```

Training uses all records in the completed 20K+FS2K training order as a one-pass
budget, with gradient accumulation. It saves a resumable state every 100 updates
and a visual validation gate every 500 updates. If Colab disconnects, restore the
same Section 9 notebook state, upload the script, and run the same command; it
loads `last_training_state.pt` and continues.

## Stop condition

At completion, do not run the old Section 10/11 export cells. Review these files:

```text
.../sketch_to_photo/visual_gate_step_*.png
.../sketch_to_photo/history.json
.../final_selection.json
```

Accept a fine-tuned checkpoint only when both the diverse COCO rows and held-out
FS2K portrait rows look natural. LPIPS selects a candidate; visual inspection is
the final gate for colour casts, identity drift, blur, duplicated edges, and
anatomy errors.

