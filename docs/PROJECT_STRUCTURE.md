# Project structure

```text
sketch2photo_test_ui/
├── app.py                  # deployable Streamlit web application
├── src/
│   ├── __init__.py
│   ├── application.py      # original-model renderer/inference compatibility backend
│   ├── diffusion_backend.py # inference for exported diffusion adapters
│   ├── study_backend.py    # both directions, three-approach export inference
│   ├── approaches/        # new U-Net, residual GAN and diffusion training core
│   └── models.py           # Autoencoder, U-Net, ResNet encoder-decoder
├── notebooks/
│   └── sketch2photo_master_training.ipynb # ONLY training notebook; both tasks
├── checkpoints/            # existing trained weights
├── results/                # existing metrics, histories, figures, samples
├── docs/
│   ├── PROJECT_STRUCTURE.md
│   ├── TRAINING_GUIDE.md
│   └── DUAL_TRAINING.md
├── requirements.txt
├── requirements-diffusion.txt # pinned Python 3.11/3.12 new-model runtime
├── README.md
└── .gitignore
```

Sketch2Photo separates application source code, Colab training notebooks,
documentation, trained checkpoints, and evaluation results into dedicated folders.

## Where to work

- Run `python app.py` or `streamlit run app.py` from the project root. Use the sidebar
  to choose original checkpoints or a new trained bundle. Without a new export, the
  original model screen is selected automatically.
- Edit `src/application.py` for interface, preprocessing, or drawing changes.
- Keep the layers in `src/models.py` compatible with saved checkpoints.
- Upload the standalone notebook to Colab for training; it does not import local source files.
- The master embeds the approach/inference modules, so Colab needs only the notebook.
  These source modules are actual architectures and shared runtime code, not local training jobs.
- Download new best checkpoints into `checkpoints/` and keep their new results labeled separately.

Checkpoint paths are relative to this project, not the terminal's current directory.
Folder organization does not change model quality or train a new model.

## GitHub submission

Track source, notebook, documentation, and result figures. Do not normally commit
the virtual environment, Python caches, local datasets, or large model weights.
Existing weights total about 859 MB; provide a real external download link or use
Git LFS if distributing them. No upload or repository has been created here.
Before submission, include the actual checkpoint download link, dataset source,
experiment settings, and honest limitations in the README.
