# Checkpoint layout

Only deployment checkpoints live here. Training notebooks and experiment history
are kept elsewhere.

```text
checkpoints/
├── photo_to_pencil/
│   ├── manifest.json
│   └── models/{unet,gan}/best.pt
└── sketch_to_photo/
    ├── manifest.json
    ├── models/{general,portrait}/best.pt
    └── reports/
```

The manifests are the source of truth for model labels, architectures, training
resolution, validation metrics, relative weight paths, file sizes and SHA-256
checksums. Do not rename or replace a weight file without updating its manifest.

Large `.pt` files are managed by Git LFS through the repository's
`.gitattributes`. Confirm that Git LFS is installed before cloning or pushing, and
keep a separate backup in controlled model storage before replacing a checkpoint.
