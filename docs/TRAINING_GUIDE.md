# Colab training guide

There is one training notebook:

`notebooks/sketch2photo_master_training.ipynb`

It trains photo → pencil and pencil → colour photo, compares all three approaches,
performs tuning and evaluation, and exports the bundle used by `app.py`.

Follow [the complete master training instructions](DUAL_TRAINING.md). Do not use an
older notebook copied from another folder because its preprocessing and export format
will not match the current application.
