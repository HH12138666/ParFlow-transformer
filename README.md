# ParFlow-PredFormer

A Transformer-based surrogate model for ParFlow-CLM groundwater simulations. The model predicts future pressure-head fields from recent pressure states, prescribed EvapTrans inputs, and optional static hydrogeologic attributes. Water table depth (WTD) is diagnosed from the predicted pressure-head fields for evaluation.

## Main Features

- PredFormer-based sequence-to-sequence pressure-head prediction.
- Implicit residual prediction relative to the last input pressure state.
- Support for dynamic EvapTrans inputs and static attributes.
- Long-horizon autoregressive rollout for groundwater pressure and WTD prediction.
- Optional extra-sample training through precomputed manifest files.

## Repository Structure

```text
configs/              Model and experiment configuration files
openstl/              Dataset, model, method, training, and evaluation code
model_deployment/     Inference code for trained checkpoints
tools/                Training entry and data-preparation utilities
scripts/parflow/      Example shell scripts for local training and inference
data/                 Empty data directory template
stats/                Local normalization statistics, not tracked by git
work_dirs/            Local training outputs, not tracked by git
inference_data/       Local inference outputs, not tracked by git
```

Analysis outputs, paper figures, checkpoints, logs, and large ParFlow files are not included in this repository.

## Data Layout

The repository only keeps the expected directory structure. Large ParFlow-CLM data files should be prepared locally.

```text
data/parflow/normal_data/
├── press/
├── evaptrans/
└── static/

data/parflow/extra_data_index/
└── extra_sample_manifest.csv
```

The dynamic data are expected to use hourly ParFlow `.pfb` files. Normalization statistics are stored separately under `stats/`.

## Installation

```bash
conda env create -f environment.yml
conda activate predformer
pip install -e .
```

## Training

Edit the paths and options in the training command as needed:

```bash
python tools/train.py \
  --config-file configs/parflow/PredFormer.py \
  --data-root data/parflow/normal_data \
  --stats-path stats/stats_press_evap_static.npz \
  --res-dir work_dirs \
  --experiment-name ParFlow_press/example_FACTS \
  --batch-size 28 \
  --val-batch-size 28 \
  --epochs 60 \
  --learning-rate 3e-4 \
  --num-workers 16 \
  --var-name press \
  --use-evap \
  --use-static-input \
  --split-mode year \
  --train-years '[2019,2020]' \
  --holdout-years '[2021]' \
  --no-use-val
```

For extra-sample training, add:

```bash
--use-extra-data \
--extra-manifest-path data/parflow/extra_data_index/extra_sample_manifest.csv
```

## Inference

Run autoregressive inference from a trained work directory:

```bash
python -m model_deployment.inference \
  --run-dir work_dirs/ParFlow_press/example_FACTS \
  --checkpoint-file latest.pth \
  --data-root data/parflow/normal_data \
  --output-dir inference_data/press \
  --run-name example_rollout \
  --start-hour 20210012 \
  --end-hour 20218771 \
  --rollout-hours 720 \
  --patch-batch-size 28
```

The inference output is pressure head. WTD should be diagnosed from the predicted pressure-head fields using the same post-processing procedure as the ParFlow-CLM reference data.

## Notes

This repository is intended to provide the model code and reproducible workflow. Large datasets, trained checkpoints, raw model outputs, paper figures, and analysis products should be archived separately and referenced in the paper or documentation.
