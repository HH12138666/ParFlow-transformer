# Systematic Evaluation of a Transformer-Based Surrogate for Distributed Groundwater Dynamics: Long-Horizon Prediction, Dynamic-Static Fusion, and Data Augmentation

This repository contains the source code for the PredFormer-based groundwater surrogate model used in the manuscript. The model emulates ParFlow-CLM pressure-head dynamics, and water table depth (WTD) is diagnosed from the predicted pressure-head fields for evaluation.

This repository is intended as a reproducible code release. Full ParFlow-CLM simulations, full checkpoints, manuscript figure data, and paper-specific analysis products are archived separately in the accompanying data package. Only a small smoke-test subset is retained here so that the training and inference workflows can be executed after cloning.

## Environment Setup

Create the conda environment and install the package in editable mode:

```bash
conda env create -f environment.yml
conda activate predformer
pip install -e .
```

The code was developed for GPU execution with PyTorch. The example scripts under `scripts/parflow/` select one GPU through `CUDA_VISIBLE_DEVICES`; this can be changed in the configuration block at the top of each script or overridden from the shell.

## Repository Structure

```text
configs/              Model and experiment configuration files
openstl/              Dataset, model, method, training, metrics, and utility code
model_deployment/     Inference code for trained PredFormer checkpoints
tools/                Training entry point and data-preparation utilities
scripts/parflow/      Minimal training and inference example scripts
data/                 ParFlow-CLM data layout; only a 36 h smoke-test subset is tracked
stats/                Normalization files; only one smoke-test stats file is tracked
work_dirs/            Training outputs; only one lightweight state-dict checkpoint is tracked
inference_data/       Local inference outputs, ignored by git
```

The tracked lightweight checkpoint is provided only to test the inference pipeline. It stores model weights only (`state_dict`) and is not the full training checkpoint used for manuscript analysis.

## Data Layout

The code expects ParFlow-CLM files in the following layout:

```text
data/parflow/normal_data/
├── press/
├── evaptrans/
└── static/

stats/
└── stats1_press_evap_static_2019_2020.npz

work_dirs/ParFlow_press/2026-07-11-18-00_FACTS/
├── model_param.json
└── checkpoints/
    └── latest_state_dict.pth
```

The smoke-test subset retained in this repository contains only:

```text
press:     20190000-20190035
evaptrans: 20190001-20190036
static:    static.pfb
```

This subset is sufficient to test `12 h input -> 12 h output` training and short autoregressive inference. It is not intended for scientific evaluation. In a local working copy containing full-year data, the same scripts will read all available files in `data/parflow/normal_data`; the GitHub release tracks only the selected 36 h subset.

For full manuscript-scale experiments, place the complete ParFlow-CLM data in the same directory structure and use the corresponding full stats files and checkpoints from the accompanying data archive.

## Path Configuration Notes

The example scripts are configured through variables at the top of each shell script. The default paths are relative to the repository root:

```bash
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
DATA_ROOT=${DATA_ROOT:-"$PROJECT_ROOT/data/parflow/normal_data"}
```

The same variables can be overridden from the shell without editing the script. For example:

```bash
DATA_ROOT=/path/to/full/parflow/normal_data \
STATS_PATH=/path/to/stats_file.npz \
bash scripts/parflow/parflow_PredFormer_FacTS_train.sh
```

For inference, the checkpoint directory and checkpoint file can be overridden in the same way:

```bash
RUN_DIR=/path/to/work_dirs/ParFlow_press/run_name \
CHECKPOINT_FILE=latest.pth \
DATA_ROOT=/path/to/full/parflow/normal_data \
bash scripts/parflow/parflow_PredFormer_FacTS_inference.sh
```

The inference output is pressure head. WTD should be diagnosed from the predicted pressure-head fields using the same post-processing procedure as for the ParFlow-CLM reference data.

## Example Run

Run the 1-epoch smoke-test training workflow:

```bash
bash scripts/parflow/parflow_PredFormer_FacTS_train.sh
```

The default training example uses:

```text
DATA_ROOT:       data/parflow/normal_data
STATS_PATH:      stats/stats1_press_evap_static_2019_2020.npz
TRAIN_YEARS:     [2019]
HOLDOUT_YEARS:   [2019]
EPOCHS:          1
BATCH_SIZE:      2
NUM_WORKERS:     0
```

Run the short inference workflow using the lightweight checkpoint:

```bash
bash scripts/parflow/parflow_PredFormer_FacTS_inference.sh
```

The default inference example uses:

```text
RUN_DIR:          work_dirs/ParFlow_press/2026-07-11-18-00_FACTS
CHECKPOINT_FILE:  latest_state_dict.pth
DATA_ROOT:        data/parflow/normal_data
START_HOUR:       20190000
END_HOUR:         20190035
ROLLOUT_HOURS:    12
```

The example inference writes predicted pressure-head PFB files under:

```text
inference_data/example_press/
```

These outputs are ignored by git.
