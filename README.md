# Systematic Evaluation of a Transformer-Based Surrogate for Distributed Groundwater Dynamics: Long-Horizon Prediction, Dynamic-Static Fusion, and Data Augmentation

Chen Yang <yangch329@mail.sysu.edu.cn>  
Hui Huang <huangh573@mail2.sysu.edu.cn>

This repository contains the source code for the PredFormer-based groundwater surrogate model used in the manuscript. The model emulates ParFlow-CLM pressure-head dynamics, and water table depth (WTD) is diagnosed from the predicted pressure-head fields for evaluation.



## Environment Setup

Create the conda environment and install the package in editable mode:

```bash
conda env create -f environment.yml
conda activate predformer
pip install -e .
```

## Repository Structure

```text
configs/              Model and experiment configuration files
openstl/              Dataset, model, method, training, metrics, and utility code
model_deployment/     Inference code for trained PredFormer checkpoints
tools/                Training entry point and data-preparation utilities
scripts/parflow/      Training and inference scripts
data/                 ParFlow-CLM data layout;
stats/                Normalization files; only one smoke-test stats file is tracked
work_dirs/            Training outputs;
```

## Example Run

Run training:

```bash
bash scripts/parflow/parflow_PredFormer_FacTS_train.sh
```

Run inference:

```bash
bash scripts/parflow/parflow_PredFormer_FacTS_inference.sh
```

Before running, edit the paths and options at the top of each script. The training script also contains notes for enabling extra-data training.
