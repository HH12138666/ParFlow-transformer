# A Transformer-Based Surrogate for Distributed Groundwater Dynamics with Zero-Shot Cross-Basin Transfer and Precipitation-Regime Data Augmentation

Chen Yang <yangch329@mail.sysu.edu.cn>  
Hui Huang <huangh573@mail2.sysu.edu.cn>

This repository contains the source code for the PredFormer-based groundwater surrogate model used in the manuscript. The model emulates ParFlow-CLM pressure-head dynamics, and water table depth (WTD) is diagnosed from the predicted pressure-head fields for evaluation.

The source region is the upper Beijiang River basin (UBRB). Zero-shot evaluation applies the same trained model and UBRB normalization statistics to Hejiang WY2021, without target-basin training or fine-tuning. PME is prescribed offline; these experiments are not online CLM coupling.


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

Run the Hejiang zero-shot experiment after downloading its data:

```bash
bash scripts/parflow/parflow_PredFormer_zero_shot_inference.sh
```

Before running, edit the paths and options at the top of each script. The training script also contains notes for enabling extra-data training. The bundled 36-hour data are only a smoke test, not the full manuscript experiment.

## Data and Figure Reproduction

The study uses two complementary ScienceDB datasets:

| Dataset | Contents | DOI |
|---|---|---|
| Ordinary UBRB simulations | Unperturbed ParFlow-CLM simulations used to construct the baseline surrogate-model dataset. | <https://doi.org/10.57760/sciencedb.46348> |
| Supporting data for this study | Precipitation-perturbed simulations, Hejiang reference simulations and model predictions, augmentation manifests, trained checkpoints, normalization files, and figure/table scripts and source data. | <https://doi.org/10.57760/sciencedb.0106u> |

The ordinary UBRB training fields are not duplicated in the supporting dataset. Follow the supporting dataset's root README for the current figure numbering and extraction layout; large archives and local analyses are not added to this code repository.

The surrogate predicts ten-layer pressure head. The paper's WTD diagnosis also uses the corresponding reference saturation; it is not a separately predicted surrogate variable. The evaluation takes the absolute difference at each grid cell before spatial averaging. Cross-basin tiled inference retains the target grid geometry and does not resize Hejiang to the UBRB domain.
