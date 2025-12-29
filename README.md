# ParFlow-PredFormer
Replacing traditional ​ParFlow hydrological model​ with ​PredFormer, a large-scale Transformer-based model, for more efficient and generalizable water flow simulation.


## Installation

```
# Exp Setting: PyTorch: 2.1.0+ Python 3.10
conda env create -f environment.yml  
conda activate predformer
pip install -e .
pip install einops
```


将切到当前仓库根目录（根据你的路径调整为真实绝对路径）
#在ParFlow-transformer/scripts/parflow/parflow_PredFormer_FacTS_train.sh文件中修改
REPO=/home/huanghui/data/ParFlow-transformer（根据你的路径调整为真实绝对路径）

