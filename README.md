# ParFlow-PredFormer

## 项目结构

```text
configs/parflow/       实验和模型配置
openstl/api/           训练流程
openstl/methods/       PredFormer 训练方法
openstl/models/        模型实现
openstl/datasets/      ParFlow 数据读取、切分和采样
model_deployment/      推理入口
tools/                 manifest 和统计量生成工具
scripts/parflow/       训练脚本
data/parflow/extra_data_index/  本地极端事件索引
extra_data/            可提交的极端事件 manifest 备份
```

## 安装

```bash
conda env create -f environment.yml
conda activate predformer
pip install -e .
```


