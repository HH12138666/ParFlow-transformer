import os
from pathlib import Path

import inference as base


# ---- User configuration ----
# 训练输出根目录（与 inference.py 保持一致，可自行修改）
WORK_DIR = "/home/huanghui/data/ParFlow-transformer/work_dirs/ParFlow_press"
CHECKPOINT_NAME = "2026-03-26-22-11_FACTS"

# 使用 latest.pth 做推理（用于续训中间状态快速验证）
CHECKPOINT_PATH = os.path.join(WORK_DIR, CHECKPOINT_NAME, "checkpoints", "latest.pth")


def main():
    # 复用原推理脚本的其余配置与流程，只覆盖 checkpoint 路径
    base.WORK_DIR = WORK_DIR
    base.CHECKPOINT_NAME = CHECKPOINT_NAME
    base.CHECKPOINT_PATH = CHECKPOINT_PATH

    if not Path(base.CHECKPOINT_PATH).exists():
        raise FileNotFoundError(f"latest checkpoint not found: {base.CHECKPOINT_PATH}")

    print(f"[checkpoint] using latest: {base.CHECKPOINT_PATH}")
    base.main()


if __name__ == "__main__":
    main()
