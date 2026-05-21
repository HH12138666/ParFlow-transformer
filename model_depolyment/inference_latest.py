from pathlib import Path

import inference as base_infer


LATEST_CKPT_NAME = "latest.pth"
RUN_PARAM_SUFFIX = "latest"
LATEST_CKPT_DIR = "checkpoints"

# sbatch /home/huanghui/data/slurm_job/inference_latest.sh
def _build_latest_checkpoint_path():
    work_dir = Path(base_infer.WORK_DIR)
    ckpt_name = base_infer.CHECKPOINT_NAME
    return str(work_dir / ckpt_name / LATEST_CKPT_DIR / LATEST_CKPT_NAME)


def _build_run_param():
    run_param = str(base_infer.RUN_PARAM).strip()
    if not run_param:
        return RUN_PARAM_SUFFIX
    if run_param.endswith(f"_{RUN_PARAM_SUFFIX}") or run_param == RUN_PARAM_SUFFIX:
        return run_param
    return f"{run_param}_{RUN_PARAM_SUFFIX}"


def main():
    base_infer.CHECKPOINT_PATH = _build_latest_checkpoint_path()
    base_infer.RUN_PARAM = _build_run_param()
    print(f"[checkpoint] use latest checkpoint: {base_infer.CHECKPOINT_PATH}")
    base_infer.main()


if __name__ == "__main__":
    main()
