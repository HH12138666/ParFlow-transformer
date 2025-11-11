import argparse
import json
import os
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from openstl.api import BaseExperiment
from openstl.utils import default_parser, setup_multi_processes
from parflow.tools.io import write_pfb


def _load_experiment_args(work_dir: str,
                          checkpoint: Optional[str],
                          device: str) -> SimpleNamespace:
    """Load the training configuration from ``model_param.json``.

    The helper fills in any missing defaults required by :class:`BaseExperiment`
    and points ``resume_from`` to the desired checkpoint so the weights are
    restored before running inference.
    """
    param_file = os.path.join(work_dir, 'model_param.json')
    if not os.path.isfile(param_file):
        raise FileNotFoundError(
            f'Cannot find "model_param.json" in work_dir: {work_dir}')

    with open(param_file, 'r') as fp:
        saved_cfg = json.load(fp)

    defaults = default_parser()
    for key, value in defaults.items():
        saved_cfg.setdefault(key, value)

    # Resolve the checkpoint path before mutating the dict.
    checkpoint_path = checkpoint or os.path.join(work_dir, 'checkpoints', 'checkpoint.pth')
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            'No checkpoint provided and the default path was not found: '
            f'{checkpoint_path}')

    # Force a pure inference setup.
    saved_cfg.update({
        'dist': False,
        'launcher': 'none',
        'test': True,
        'inference': False,
        'auto_resume': False,
        'resume_from': checkpoint_path,
        'device': device,
        'use_gpu': device.startswith('cuda') and torch.cuda.is_available(),
        'res_dir': os.path.dirname(work_dir) or '.',
        'ex_name': os.path.basename(work_dir.rstrip(os.sep)),
        'tb_dir': os.path.join(work_dir, 'logs_tb'),
    })

    return SimpleNamespace(**saved_cfg)


def _unnormalize(arr: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> np.ndarray:
    if mean is None or std is None:
        return arr
    mean = mean.reshape(1, 1, -1, 1, 1)
    std = std.reshape(1, 1, -1, 1, 1)
    return arr * std + mean


def _save_predictions(preds: np.ndarray, dataset, output_dir: str) -> int:
    """Persist predictions to ``.pfb`` files matching the dataset chronology."""

    start_indices = dataset.start_indices
    pre = dataset.pre
    source_files = dataset.files

    os.makedirs(output_dir, exist_ok=True)

    total_saved = 0
    for sample_idx, start in enumerate(start_indices):
        sample_preds = preds[sample_idx]
        for step_idx, field in enumerate(sample_preds):
            target_idx = start + pre + step_idx
            if 0 <= target_idx < len(source_files):
                base_name = os.path.basename(source_files[target_idx])
            else:
                base_name = f'sample{sample_idx:05d}_step{step_idx:02d}.pfb'

            dest_path = os.path.join(output_dir, base_name)
            if os.path.exists(dest_path):
                root, ext = os.path.splitext(base_name)
                dest_path = os.path.join(
                    output_dir,
                    f'{root}_sample{sample_idx:05d}_step{step_idx:02d}{ext}')

            write_pfb(dest_path, field.astype(np.float32, copy=False))
            total_saved += 1

    return total_saved


def main():
    parser = argparse.ArgumentParser(description='Run ParFlow inference and export predictions as .pfb files.')
    parser.add_argument('--work-dir', required=True, help='Path to the experiment work_dir created during training.')
    parser.add_argument('--checkpoint', help='Checkpoint path to load. Defaults to work_dir/checkpoints/checkpoint.pth')
    parser.add_argument('--device', default='cuda', help='Device used for inference (cuda or cpu).')
    parser.add_argument('--output-dir', help='Directory to store prediction .pfb files. Defaults to <work_dir>/pred_result.')
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    checkpoint = os.path.abspath(args.checkpoint) if args.checkpoint else None

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA is not available. Falling back to CPU inference.')
        device = 'cpu'

    exp_args = _load_experiment_args(work_dir, checkpoint, device)
    setup_multi_processes(exp_args.__dict__)

    exp = BaseExperiment(exp_args)
    exp_args.resume_from = None  # avoid re-loading if hooks use it

    exp.call_hook('before_val_epoch')
    results = exp.method.test_one_epoch(exp, exp.test_loader)
    exp.call_hook('after_val_epoch')

    preds = np.asarray(results['preds'])  # [N, aft, C, H, W]
    dataset = exp.test_loader.dataset
    mean = dataset.mean
    std = dataset.std

    preds = _unnormalize(preds, mean, std)

    output_dir = args.output_dir or os.path.join(work_dir, 'pred_result')
    saved = _save_predictions(preds, dataset, output_dir)
    print(f'Saved {saved} prediction frames to {output_dir}')


if __name__ == '__main__':
    main()
