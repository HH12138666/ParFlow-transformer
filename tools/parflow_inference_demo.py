import argparse
import json
import os
from types import SimpleNamespace
from typing import Optional

import matplotlib

# Use a non-interactive backend when no display is available (e.g. on servers).
if os.name != 'nt' and not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch

from openstl.api import BaseExperiment
from openstl.utils import default_parser, setup_multi_processes


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


def _select_frame(sequence: np.ndarray, time_idx: int, channel: int) -> np.ndarray:
    if sequence.ndim != 4:
        raise ValueError(f'Expected [T, C, H, W] array, got shape {sequence.shape}')
    if not (0 <= time_idx < sequence.shape[0]):
        raise IndexError(f'time index {time_idx} out of range for length {sequence.shape[0]}')
    if not (0 <= channel < sequence.shape[1]):
        raise IndexError(f'channel index {channel} out of range for {sequence.shape[1]} channels')
    return sequence[time_idx, channel]


def _plot_fields(observed: np.ndarray,
                 target: np.ndarray,
                 prediction: np.ndarray,
                 output: Optional[str],
                 show: bool) -> None:
    diff = prediction - target
    vmin = np.min([observed.min(), target.min(), prediction.min()])
    vmax = np.max([observed.max(), target.max(), prediction.max()])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    titles = ['Last observed frame', 'Ground truth', 'Prediction', 'Prediction − Truth']
    data = [observed, target, prediction, diff]
    cmaps = ['viridis', 'viridis', 'viridis', 'coolwarm']
    norms = [(vmin, vmax), (vmin, vmax), (vmin, vmax), (diff.min(), diff.max())]

    for ax, title, field, cmap, (fmin, fmax) in zip(axes, titles, data, cmaps, norms):
        im = ax.imshow(field, cmap=cmap, vmin=fmin, vmax=fmax)
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=200)
        print(f'Saved figure to {output}')
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Run ParFlow inference and visualize predictions.')
    parser.add_argument('--work-dir', required=True, help='Path to the experiment work_dir created during training.')
    parser.add_argument('--checkpoint', help='Checkpoint path to load. Defaults to work_dir/checkpoints/checkpoint.pth')
    parser.add_argument('--device', default='cuda', help='Device used for inference (cuda or cpu).')
    parser.add_argument('--sample-index', type=int, default=0, help='Index of the sequence in the test set to visualize.')
    parser.add_argument('--pred-step', type=int, default=0, help='Target step to visualize (0 <= pred_step < aft_seq_length).')
    parser.add_argument('--channel', type=int, default=0, help='Channel index to visualize.')
    parser.add_argument('--output', help='Optional path to save the comparison figure.')
    parser.add_argument('--show', action='store_true', help='Display the matplotlib window after saving.')
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

    inputs = results['inputs']  # [N, pre, C, H, W]
    trues = results['trues']    # [N, aft, C, H, W]
    preds = results['preds']    # [N, aft, C, H, W]

    if args.sample_index >= inputs.shape[0]:
        raise IndexError(f'sample_index {args.sample_index} >= dataset size {inputs.shape[0]}')

    dataset = exp.test_loader.dataset
    mean = dataset.mean
    std = dataset.std

    inputs = _unnormalize(inputs, mean, std)
    trues = _unnormalize(trues, mean, std)
    preds = _unnormalize(preds, mean, std)

    sequence_inputs = inputs[args.sample_index]
    sequence_trues = trues[args.sample_index]
    sequence_preds = preds[args.sample_index]

    if not (0 <= args.pred_step < sequence_trues.shape[0]):
        raise IndexError(
            f'pred_step {args.pred_step} out of range for {sequence_trues.shape[0]} target steps')

    last_observed = _select_frame(sequence_inputs, sequence_inputs.shape[0] - 1, args.channel)
    target_frame = _select_frame(sequence_trues, args.pred_step, args.channel)
    predicted_frame = _select_frame(sequence_preds, args.pred_step, args.channel)

    _plot_fields(last_observed, target_frame, predicted_frame, args.output, args.show)


if __name__ == '__main__':
    main()
