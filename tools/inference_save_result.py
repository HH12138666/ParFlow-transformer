import argparse
import json
import os
import warnings
warnings.filterwarnings('ignore')
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Union
import sys
import numpy as np
import torch
from openstl.api import BaseExperiment
from openstl.core import metric
from openstl.utils import (
    check_dir,
    create_parser,
    default_parser,
    get_dist_info,
    load_config,
    print_log,
    setup_multi_processes,
    update_config,
)


try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        'Pillow is required to export prediction images. '
        'Please install it with `pip install pillow`.') from exc

try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False


def _ensure_defaults(config: Dict) -> Dict:

    defaults = default_parser()
    for key, value in defaults.items():
        if config.get(key) is None:
            config[key] = value
    return config

def _load_saved_config(work_dir: str) -> Dict:
    param_file = os.path.join(work_dir, 'model_param.json')
    if not os.path.isfile(param_file):
        return {}
    with open(param_file, 'r', encoding='utf-8') as fp:
        return json.load(fp)
    
    
def _select_checkpoint(work_dir: str, checkpoint: Optional[str]) -> str:
    if checkpoint:
        ckpt_path = os.path.abspath(checkpoint)
    else:
        ckpt_path = os.path.join(work_dir, 'checkpoints', 'checkpoint.pth')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f'Unable to locate checkpoint file: {ckpt_path}')
    return ckpt_path




def _unnormalize(arr, mean, std) :
    if mean is None or std is None:
        return arr
    mean = mean.reshape(1, 1, -1, 1, 1)
    std = std.reshape(1, 1, -1, 1, 1)
    return arr * std + mean


def _prepare_volume(field) :
    """Ensure a prediction slice is a contiguous 3-D ``float32`` volume."""
 
    volume = np.asarray(field)
    if volume.ndim == 5:
        # [B, T, C, H, W] -> assume batch of 1 and single step
        if volume.shape[0] != 1 or volume.shape[1] != 1:
            raise ValueError(
                'Expected a single prediction sample, '
                f'got tensor with shape {volume.shape}.')
        volume = volume[0, 0]
    if volume.ndim == 4:
        if volume.shape[0] == 1:
            volume = volume[0]
        else:
            raise ValueError(
                f'Unable to convert tensor with shape {volume.shape} to an image.')
    if volume.ndim == 2:
        volume = volume[np.newaxis, ...]
    if volume.ndim != 3:
        raise ValueError(
            f'Prediction must be 2-D or channel-first 3-D. Got shape {volume.shape}.')
    return np.asarray(volume, dtype=np.float32)


def _to_numpy(array) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data_min = float(data.min())
    data = data - data_min
    data_max = float(data.max())
    if data_max > 0:
        data = data / data_max
    data = (data * 255.0).clip(0, 255).astype(np.uint8)
    return data

def _to_pil_image(volume: np.ndarray) -> Image.Image:
    if volume.ndim == 2:
        array = _normalize_to_uint8(volume)
        return Image.fromarray(array, mode='L')

    if volume.ndim != 3:
        raise ValueError(f'Expected a 2-D array or 3-D volume, got {volume.shape}.')

    channels_first = volume.shape[0] <= 4
    if channels_first:
        volume = np.moveaxis(volume, 0, -1)

    if volume.shape[-1] == 1:
        array = _normalize_to_uint8(volume[..., 0])
        return Image.fromarray(array, mode='L')
    if volume.shape[-1] == 3:
        array = _normalize_to_uint8(volume)
        return Image.fromarray(array, mode='RGB')

    raise ValueError(
        'Prediction has more than 3 channels. Consider saving channel-wise images '
        f'or reduce the output dimension. Got shape {volume.shape}.')


def _save_volume_as_images(
    volume: np.ndarray,
    output_dir: str,
    base_name: str,
    channel_suffix: bool,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)

    if volume.shape[0] in (1, 3):
        image = _to_pil_image(volume)
        path = os.path.join(output_dir, f'{base_name}.png')
        image.save(path)
        return [path]

    paths: List[str] = []
    for channel, slice_ in enumerate(volume):
        image = _to_pil_image(slice_)
        suffix = f'_c{channel}' if channel_suffix else ''
        path = os.path.join(output_dir, f'{base_name}{suffix}.png')
        image.save(path)
        paths.append(path)
    return paths


def _save_predictions(
    preds: np.ndarray,
    dataset,
    output_dir: str,
    channel_suffix: bool,
) -> List[Dict[str, Union[int, str, Sequence[str]]]]:
    index: List[Dict[str, Union[int, str, Sequence[str]]]] = []

    start_indices = getattr(dataset, 'start_indices', list(range(len(preds))))
    pre = getattr(dataset, 'pre', 0)
    files = getattr(dataset, 'files', None)

    for sample_idx, sample_preds in enumerate(preds):
        start = start_indices[sample_idx] if sample_idx < len(start_indices) else 0
        for step_idx, field in enumerate(sample_preds):
            volume = _prepare_volume(field)
            target_idx = start + pre + step_idx
            
            base_name = f'sample{sample_idx:05d}_step{step_idx:02d}'
            if files and 0 <= target_idx < len(files):
                file_name = os.path.splitext(os.path.basename(files[target_idx]))[0]
                base_name = f'{file_name}'
            saved_paths = _save_volume_as_images(volume, output_dir, base_name, channel_suffix)
            index.append({
                'sample_index': sample_idx,
                'time_index': int(target_idx),
                'step_index': step_idx,
                'file': base_name,
                'images': saved_paths,
            })
    return index





def _unnormalize(preds: np.ndarray, mean, std) -> np.ndarray:
    if mean is None or std is None:
        return preds
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1, 1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, -1, 1, 1)
    return preds * std + mean


def _run_inference(exp: BaseExperiment, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError(
            f'Invalid checkpoint format at {checkpoint_path}. Expected a dict with "state_dict".')

    exp._load_from_state_dict(state_dict)
    

    exp.call_hook('before_val_epoch')
    results = exp.method.test_one_epoch(exp, exp.test_loader)
    exp.call_hook('after_val_epoch')
    return results

def main() -> None:
    parser = create_parser()
    parser.add_argument('--work-dir', type=str, help='Path to the trained experiment directory.')
    parser.add_argument('--checkpoint', type=str, help='Optional checkpoint path to load.')
    parser.add_argument('--output-dir', type=str,
                        help='Directory to store exported prediction images. Defaults to <work_dir>/pred_result.')
    parser.add_argument('--channel-suffix', action='store_true',
                        help='Append channel indices to filenames even for single-channel outputs.')
    parser.add_argument('--save-npz', action='store_true',
                        help='Also store raw prediction arrays as predictions.npz in the output directory.')

    args = parser.parse_args()
    config = vars(args).copy()

    if has_nni:
        tuner_params = nni.get_next_parameter()
        config.update(tuner_params)

    if config.get('config_file') is None:
        raise ValueError('Config file is required for inference.')

    loaded_cfg = load_config(config['config_file'])
    config = update_config(config, loaded_cfg, exclude_keys=['method', 'batch_size', 'val_batch_size'])
    config = _ensure_defaults(config)

    work_dir = config.get('work_dir') or os.path.join(config['res_dir'], config['ex_name'])
    if not work_dir:
        raise ValueError('Unable to infer work_dir. Please specify --work-dir.')
    work_dir = os.path.abspath(work_dir)
    if not os.path.isdir(work_dir):
        raise FileNotFoundError(f'work_dir does not exist: {work_dir}')
    config['work_dir'] = work_dir

    saved_cfg = _load_saved_config(work_dir)
    preserved = {'device', 'dist', 'launcher', 'resume_from', 'auto_resume', 'test', 'inference',
                 'work_dir', 'output_dir', 'checkpoint', 'channel_suffix', 'save_npz'}
    for key, value in saved_cfg.items():
        if key not in preserved:
            config[key] = value

    res_dir = os.path.dirname(work_dir) or '.'
    ex_name = os.path.basename(work_dir.rstrip(os.sep))
    config['res_dir'] = res_dir
    config['ex_name'] = ex_name
    config['tb_dir'] = os.path.join(work_dir, 'logs_tb')

    config['dist'] = False
    config['launcher'] = 'none'
    config['test'] = False
    config['inference'] = True
    config['auto_resume'] = False
    config['resume_from'] = None

    device = config.get('device', 'cuda')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        warnings.warn('CUDA is not available. Falling back to CPU inference.')
        device = 'cpu'
    config['device'] = device
    config['use_gpu'] = str(device).startswith('cuda') and torch.cuda.is_available()

    config.setdefault('opencv_num_threads', 0)
    config.setdefault('mp_start_method', 'fork')

    setup_multi_processes(config)

    exp_args = SimpleNamespace(**config)
    exp = BaseExperiment(exp_args)

    checkpoint_path = _select_checkpoint(work_dir, config.get('checkpoint'))
    results = _run_inference(exp, checkpoint_path)

    preds = _to_numpy(results['preds'])

    dataset = exp.test_loader.dataset
    mean = getattr(dataset, 'mean', None)
    std = getattr(dataset, 'std', None)
    preds = _unnormalize(preds, mean, std)

    output_dir = config.get('output_dir') or os.path.join(work_dir, 'pred_result')
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    index = _save_predictions(preds, dataset, output_dir, channel_suffix=config.get('channel_suffix', False))

    if config.get('save_npz', False):
        np.savez_compressed(os.path.join(output_dir, 'predictions.npz'), preds=preds)

    with open(os.path.join(output_dir, 'index.json'), 'w', encoding='utf-8') as fp:
        json.dump(index, fp, indent=2)

    metric_list = getattr(exp_args, 'metrics', None)
    spatial_norm = True
    channel_names = None
    eval_res, eval_log = metric(results['preds'], results['trues'],
                                dataset.mean, dataset.std,
                                metrics=metric_list,
                                channel_names=channel_names,
                                spatial_norm=spatial_norm)

    results['metrics'] = np.array([eval_res['mae'], eval_res['mse'], eval_res['rmse'], eval_res['mape']])

    rank, _ = get_dist_info()
    if rank == 0:
        print_log(eval_log)
        saved_dir = os.path.join(work_dir, 'saved')
        check_dir(saved_dir)
        for key in ['metrics', 'inputs', 'trues', 'preds']:
            np.save(os.path.join(saved_dir, f'{key}.npy'), _to_numpy(results[key]))
        print(f'Saved {len(index)} prediction steps to {output_dir}.')
        print(f'Metrics: MAE={eval_res["mae"]:.6f}, MSE={eval_res["mse"]:.6f}, '
              f'RMSE={eval_res["rmse"]:.6f}, MAPE={eval_res["mape"]:.6f}')

        if has_nni:
            nni.report_final_result(eval_res['mse'])


if __name__ == '__main__':
    main()