import argparse
import json
import os
import warnings
warnings.filterwarnings('ignore')
from types import SimpleNamespace
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from openstl.api import BaseExperiment
from openstl.core import metric
from parflow.tools.io import write_pfb
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



def _save_volume_as_pfb(
    volume: np.ndarray,
    output_dir: str,
    base_name: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    volume = np.asarray(volume, dtype=np.float32)
    path = os.path.join(output_dir, f'{base_name}.pfb')
    write_pfb(path, volume)
    return path


def _save_predictions(
    preds: np.ndarray,
    dataset,
    output_dir: str,
) -> List[Dict[str, Union[int, str]]]:
    index: List[Dict[str, Union[int, str]]] = []

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
            saved_paths = _save_volume_as_pfb(volume, output_dir, base_name)
            index.append({
                'sample_index': sample_idx,
                'time_index': int(target_idx),
                'step_index': step_idx,
                'file': base_name,
                'pfb': saved_path,
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
                        help='Directory to store exported prediction PFB files. Defaults to <work_dir>/pred_result.')
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
                 'work_dir', 'output_dir', 'checkpoint', 'save_npz'}
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

    index = _save_predictions(preds, dataset, output_dir)

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