from typing import Dict, List, Union
import numpy as np

import torch
from contextlib import suppress
from timm.utils.agc import adaptive_clip_grad

from openstl.core import metric
from openstl.core.optim_scheduler import get_optim_scheduler
from openstl.utils import ProgressBar

class Base_method(object):
    """Base Method.

    This class defines the basic functions of a video prediction (VP)
    method training and testing. Any VP method that inherits this class
    should at least define its own `train_one_epoch`, `vali_one_epoch`,
    and `test_one_epoch` function.

    """

    def __init__(self, args, device, steps_per_epoch):
        super(Base_method, self).__init__()
        self.args = args
        self.device = device
        self.config = args.__dict__
        self.criterion = None
        self.model_optim = None
        self.scheduler = None
        self.rank, self.world_size = 0, 1
        self.clip_value = self.args.clip_grad
        self.clip_mode = self.args.clip_mode if self.clip_value is not None else None
        # setup automatic mixed-precision (AMP) loss scaling and op casting
        self.amp_autocast = suppress  # do nothing
        self.loss_scaler = None
        # setup metrics 
        self.metric_list, self.spatial_norm = ['mse', 'rmse','mae','mape'], True

    def _build_model(self, **kwargs):
        raise NotImplementedError
    
        
    def _init_optimizer(self, steps_per_epoch):
        return get_optim_scheduler(
            self.args, self.args.epoch, self.model, steps_per_epoch)

    def train_one_epoch(self, runner, train_loader, **kwargs): 
        """Train the model with train_loader.

        Args:
            runner: the trainer of methods.
            train_loader: dataloader of train.
        """
        raise NotImplementedError

    def _predict(self, batch_x, batch_y, **kwargs):
        """Forward the model.

        Args:
            batch_x, batch_y: testing samples and groung truth.
        """
        raise NotImplementedError

    def _get_valid_spatial_size(self):
        target_h = getattr(self.args, 'space_h', None)
        target_w = getattr(self.args, 'space_w', None)
        if target_h is None or target_w is None:
            target_h = getattr(self.args, 'height', None)
            target_w = getattr(self.args, 'width', None)
        return target_h, target_w

    def _crop_to_valid_spatial(self, tensor):
        if tensor is None or tensor.ndim < 2:
            return tensor
        target_h, target_w = self._get_valid_spatial_size()
        if target_h is None or target_w is None:
            return tensor
        h, w = tensor.shape[-2], tensor.shape[-1]
        if h == target_h and w == target_w:
            return tensor
        if h < target_h or w < target_w:
            raise ValueError(
                f"Cannot crop tensor from spatial size {(h, w)} to larger target {(target_h, target_w)}."
            )
        return tensor[..., :target_h, :target_w]

    def _nondist_forward_collect(self, data_loader, length=None):
        """Forward and collect predictios.

        Args:
            data_loader: dataloader of evaluation.
            length (int): Expected length of output arrays.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        """
        # preparation
        dataset = data_loader.dataset
        results = []
        prog_bar = ProgressBar(len(data_loader))
        length = len(dataset) if length is None else length
        # collect all patch-level outputs first, then merge them after the full pass
        for batch_x, batch_y in data_loader:
            with torch.no_grad():
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self._predict(batch_x, batch_y)
                batch_x_eval = self._crop_to_valid_spatial(batch_x)
                batch_y_eval = self._crop_to_valid_spatial(batch_y)
                pred_y_eval = self._crop_to_valid_spatial(pred_y)

            loss_channels = getattr(self.args, "loss_channels", 10)
            if pred_y_eval.shape[2] < loss_channels or batch_y_eval.shape[2] < loss_channels:
                raise ValueError(
                    f"Loss expects at least {loss_channels} channels, got "
                    f"{pred_y_eval.shape[2]} (pred) and {batch_y_eval.shape[2]} (true)."
                )
            loss_value = self.criterion(
                pred_y_eval[:, :, :loss_channels, ...],
                batch_y_eval[:, :, :loss_channels, ...],
            ).detach().cpu().numpy().reshape(1)
            results.append(dict(zip(['inputs', 'preds', 'trues', 'loss'],
                                    [batch_x_eval.cpu().numpy(), pred_y_eval.cpu().numpy(), batch_y_eval.cpu().numpy(), loss_value])))

            prog_bar.update()
            if self.args.empty_cache:
                torch.cuda.empty_cache()

        results_all = {}
        for k in results[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in results], axis=0)
        results_all = self._merge_spatial_results(results_all, dataset)
        return results_all

    def _merge_spatial_results(self, results_all, dataset):
        """Reconstruct full-frame samples from spatially split patches.

        Args:
            results_all (dict): Concatenated outputs from the dataloader.
            dataset: Dataset instance that may contain spatial split metadata.

        Returns:
            dict: Potentially merged results with full-frame tensors and updated loss/metrics.
        """
        if not getattr(dataset, 'use_space', False):
            return results_all

        sample_indices = getattr(dataset, 'sample_indices', [])
        coords = getattr(dataset, 'space_coords', [(0, 0)])
        sample_count = len(results_all.get('preds', []))
        if len(sample_indices) == 0 or sample_count == 0:
            return results_all

        # Align to the actually collected samples to avoid indexing errors

        if sample_count != len(sample_indices):
            print(f"Warning: sample count ({sample_count}) does not match "
                  f"sample_indices length ({len(sample_indices)}). "
                  f"Truncating sample_indices.")
            sample_indices = sample_indices[:sample_count]

        # Build a robust mapping from raw time indices to contiguous slots
        # based on the actually available samples.
        if sample_indices and isinstance(sample_indices[0], tuple):
            time_slots = sorted({t for t, _ in sample_indices})
        else:
            time_slots = sorted(set(sample_indices))

        time_to_slot = {t: i for i, t in enumerate(time_slots)}
        num_sequences = len(time_slots)


        space_h = dataset.space_h
        space_w = dataset.space_w
        full_h, full_w = dataset.H, dataset.W

        def _merge_tensor(arr, seq_len):
            if arr.size == 0:
                return arr
            channels = arr.shape[2]
            merged = np.zeros((num_sequences, seq_len, channels, full_h, full_w), dtype=arr.dtype)
            counts = np.zeros_like(merged, dtype=np.int32)

            for idx, (t_idx, p_idx) in enumerate(sample_indices[: len(arr)]):
                slot = time_to_slot.get(t_idx)
                if slot is None:
                    continue
                top, left = coords[p_idx]
                merged[slot, :, :, top: top + space_h, left: left + space_w] += arr[idx]
                counts[slot, :, :, top: top + space_h, left: left + space_w] += 1

            # Avoid division-by-zero by keeping untouched regions at zero when no
            # tiles were written (should not happen for valid tiling setups).
            mask = counts > 0
            merged[mask] = merged[mask] / counts[mask]
            return merged

        if 'inputs' in results_all:
            results_all['inputs'] = _merge_tensor(results_all['inputs'], dataset.pre)
        results_all['preds'] = _merge_tensor(results_all['preds'], dataset.aft)
        results_all['trues'] = _merge_tensor(results_all['trues'], dataset.aft)

        return results_all

    def vali_one_epoch(self, runner, vali_loader, **kwargs):
        """Evaluate the model with val_loader.

        Args:
            runner: the trainer of methods.
            val_loader: dataloader of validation.

        Returns:
            list(tensor, ...): The list of predictions and losses.
            eval_log(str): The string of metrics.
        """
        self.model.eval()
        dataset = vali_loader.dataset
        results = self._nondist_forward_collect(vali_loader, len(dataset))

        eval_log = ""

        save_channels = getattr(self.args, "save_channels", None)
        preds_eval = results['preds']
        trues_eval = results['trues']
        if save_channels is not None:
            if preds_eval.shape[2] < save_channels or trues_eval.shape[2] < save_channels:
                raise ValueError(
                    f"Metrics expect at least {save_channels} channels, got "
                    f"{preds_eval.shape[2]} (pred) and {trues_eval.shape[2]} (true)."
                )
            preds_eval = preds_eval[:, :, :save_channels, ...]
            trues_eval = trues_eval[:, :, :save_channels, ...]
        eval_res, eval_log = metric(preds_eval, trues_eval,
                                    dataset.mean, dataset.std,
                                    metrics=self.metric_list, spatial_norm=self.spatial_norm)
        for k in self.metric_list:
            results[k] = np.array(eval_res[k]).reshape(1)
        results['metrics'] = np.array([eval_res['mae'], eval_res['mse'], eval_res['rmse'], eval_res['mape']])
        results['metric_dict'] = eval_res

        return results, eval_log

    def test_one_epoch(self, runner, test_loader, **kwargs):
        """Evaluate the model with test_loader.

        Args:
            runner: the trainer of methods.
            test_loader: dataloader of testing.

        Returns:
            list(tensor, ...): The list of inputs and predictions.
        """
        self.model.eval()
        dataset = test_loader.dataset
        results = self._nondist_forward_collect(test_loader, len(dataset))
            
        eval_log = ""
        save_channels = getattr(self.args, "save_channels", None)
        preds_eval = results['preds']
        trues_eval = results['trues']
        if save_channels is not None:
            if preds_eval.shape[2] < save_channels or trues_eval.shape[2] < save_channels:
                raise ValueError(
                    f"Metrics expect at least {save_channels} channels, got "
                    f"{preds_eval.shape[2]} (pred) and {trues_eval.shape[2]} (true)."
                )
            preds_eval = preds_eval[:, :, :save_channels, ...]
            trues_eval = trues_eval[:, :, :save_channels, ...]
        eval_res, eval_log = metric(preds_eval, trues_eval,
                                    dataset.mean, dataset.std,
                                    metrics=self.metric_list, spatial_norm=self.spatial_norm)
        for k in self.metric_list:
            results[k] = np.array(eval_res[k]).reshape(1)
        results['metrics'] = np.array([eval_res['mae'], eval_res['mse'], eval_res['rmse'], eval_res['mape']])
        results['metric_dict'] = eval_res

        results['eval_log'] = eval_log
        
        return results

    def current_lr(self) -> Union[List[float], Dict[str, List[float]]]:
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """
        lr: Union[List[float], Dict[str, List[float]]]
        if isinstance(self.model_optim, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.model_optim.param_groups]
        elif isinstance(self.model_optim, dict):
            lr = dict()
            for name, optim in self.model_optim.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    def clip_grads(self, params, norm_type: float = 2.0):
        """ Dispatch to gradient clipping method

        Args:
            parameters (Iterable): model parameters to clip
            value (float): clipping value/factor/norm, mode dependant
            mode (str): clipping mode, one of 'norm', 'value', 'agc'
            norm_type (float): p-norm, default 2.0
        """
        if self.clip_mode is None:
            return
        if self.clip_mode == 'norm':
            torch.nn.utils.clip_grad_norm_(params, self.clip_value, norm_type=norm_type)
        elif self.clip_mode == 'value':
            torch.nn.utils.clip_grad_value_(params, self.clip_value)
        elif self.clip_mode == 'agc':
            adaptive_clip_grad(params, self.clip_value, norm_type=norm_type)
        else:
            assert False, f"Unknown clip mode ({self.clip_mode})."
