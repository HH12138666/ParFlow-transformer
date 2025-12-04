from typing import Dict, List, Union
import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from contextlib import suppress
from timm.utils import NativeScaler
from timm.utils.agc import adaptive_clip_grad

from openstl.core import metric
from openstl.core.optim_scheduler import get_optim_scheduler
from openstl.utils import gather_tensors_batch, get_dist_info, ProgressBar

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass


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
        self.dist = args.dist
        self.device = device
        self.config = args.__dict__
        self.criterion = None
        self.model_optim = None
        self.scheduler = None
        if self.dist:
            self.rank, self.world_size = get_dist_info()
            assert self.rank == int(device.split(':')[-1])
        else:
            self.rank, self.world_size = 0, 1
        self.clip_value = self.args.clip_grad
        self.clip_mode = self.args.clip_mode if self.clip_value is not None else None
        # setup automatic mixed-precision (AMP) loss scaling and op casting
        self.amp_autocast = suppress  # do nothing
        self.loss_scaler = None
        # setup metrics 修改
        self.metric_list, self.spatial_norm = ['mse', 'rmse','mae','mape'], True

    def _build_model(self, **kwargs):
        raise NotImplementedError
    
        
    def _init_optimizer(self, steps_per_epoch):
        return get_optim_scheduler(
            self.args, self.args.epoch, self.model, steps_per_epoch)

    def _init_distributed(self):
        """Initialize DDP training"""
        if self.args.fp16 and has_native_amp:
            self.amp_autocast = torch.cuda.amp.autocast
            self.loss_scaler = NativeScaler()
            if self.rank == 0:
               print('Using native PyTorch AMP. Training in mixed precision (fp16).')
        else:
            print('AMP not enabled. Training in float32.')
        self.model = NativeDDP(self.model, device_ids=[self.rank],
                               broadcast_buffers=self.args.broadcast_buffers,
                               find_unused_parameters=self.args.find_unused_parameters)

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

    def _dist_forward_collect(self, data_loader, length=None, gather_data=False):
        """Forward and collect predictios in a distributed manner.

        Args:
            data_loader: dataloader of evaluation.
            length (int): Expected length of output arrays.
            gather_data (bool): Whether to gather raw predictions and inputs.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        """
        # preparation
        results = []
        length = len(data_loader.dataset) if length is None else length
        if self.rank == 0:
            prog_bar = ProgressBar(len(data_loader))

        # loop
        for idx, (batch_x, batch_y) in enumerate(data_loader):
            if idx == 0:
                part_size = batch_x.shape[0]
            with torch.no_grad():
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self._predict(batch_x, batch_y)

            if gather_data:  # return raw datas
                loss_value = self.criterion(pred_y, batch_y).detach().cpu().numpy().reshape(1)
                results.append(dict(zip(['inputs', 'preds', 'trues', 'loss'],
                                        [batch_x.cpu().numpy(), pred_y.cpu().numpy(), batch_y.cpu().numpy(), loss_value])))
            else:  # return metrics
                eval_res, _ = metric(pred_y.cpu().numpy(), batch_y.cpu().numpy(),
                                     data_loader.dataset.mean, data_loader.dataset.std,
                                     metrics=self.metric_list, spatial_norm=self.spatial_norm, return_log=False)
                eval_res['loss'] = self.criterion(pred_y, batch_y).cpu().numpy()
                for k in eval_res.keys():
                    eval_res[k] = eval_res[k].reshape(1)
                results.append(eval_res)

            if self.args.empty_cache:
                torch.cuda.empty_cache()
            if self.rank == 0:
                prog_bar.update()

        # post gather tensors
        results_all = {}
        for k in results[0].keys():
            results_cat = np.concatenate([batch[k] for batch in results], axis=0)
            # gether tensors by GPU (it's no need to empty cache)
            results_gathered = gather_tensors_batch(results_cat, part_size=min(part_size*8, 16))
            results_strip = np.concatenate(results_gathered, axis=0)[:length]
            results_all[k] = results_strip
        results_all = self._merge_spatial_results(results_all, data_loader.dataset, gather_data)
        return results_all

    def _nondist_forward_collect(self, data_loader, length=None, gather_data=False):
        """Forward and collect predictios.

        Args:
            data_loader: dataloader of evaluation.
            length (int): Expected length of output arrays.
            gather_data (bool): Whether to gather raw predictions and inputs.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        """
        # preparation
        results = []
        prog_bar = ProgressBar(len(data_loader))
        length = len(data_loader.dataset) if length is None else length

        # loop
        for idx, (batch_x, batch_y) in enumerate(data_loader):
            with torch.no_grad():
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self._predict(batch_x, batch_y)

            if gather_data:  # return raw datas
                
                loss_value = self.criterion(pred_y, batch_y).detach().cpu().numpy().reshape(1)
                results.append(dict(zip(['inputs', 'preds', 'trues', 'loss'],
                                        [batch_x.cpu().numpy(), pred_y.cpu().numpy(), batch_y.cpu().numpy(), loss_value])))
            else:  # evaluation-only path when we do not need to store raw tensors
                eval_res, _ = metric(pred_y.cpu().numpy(), batch_y.cpu().numpy(),
                                     data_loader.dataset.mean, data_loader.dataset.std,
                                     metrics=self.metric_list, spatial_norm=self.spatial_norm, return_log=False)
                eval_res['loss'] = self.criterion(pred_y, batch_y).cpu().numpy()
                for k in eval_res.keys():
                    eval_res[k] = eval_res[k].reshape(1)
                results.append(eval_res)

            prog_bar.update()
            if self.args.empty_cache:
                torch.cuda.empty_cache()

        # post gather tensors
        results_all = {}
        for k in results[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in results], axis=0)
        results_all = self._merge_spatial_results(results_all, data_loader.dataset, gather_data)
        return results_all

    def _merge_spatial_results(self, results_all, dataset, gather_data):
        """Reconstruct full-frame samples from spatially split patches.

        Args:
            results_all (dict): Concatenated outputs from the dataloader.
            dataset: Dataset instance that may contain spatial split metadata.
            gather_data (bool): Whether raw predictions were gathered.

        Returns:
            dict: Potentially merged results with full-frame tensors and updated loss/metrics.
        """
        if not gather_data:
            return results_all

        if not getattr(dataset, 'use_space', False) or not getattr(dataset, 'eval_non_overlap', False):
            return results_all

        num_sequences = len(dataset.time_indices)
        sample_indices = getattr(dataset, 'sample_indices', [])
        coords = getattr(dataset, 'space_coords', [(0, 0)])
        if len(sample_indices) == 0 or len(results_all.get('preds', [])) != len(sample_indices):
            return results_all

        space_h = dataset.space_h
        space_w = dataset.space_w
        full_h, full_w = dataset.H, dataset.W

        def _merge_tensor(arr, seq_len):
            merged = np.zeros((num_sequences, seq_len, dataset.C, full_h, full_w), dtype=arr.dtype)
            counts = np.zeros_like(merged, dtype=np.int32)
            for idx, (t_idx, p_idx) in enumerate(sample_indices):
                top, left = coords[p_idx]
                merged[t_idx, :, :, top: top + space_h, left: left + space_w] = arr[idx]
                merged[t_idx, :, :, top: top + space_h, left: left + space_w] += arr[idx]
                counts[t_idx, :, :, top: top + space_h, left: left + space_w] += 1

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

    def vali_one_epoch(self, runner, vali_loader, gather_data=False, **kwargs):
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
        should_gather = gather_data or (getattr(dataset, 'use_space', False) and getattr(dataset, 'eval_non_overlap', False))
        if self.dist and self.world_size > 1:
            results = self._dist_forward_collect(vali_loader, len(dataset), gather_data=should_gather)
        else:
            results = self._nondist_forward_collect(vali_loader, len(dataset), gather_data=should_gather)

        eval_log = ""
        
        if should_gather:
            eval_res, eval_log = metric(results['preds'], results['trues'],
                                        dataset.mean, dataset.std,
                                        metrics=self.metric_list, spatial_norm=self.spatial_norm)
            for k in self.metric_list:
                results[k] = np.array(eval_res[k]).reshape(1)
            results['metrics'] = np.array([eval_res['mae'], eval_res['mse'], eval_res['rmse'], eval_res['mape']])
            results['metric_dict'] = eval_res
        else:
            for k, v in results.items():
                if k != "loss":
                    v = v.mean()
                    eval_str = f"{k}:{v.mean()}" if len(eval_log) == 0 else f", {k}:{v.mean()}"
                    eval_log += eval_str

        return results, eval_log

    def test_one_epoch(self, runner, test_loader, gather_data=True, **kwargs):
        """Evaluate the model with test_loader.

        Args:
            runner: the trainer of methods.
            test_loader: dataloader of testing.

        Returns:
            list(tensor, ...): The list of inputs and predictions.
        """
        self.model.eval()
        dataset = test_loader.dataset
        should_gather = gather_data or (getattr(dataset, 'use_space', False) and getattr(dataset, 'eval_non_overlap', False))

        if self.dist and self.world_size > 1:
            results = self._dist_forward_collect(test_loader, len(dataset), gather_data=should_gather)
        else:
            results = self._nondist_forward_collect(test_loader, len(dataset), gather_data=should_gather)
            
        eval_log = ""
        if should_gather:
            eval_res, eval_log = metric(results['preds'], results['trues'],
                                        dataset.mean, dataset.std,
                                        metrics=self.metric_list, spatial_norm=self.spatial_norm)
            for k in self.metric_list:
                results[k] = np.array(eval_res[k]).reshape(1)
            results['metrics'] = np.array([eval_res['mae'], eval_res['mse'], eval_res['rmse'], eval_res['mape']])
            results['metric_dict'] = eval_res
        else:
            for k, v in results.items():
                if k in {"inputs", "preds", "trues"}:
                    continue
                if k != "loss":
                    v = v.mean()
                    eval_str = f"{k}:{v.mean()}" if len(eval_log) == 0 else f", {k}:{v.mean()}"
                    eval_log += eval_str

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
