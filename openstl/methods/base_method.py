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
        # 训练/验证阶段默认只计算 MAE 和 RMSE，减轻 full-frame 评估开销
        self.metric_list, self.spatial_norm = ['mae', 'rmse'], True

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

    def _nondist_forward_collect(self, data_loader, length=None):
        """Forward non-spatial evaluation samples and collect arrays."""
        dataset = data_loader.dataset
        if getattr(dataset, 'use_space', False):
            raise RuntimeError('Spatial evaluation must use streaming full-frame metrics.')

        results = []
        prog_bar = ProgressBar(len(data_loader))
        for batch_x, batch_y in data_loader:
            with torch.inference_mode():
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                pred_y = self._predict(batch_x, batch_y)

            self._check_eval_channels(pred_y, batch_y)
            loss_value = self.criterion(pred_y, batch_y).detach().cpu().numpy().reshape(1)
            results.append({
                'preds': pred_y.cpu().numpy(),
                'trues': batch_y.cpu().numpy(),
                'loss': loss_value,
            })
            prog_bar.update()
            if self.args.empty_cache:
                torch.cuda.empty_cache()

        if not results:
            return self._empty_array_results()
        return {key: np.concatenate([item[key] for item in results], axis=0)
                for key in results[0].keys()}

    def _check_eval_channels(self, pred_y, true_y):
        if pred_y.shape[2] == true_y.shape[2]:
            return
        raise ValueError(
            f"Loss expects matching output channels, got "
            f"{pred_y.shape[2]} (pred) and {true_y.shape[2]} (true)."
        )

    def _empty_array_results(self):
        return {
            'preds': np.empty((0,), dtype=np.float32),
            'trues': np.empty((0,), dtype=np.float32),
            'loss': np.empty((0,), dtype=np.float32),
        }

    def _stream_spatial_metric_collect(self, data_loader):
        """Evaluate spatial windows by finalizing one full-frame time sample at a time."""
        dataset = data_loader.dataset
        sample_count = len(getattr(dataset, 'sample_indices', []))
        if sample_count == 0:
            return self._make_metric_results({}, [], '')

        state = self._new_spatial_stream_state(dataset)
        loss_values = []
        offset = 0
        prog_bar = ProgressBar(len(data_loader))
        for batch_x, batch_y in data_loader:
            pred_y, batch_y = self._eval_batch(batch_x, batch_y)
            loss_values.append(self.criterion(pred_y, batch_y).detach().cpu().numpy().reshape(1))
            offset = self._consume_spatial_batch(dataset, state, pred_y, batch_y, offset)
            prog_bar.update()

        if offset != sample_count:
            raise ValueError(f'Spatial stream consumed {offset} samples, expected {sample_count}.')
        self._finalize_active_spatial_slot(state)
        eval_res = self._compute_stream_metrics(state['metric_sums'])
        return self._make_metric_results(eval_res, loss_values, self._format_eval_log(eval_res))

    def _eval_batch(self, batch_x, batch_y):
        with torch.inference_mode():
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            pred_y = self._predict(batch_x, batch_y)
        self._check_eval_channels(pred_y, batch_y)
        return pred_y, batch_y

    def _new_spatial_stream_state(self, dataset):
        return {
            'active_slot': None,
            'finalized_slots': set(),
            'pred_buffer': None,
            'true_buffer': None,
            'counts': None,
            'dataset': dataset,
            'metric_sums': {'abs': 0.0, 'sq': 0.0, 'denom': 0.0},
        }

    def _consume_spatial_batch(self, dataset, state, pred_y, true_y, offset):
        pred_np = pred_y.cpu().numpy()
        true_np = true_y.cpu().numpy()
        batch_n = pred_np.shape[0]
        end = offset + batch_n
        sample_count = len(getattr(dataset, 'sample_indices', []))
        if end > sample_count:
            raise ValueError(f'Batch exceeds spatial sample plan: end={end}, sample_count={sample_count}')

        slots = dataset.merge_slots[offset:end]
        tops = dataset.merge_tops[offset:end]
        lefts = dataset.merge_lefts[offset:end]
        for i in range(batch_n):
            self._consume_spatial_patch(state, int(slots[i]), int(tops[i]), int(lefts[i]), pred_np[i], true_np[i])
        return end

    def _consume_spatial_patch(self, state, slot, top, left, pred_patch, true_patch):
        if state['active_slot'] is None:
            self._start_spatial_slot(state, slot, pred_patch)
        elif slot != state['active_slot']:
            self._finalize_active_spatial_slot(state)
            self._start_spatial_slot(state, slot, pred_patch)
        self._add_spatial_patch(state, top, left, pred_patch, true_patch)

    def _start_spatial_slot(self, state, slot, pred_patch):
        if slot in state['finalized_slots']:
            raise ValueError(f'Spatial slot {slot} is not contiguous in evaluation loader.')
        dataset = state['dataset']
        state['active_slot'] = slot
        buffer_shape = (pred_patch.shape[0], pred_patch.shape[1], dataset.H, dataset.W)
        state['pred_buffer'] = np.zeros(buffer_shape, dtype=pred_patch.dtype)
        state['true_buffer'] = np.zeros(buffer_shape, dtype=pred_patch.dtype)
        state['counts'] = np.zeros((dataset.H, dataset.W), dtype=np.int32)

    def _add_spatial_patch(self, state, top, left, pred_patch, true_patch):
        dataset = state['dataset']
        bottom = top + dataset.space_h
        right = left + dataset.space_w
        state['pred_buffer'][:, :, top:bottom, left:right] += pred_patch
        state['true_buffer'][:, :, top:bottom, left:right] += true_patch
        state['counts'][top:bottom, left:right] += 1

    def _finalize_active_spatial_slot(self, state):
        if state['active_slot'] is None:
            return
        counts = state['counts']
        if np.any(counts <= 0):
            raise ValueError(f"Spatial slot {state['active_slot']} has uncovered pixels.")
        counts_4d = counts[None, None, :, :]
        pred_full = state['pred_buffer'] / counts_4d
        true_full = state['true_buffer'] / counts_4d
        self._update_stream_metric_sums(state['metric_sums'], pred_full, true_full, state['dataset'])
        state['finalized_slots'].add(state['active_slot'])
        state['active_slot'] = None
        state['pred_buffer'] = None
        state['true_buffer'] = None
        state['counts'] = None

    def _update_stream_metric_sums(self, sums, pred, true, dataset):
        diff = pred.astype(np.float64) - true.astype(np.float64)
        std = self._output_std(dataset, diff.shape[1])
        if std is not None:
            diff *= std
        sums['abs'] += np.abs(diff).sum(dtype=np.float64)
        sums['sq'] += np.square(diff).sum(dtype=np.float64)
        sums['denom'] += diff.size if self.spatial_norm else diff.shape[0]

    def _output_std(self, dataset, channels):
        std = getattr(dataset, 'std', None)
        if std is None:
            return None
        std = np.asarray(std, dtype=np.float64)
        if channels > std.shape[0]:
            raise ValueError(f'Output channels {channels} exceed std channels {std.shape[0]}.')
        return std[:channels].reshape(1, channels, 1, 1)

    def _compute_stream_metrics(self, sums):
        if sums['denom'] <= 0:
            raise ValueError('Cannot compute metrics with zero denominator.')
        eval_res = {}
        invalid = set(self.metric_list) - {'mae', 'mse', 'rmse'}
        if invalid:
            raise ValueError(f'metric {invalid} is not supported.')
        if 'mse' in self.metric_list:
            eval_res['mse'] = sums['sq'] / sums['denom']
        if 'mae' in self.metric_list:
            eval_res['mae'] = sums['abs'] / sums['denom']
        if 'rmse' in self.metric_list:
            eval_res['rmse'] = np.sqrt(sums['sq'] / sums['denom'])
        return eval_res

    def _format_eval_log(self, eval_res):
        return ', '.join(f'{key}:{eval_res[key]}' for key in eval_res)

    def _make_metric_results(self, eval_res, loss_values, eval_log):
        results = {'metric_dict': eval_res, 'eval_log': eval_log}
        results['loss'] = np.concatenate(loss_values, axis=0) if loss_values else np.empty((0,), dtype=np.float32)
        for key in self.metric_list:
            if key in eval_res:
                results[key] = np.array(eval_res[key]).reshape(1)
        results['metrics'] = np.array([eval_res[k] for k in self.metric_list if k in eval_res], dtype=np.float32)
        return results

    def _evaluate_collected_results(self, results, dataset):
        preds_eval = results['preds']
        trues_eval = results['trues']
        metric_tasks = len(self.metric_list)
        if dataset.mean is not None and dataset.std is not None:
            metric_tasks += 1
        metric_bar = ProgressBar(metric_tasks)

        def _metric_progress():
            metric_bar.update()

        eval_res, eval_log = metric(
            preds_eval, trues_eval, dataset.mean, dataset.std,
            metrics=self.metric_list, spatial_norm=self.spatial_norm,
            progress_hook=_metric_progress,
        )
        metric_bar.file.write('\n')
        return self._make_metric_results(eval_res, [results['loss']], eval_log)

    def vali_one_epoch(self, runner, vali_loader, **kwargs):
        """Evaluate the model with val_loader."""
        self.model.eval()
        dataset = vali_loader.dataset
        if getattr(dataset, 'use_space', False):
            results = self._stream_spatial_metric_collect(vali_loader)
            return results, results['eval_log']
        results = self._nondist_forward_collect(vali_loader, len(dataset))
        results = self._evaluate_collected_results(results, dataset)
        return results, results['eval_log']

    def test_one_epoch(self, runner, test_loader, **kwargs):
        """Evaluate the model with test_loader."""
        self.model.eval()
        dataset = test_loader.dataset
        if getattr(dataset, 'use_space', False):
            return self._stream_spatial_metric_collect(test_loader)
        results = self._nondist_forward_collect(test_loader, len(dataset))
        return self._evaluate_collected_results(results, dataset)

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
