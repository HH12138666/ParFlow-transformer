from contextlib import suppress
from functools import partial
from typing import Dict, List, Union

import torch
from timm.utils import NativeScaler
from timm.utils.agc import adaptive_clip_grad

from openstl.core.evaluation import EvaluationRunner
from openstl.core.optim_scheduler import get_optim_scheduler


class Base_method:
    def __init__(self, args, device, steps_per_epoch):
        del steps_per_epoch
        self.args = args
        self.device = device
        self.config = args.__dict__
        self.criterion = None
        self.model_optim = None
        self.scheduler = None
        self.rank, self.world_size = 0, 1
        self.clip_value = self.args.clip_grad
        self.clip_mode = (
            self.args.clip_mode if self.clip_value is not None else None
        )
        self.amp_autocast = suppress
        self.loss_scaler = None
        self._configure_amp()
        self.metric_list = ["mae", "rmse"]
        self.spatial_norm = True
        self.evaluator = EvaluationRunner(self)

    def _configure_amp(self):
        if not self.args.fp16:
            return
        if self.device.type != "cuda":
            raise ValueError("fp16 training requires a CUDA device")
        self.amp_autocast = partial(
            torch.autocast, device_type="cuda", dtype=torch.float16
        )
        self.loss_scaler = NativeScaler()

    def _init_optimizer(self, steps_per_epoch):
        return get_optim_scheduler(
            self.args, self.args.epoch, self.model, steps_per_epoch
        )

    def train_one_epoch(self, runner, train_loader, **kwargs):
        raise NotImplementedError

    def _predict(self, batch_x, batch_y, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _check_eval_channels(pred_y, true_y):
        if pred_y.shape[2] != true_y.shape[2]:
            raise ValueError(
                f"Prediction C={pred_y.shape[2]} does not match target C={true_y.shape[2]}"
            )

    def vali_one_epoch(self, runner, vali_loader, **kwargs):
        del runner, kwargs
        self.model.eval()
        results = self.evaluator.run(vali_loader)
        return results, results["eval_log"]

    def test_one_epoch(self, runner, test_loader, **kwargs):
        del runner, kwargs
        self.model.eval()
        return self.evaluator.run(test_loader)

    def current_lr(self) -> Union[List[float], Dict[str, List[float]]]:
        if isinstance(self.model_optim, torch.optim.Optimizer):
            return [group["lr"] for group in self.model_optim.param_groups]
        if isinstance(self.model_optim, dict):
            return {
                name: [group["lr"] for group in optimizer.param_groups]
                for name, optimizer in self.model_optim.items()
            }
        raise RuntimeError("Learning rate is unavailable without an optimizer")

    def clip_grads(self, params, norm_type=2.0):
        if self.clip_mode is None:
            return
        if self.clip_mode == "norm":
            torch.nn.utils.clip_grad_norm_(params, self.clip_value, norm_type=norm_type)
            return
        if self.clip_mode == "value":
            torch.nn.utils.clip_grad_value_(params, self.clip_value)
            return
        if self.clip_mode == "agc":
            adaptive_clip_grad(params, self.clip_value, norm_type=norm_type)
            return
        raise ValueError(f"Unknown clip mode: {self.clip_mode}")
