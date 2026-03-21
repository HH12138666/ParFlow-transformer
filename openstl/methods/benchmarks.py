import time

import torch
import torch.nn as nn
from timm.utils import AverageMeter
from tqdm import tqdm

from .base_method import Base_method
from openstl.models.benchmark_baselines import (
    CNNForecastModel,
    ConvLSTMForecastModel,
    PixelLSTMForecastModel,
    PixelRNNForecastModel,
)
from openstl.utils import reduce_tensor


class _BaseBenchmarkMethod(Base_method):
    model_cls = None

    def __init__(self, args, device, steps_per_epoch):
        super().__init__(args, device, steps_per_epoch)
        if self.model_cls is None:
            raise ValueError("model_cls must be set for benchmark methods.")
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        return self.model_cls(**args).to(self.device)

    def _predict(self, batch_x, batch_y=None, **kwargs):
        pred_y = self.model(batch_x)
        aft = int(self.args.aft_seq_length)
        if pred_y.shape[1] < aft:
            raise ValueError(
                f"Model prediction length {pred_y.shape[1]} is shorter than aft_seq_length={aft}."
            )
        if pred_y.shape[1] > aft:
            pred_y = pred_y[:, :aft]
        return pred_y

    def train_one_epoch(self, runner, train_loader, epoch, num_updates, **kwargs):
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader
        end = time.time()

        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            with self.amp_autocast():
                pred_y = self._predict(batch_x, batch_y=batch_y)
                pred_y_loss = self._crop_to_valid_spatial(pred_y)
                batch_y_loss = self._crop_to_valid_spatial(batch_y)
                loss_channels = getattr(self.args, "loss_channels", 10)
                if pred_y_loss.shape[2] < loss_channels or batch_y_loss.shape[2] < loss_channels:
                    raise ValueError(
                        f"Loss expects at least {loss_channels} channels, got "
                        f"{pred_y_loss.shape[2]} (pred) and {batch_y_loss.shape[2]} (true)."
                    )
                loss = self.criterion(
                    pred_y_loss[:, :, :loss_channels, ...],
                    batch_y_loss[:, :, :loss_channels, ...],
                )

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters())
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())
                self.model_optim.step()

            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner._iter += 1

            if self.rank == 0:
                log_buffer = f'train loss: {loss.item():.4f}'
                log_buffer += f' | data time: {data_time_m.avg:.4f}'
                train_pbar.set_description(log_buffer)
            end = time.time()

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m


class CNNMethod(_BaseBenchmarkMethod):
    model_cls = CNNForecastModel


class RNNMethod(_BaseBenchmarkMethod):
    model_cls = PixelRNNForecastModel


class LSTMMethod(_BaseBenchmarkMethod):
    model_cls = PixelLSTMForecastModel


class ConvLSTMMethod(_BaseBenchmarkMethod):
    model_cls = ConvLSTMForecastModel
