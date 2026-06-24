import time

import torch
import torch.nn as nn
from timm.utils import AverageMeter
from tqdm import tqdm

from openstl.models import PredFormer_Model
from openstl.models.rollout import merge_pred_with_aux

from .base_method import Base_method


class PredFormer(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        super().__init__(args, device, steps_per_epoch)
        self.model = PredFormer_Model(self.config["model_config"]).to(self.device)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(
            steps_per_epoch
        )
        self.criterion = nn.MSELoss()

    def _predict(self, batch_x, batch_y=None, **kwargs):
        del batch_y, kwargs
        pre = self.args.pre_seq_length
        aft = self.args.aft_seq_length
        if aft <= pre:
            return self.model(batch_x)[:, :aft]
        return self._rollout(batch_x, aft, pre)

    def _rollout(self, batch_x, aft, block_size):
        predictions = []
        current = batch_x
        while sum(block.shape[1] for block in predictions) < aft:
            block = self.model(current)
            remaining = aft - sum(item.shape[1] for item in predictions)
            predictions.append(block[:, :remaining])
            if remaining <= block_size:
                break
            current = merge_pred_with_aux(
                block,
                current,
                self.model.input_channels,
                self.model.out_channels,
            )
        return torch.cat(predictions, dim=1)

    def train_one_epoch(self, runner, train_loader, epoch, num_updates, **kwargs):
        del kwargs
        data_time = AverageMeter()
        losses = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        progress = tqdm(train_loader) if self.rank == 0 else train_loader
        end = time.time()
        for batch_x, batch_y in progress:
            data_time.update(time.time() - end)
            loss = self._train_batch(batch_x, batch_y)
            losses.update(loss, batch_x.size(0))
            num_updates += 1
            runner._iter += 1
            self._update_progress(progress, loss, data_time.avg)
            end = time.time()
        if hasattr(self.model_optim, "sync_lookahead"):
            self.model_optim.sync_lookahead()
        return num_updates, losses

    def _train_batch(self, batch_x, batch_y):
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        self.model_optim.zero_grad()
        with self.amp_autocast():
            pred_y = self._predict(batch_x)
            self._check_eval_channels(pred_y, batch_y)
            loss = self.criterion(pred_y, batch_y)
        self._step_optimizer(loss)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        if not self.by_epoch:
            self.scheduler.step()
        return loss.item()

    def _step_optimizer(self, loss):
        if not torch.isfinite(loss):
            raise ValueError("Training loss is NaN or Inf")
        if self.loss_scaler is not None:
            self.loss_scaler(
                loss,
                self.model_optim,
                clip_grad=self.args.clip_grad,
                clip_mode=self.args.clip_mode,
                parameters=self.model.parameters(),
            )
            return
        loss.backward()
        self.clip_grads(self.model.parameters())
        self.model_optim.step()

    def _update_progress(self, progress, loss, data_time):
        if self.rank != 0:
            return
        progress.set_description(
            f"train loss: {loss:.4f} | data time: {data_time:.4f}"
        )
