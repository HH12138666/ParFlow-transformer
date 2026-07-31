import time

import torch
import torch.nn as nn
from timm.utils import AverageMeter
from tqdm import tqdm

from openstl.models import PredFormer_Model

from .base_method import Base_method


class PredFormer(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        super().__init__(args, device, steps_per_epoch)
        self.model = PredFormer_Model(self.config["model_config"]).to(self.device)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(
            steps_per_epoch
        )
        self.criterion = nn.MSELoss()

    def _predict(self, batch_x, batch_y=None, future_aux=None, **kwargs):
        del batch_y, kwargs
        pre = self.args.pre_seq_length
        aft = self.args.aft_seq_length
        if aft <= pre:
            return self.model(batch_x)[:, :aft]
        if future_aux is None or future_aux.shape[1] < aft:
            raise ValueError(f"Long rollout requires {aft} future auxiliary frames")
        return self._rollout(batch_x, future_aux, aft, pre)

    def _rollout(self, batch_x, future_aux, aft, block_size):
        predictions = []
        current = batch_x
        predicted = 0
        while predicted < aft:
            count = min(block_size, aft - predicted)
            block = self.model(current)[:, :count]
            predictions.append(block)
            if predicted + count >= aft:
                break
            aux = future_aux[:, predicted:predicted + count]
            current = torch.cat([block, aux], dim=2)
            predicted += count
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
        for batch_x, batch_y, future_aux in progress:
            data_time.update(time.time() - end)
            loss = self._train_batch(batch_x, batch_y, future_aux)
            losses.update(loss, batch_x.size(0))
            num_updates += 1
            runner._iter += 1
            self._update_progress(progress, loss, data_time.avg)
            end = time.time()
        if hasattr(self.model_optim, "sync_lookahead"):
            self.model_optim.sync_lookahead()
        return num_updates, losses

    def _train_batch(self, batch_x, batch_y, future_aux):
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        future_aux = future_aux.to(self.device)
        self.model_optim.zero_grad()
        if self.args.aft_seq_length <= self.args.pre_seq_length:
            loss = self._train_direct(batch_x, batch_y)
        else:
            loss = self._train_rollout(batch_x, batch_y, future_aux)
        self._finish_optimizer_step()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        if not self.by_epoch:
            self.scheduler.step()
        return loss

    def _train_direct(self, batch_x, batch_y):
        with self.amp_autocast():
            pred_y = self._predict(batch_x)
            self._check_eval_channels(pred_y, batch_y)
            loss = self.criterion(pred_y, batch_y)
        self._backward(loss)
        return loss.item()

    def _train_rollout(self, batch_x, batch_y, future_aux):
        aft = self.args.aft_seq_length
        block_size = self.args.pre_seq_length
        current = batch_x
        total_loss = 0.0
        for start in range(0, aft, block_size):
            count = min(block_size, aft - start)
            with self.amp_autocast():
                prediction = self.model(current)[:, :count]
                target = batch_y[:, start:start + count]
                block_loss = self.criterion(prediction, target) * (count / aft)
            self._backward(block_loss)
            total_loss += block_loss.detach().item()
            if start + count < aft:
                aux = future_aux[:, start:start + count]
                current = torch.cat([prediction.detach(), aux], dim=2)
        return total_loss

    def _backward(self, loss):
        if not torch.isfinite(loss):
            raise ValueError("Training loss is NaN or Inf")
        if self.loss_scaler is not None:
            self.loss_scaler._scaler.scale(loss).backward()
            return
        loss.backward()

    def _finish_optimizer_step(self):
        if self.loss_scaler is not None:
            scaler = self.loss_scaler._scaler
            if self.args.clip_grad is not None:
                scaler.unscale_(self.model_optim)
                self.clip_grads(self.model.parameters())
            scaler.step(self.model_optim)
            scaler.update()
            return
        self.clip_grads(self.model.parameters())
        self.model_optim.step()

    def _update_progress(self, progress, loss, data_time):
        if self.rank != 0:
            return
        progress.set_description(
            f"train loss: {loss:.4f} | data time: {data_time:.4f}"
        )
