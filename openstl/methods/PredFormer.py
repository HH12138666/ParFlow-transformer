import time
import torch
import torch.nn as nn
from tqdm import tqdm
from timm.utils import AverageMeter
import torch.nn.functional as F

from openstl.models import PredFormer_Model
from .base_method import Base_method

class PredFormer(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()
        
    def _build_model(self, args):
        return PredFormer_Model(**args).to(self.device)
    
    def _merge_pred_with_aux(self, pred, prev_seq):
        """Pad predicted channels with auxiliary input channels for autoregressive rollout."""
        if not hasattr(self.model, "in_channels") or not hasattr(self.model, "out_channels"):
            return pred
        in_ch = self.model.in_channels
        out_ch = self.model.out_channels
        # Prefer full input channel count (before static projection) when available.
        input_ch = getattr(self.model, "input_channels", in_ch)
        if input_ch == out_ch:
            return pred
        if prev_seq.shape[2] < input_ch:
            raise ValueError(f"Input has {prev_seq.shape[2]} channels, expected {input_ch}")
        if out_ch > input_ch:
            raise ValueError(f"Output channels {out_ch} cannot exceed input channels {input_ch}")
        aux = prev_seq[:, :, out_ch:input_ch, :, :]
        return torch.cat([pred, aux], dim=2)

    def _replace_evap_with_true(self, pred_block, batch_y, step_idx):
        if not getattr(self.args, "use_true_evap", False):
            return pred_block
        if batch_y is None:
            raise ValueError("use_true_evap requires batch_y for alignment.")
        loss_channels = getattr(self.args, "loss_channels", None)
        if loss_channels is None or pred_block.shape[2] <= loss_channels:
            return pred_block
        start = step_idx * self.args.pre_seq_length
        end = start + self.args.pre_seq_length
        if batch_y.shape[1] < end:
            raise ValueError(
                f"batch_y length {batch_y.shape[1]} is shorter than required {end} steps."
            )
        if batch_y.shape[2] < pred_block.shape[2]:
            raise ValueError(
                f"batch_y channels {batch_y.shape[2]} are fewer than pred channels {pred_block.shape[2]}."
            )
        out = pred_block.clone()
        out[:, :, loss_channels:pred_block.shape[2], ...] = batch_y[
            :, start:end, loss_channels:pred_block.shape[2], ...
        ]
        return out

    def _predict(self, batch_x, batch_y=None, **kwargs):
        """Forward the model"""
        if self.args.aft_seq_length == self.args.pre_seq_length:
            pred_y = self.model(batch_x)
        elif self.args.aft_seq_length < self.args.pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :self.args.aft_seq_length]
        elif self.args.aft_seq_length > self.args.pre_seq_length:
            pred_y = []
            d = self.args.aft_seq_length // self.args.pre_seq_length
            m = self.args.aft_seq_length % self.args.pre_seq_length
            
            cur_seq = batch_x.clone()
            for step_idx in range(d):
                pred_block = self.model(cur_seq)
                pred_y.append(pred_block)
                next_block = self._replace_evap_with_true(pred_block, batch_y, step_idx)
                cur_seq = self._merge_pred_with_aux(next_block, cur_seq)

            if m != 0:
                pred_block = self.model(cur_seq)
                pred_y.append(pred_block[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y
    
    def train_one_epoch(self, runner, train_loader, epoch, num_updates, **kwargs):
        """Train the model with train_loader."""
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

            if not self.by_epoch:
                self.scheduler.step()
            runner._iter += 1

            if self.rank == 0:

                log_buffer = 'train loss: {:.4f}'.format(   loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        

        return num_updates, losses_m
