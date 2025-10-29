import time
import torch
import torch.nn as nn
from tqdm import tqdm
from timm.utils import AverageMeter
import torch.nn.functional as F

from openstl.models import PredFormer_Model
from openstl.utils import reduce_tensor
from .base_method import Base_method

class PredFormer(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()
        self._loss_std_cache = {} #修改 cache for loss std of each variable
        
    def _build_model(self, args):
        return PredFormer_Model(**args).to(self.device)
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
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y
    
    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        #修改 Get dataset mean and std for loss denormalization if needed
        dataset_std = getattr(train_loader.dataset, 'std_t', None)

        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')

            with self.amp_autocast():
                pred_y = self._predict(batch_x)
                loss = self.criterion(pred_y, batch_y)
                #修改 compute physical loss for denormalized data if needed
                log_loss = self._compute_physical_loss(pred_y, batch_y, dataset_std)
            log_loss_detached = log_loss.detach()

            if not self.dist:
                #修改 Update loss meter
                losses_m.update(log_loss_detached.item(), batch_x.size(0))

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
                #修改 Update loss meter for distributed training
                losses_m.update(reduce_tensor(log_loss_detached), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                # 修改 Log training information
                log_buffer = 'train loss: {:.4f}'.format(log_loss_detached.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        

        return num_updates, losses_m, eta
    
# 修改 loss std cache getter
    def _get_loss_std(self, std_tensor, reference):
        if std_tensor is None:
            return None

        key = (id(std_tensor), reference.device, reference.dim())
        cached = self._loss_std_cache.get(key)
        if cached is not None:
            return cached

        std = std_tensor.to(reference.device)
        while std.dim() < reference.dim():
            std = std.unsqueeze(0)
        self._loss_std_cache[key] = std
        return std

    def _compute_physical_loss(self, pred_y, batch_y, std_tensor):
        std = self._get_loss_std(std_tensor, pred_y)
        if std is None:
            return self.criterion(pred_y, batch_y)
        return self.criterion(pred_y * std, batch_y * std)

    def _compute_logging_loss(self, pred_y, batch_y, dataset):
        std_tensor = getattr(dataset, 'std_t', None)
        return self._compute_physical_loss(pred_y, batch_y, std_tensor)