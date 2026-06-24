import os
import time

import numpy as np
import torch
import torch.distributed as dist

from openstl.core.recorder import Recorder
from openstl.utils import barrier, print_log


class ExperimentTrainingMixin:
    def _after_training(self, load_best=False):
        if self._distributed:
            barrier()
        if not os.path.isdir(self.path):
            raise FileNotFoundError(f"Work directory was removed: {self.path}")
        if self._distributed and hasattr(self.method.model, 'module'):
            self.method.model = self.method.model.module
        if load_best:
            self._load_best_checkpoint()
        time.sleep(1)

    def _current_lr_mean(self):
        cur_lr = self.method.current_lr()
        return sum(cur_lr) / len(cur_lr)

    def _train_with_val(self, num_updates):
        recorder = Recorder(
            verbose=True,
            early_stop_time=min(self._max_epochs // 10, 10),
            monitor_name='rmse',
        )
        early_stop = False
        for epoch in range(self._epoch, self._max_epochs):
            num_updates, loss_mean = self._run_train_epoch(epoch, num_updates)
            if epoch % self.args.log_step == 0:
                early_stop = self._run_val_logging(epoch, loss_mean, recorder)
            if self._use_gpu and self.args.empty_cache:
                torch.cuda.empty_cache()
            if epoch > self._early_stop and early_stop:
                print_log('Early stop training at {} epoch'.format(epoch + 1))
                break
        self._after_training(load_best=True)

    def _train_fixed_epoch(self, num_updates):
        save_interval = int(getattr(self.args, 'save_interval', 1))
        test_interval = int(getattr(self.args, 'test_interval', 0))
        if save_interval <= 0:
            raise ValueError(f'save_interval must be positive, got {save_interval}')
        if test_interval < 0:
            raise ValueError(f'test_interval must be >= 0, got {test_interval}')
        for epoch in range(self._epoch, self._max_epochs):
            num_updates, loss_mean = self._run_train_epoch(epoch, num_updates)
            if epoch % self.args.log_step == 0 and self._rank == 0:
                self._log_train_only_epoch(epoch, loss_mean)
            if (epoch + 1) % save_interval == 0:
                self._save(name=f'epoch_{epoch + 1:04d}')
                self._save(name='latest')
            if test_interval > 0 and (epoch + 1) % test_interval == 0 and self._rank == 0:
                print_log(f'test_monitoring at epoch {epoch + 1}: not used for checkpoint selection')
                with torch.no_grad():
                    self.test()
            if self._use_gpu and self.args.empty_cache:
                torch.cuda.empty_cache()
        self._save(name='latest')
        self._after_training(load_best=False)

    def _run_train_epoch(self, epoch, num_updates):
        if self._distributed and hasattr(self.train_loader, 'sampler'):
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
        num_updates, loss_mean = self.method.train_one_epoch(
            self, self.train_loader, epoch, num_updates
        )
        self._epoch = epoch
        return num_updates, loss_mean

    def _run_val_logging(self, epoch, loss_mean, recorder):
        early_stop = False
        if self._rank == 0:
            with torch.no_grad():
                vali_stats = self.vali()
            self._log_val_epoch(epoch, loss_mean, vali_stats)
            early_stop = recorder(vali_stats['rmse'], self.method.model, self.path)
            self._save(name='latest')
        if self._distributed:
            stop_tensor = torch.tensor(int(early_stop), device=self.device)
            dist.broadcast(stop_tensor, src=0)
            early_stop = bool(stop_tensor.item())
        return early_stop

    def _log_val_epoch(self, epoch, loss_mean, vali_stats):
        print_log(
            'Epoch: {0}, Steps: {1} | Lr: {2:.7f} | Train Loss: {3:.7f} | '
            'Vali Loss: {4:.7f} | Vali RMSE: {5:.7f}\n'.format(
                epoch + 1,
                len(self.train_loader),
                self._current_lr_mean(),
                loss_mean.avg,
                vali_stats['loss'],
                vali_stats['rmse'],
            )
        )

    def _log_train_only_epoch(self, epoch, loss_mean):
        print_log(
            'Epoch: {0}, Steps: {1} | Lr: {2:.7f} | Train Loss: {3:.7f} | '
            'Val disabled: fixed-epoch training, latest checkpoint will be reported\n'.format(
                epoch + 1,
                len(self.train_loader),
                self._current_lr_mean(),
                loss_mean.avg,
            )
        )

    def train(self):
        """Training loops of STL methods."""
        num_updates = self._epoch * self.steps_per_epoch
        if getattr(self.args, 'use_val', True):
            print_log('Training mode: validation + early stopping/checkpoint selection')
            self._train_with_val(num_updates)
            return
        print_log('Training mode: fixed epoch, validation disabled, latest checkpoint reported')
        self._train_fixed_epoch(num_updates)

    def vali(self):
        """A validation loop during training"""
        t0 = time.time()
        results, eval_log = self.method.vali_one_epoch(self, self.vali_loader)
        t1 = time.time()

        eval_res = results.get('metric_dict', None)
        if eval_res is None:
            raise RuntimeError("Validation metric_dict is missing. Please check vali_one_epoch output.")

        val_msg = f"val_timing\tforward_collect={t1 - t0:.2f}s"
        print_log(val_msg)
        results['metrics'] = np.array([eval_res[k] for k in self.method.metric_list], dtype=np.float32)
        
        print_log('val\t '+eval_log)

        return {
            'loss': float(results['loss'].mean()),
            'rmse': float(eval_res['rmse']),
        }

    def test(self):
        """A testing loop of STL methods"""
        if self.args.test:
            if getattr(self.args, 'use_val', True):
                self._load_best_checkpoint()
            else:
                self._load_latest_checkpoint()

        t0 = time.time()
        results = self.method.test_one_epoch(self, self.test_loader)
        t1 = time.time()

        eval_res = results.get('metric_dict', None)
        eval_log = results.get('eval_log', '')
        if eval_res is None:
            raise RuntimeError("Test metric_dict is missing. Please check test_one_epoch output.")

        print_log(f"test_timing\tforward_collect={t1 - t0:.2f}s")
        print_log(eval_log)
        # Saving test outputs is intentionally disabled.

        if 'rmse' not in eval_res:
            raise KeyError(
                "Test metric_dict must contain 'rmse' because ParFlow metrics are configured as ['mae', 'rmse']."
            )
        return eval_res['rmse']
