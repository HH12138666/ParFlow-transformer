import os
import os.path as osp
import time
import logging
import json
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from openstl.core import Recorder
from openstl.methods import method_maps
from openstl.utils import (set_seed, print_log, output_namespace, check_dir, collect_env,
                           get_dataset, measure_throughput, weights_to_cpu, barrier)


class BaseExperiment(object):
    """The basic class of PyTorch training and evaluation."""

    def __init__(self, args, dataloaders=None):
        """Initialize experiments (non-dist as an example)"""
        self.args = args
        self.config = self.args.__dict__
        self.device = self.args.device
        self.method = None
        self.args.method = self.args.method.lower()
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = self.config['epoch']
        self._max_iters = None
        self._early_stop = self.args.early_stop_epoch
        self._distributed = getattr(self.args, 'distributed', False)
        self._rank = getattr(self.args, 'rank', 0)
        self._world_size = getattr(self.args, 'world_size', 1)
        self._preparation(dataloaders)
        print_log(output_namespace(self.args))
        if (not self.args.no_display_method_info) and self._rank == 0:
            self.display_method_info()

    def _acquire_device(self):
        """Setup devices"""
        if self.args.use_gpu:
            self._use_gpu = True
            if self._distributed:
                device = torch.device(f'cuda:{self.args.local_rank}')
            else:
                device = torch.device('cuda:0')
            print_log(f'Use GPU: {device}')
        else:
            self._use_gpu = False
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self, dataloaders=None):
        """Preparation of environment and basic experiment setups"""
        self.device = self._acquire_device()
        if self._early_stop < 0:
            self._early_stop = self._max_epochs * 2

        # log and checkpoint
        base_dir = self.args.res_dir if self.args.res_dir is not None else 'work_dirs'
        self.path = osp.join(base_dir, self.args.ex_name if not self.args.ex_name.startswith(self.args.res_dir) \
            else self.args.ex_name.split(self.args.res_dir+'/')[-1])
        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.path)
        check_dir(self.checkpoints_path)

        if self._rank == 0:
            sv_param = osp.join(self.path, 'model_param.json')
            with open(sv_param, 'w') as file_obj:
                json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        prefix = 'train' if (not self.args.test and not self.args.inference) else 'test'
        if self._rank == 0:
            logging.basicConfig(level=logging.INFO,
                                filename=osp.join(self.path, '{}_{}.log'.format(prefix, timestamp)),
                                filemode='a', format='%(asctime)s - %(message)s')
        else:
            logging.basicConfig(level=logging.ERROR)

        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        print_log('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

        # set random seeds
        set_seed(self.args.seed, deterministic=self.args.deterministic)

        # prepare data
        self._get_data(dataloaders)
        # build the method
        self._build_method()
        # resume traing
        if self.args.auto_resume:
            self.args.resume_from = osp.join(self.checkpoints_path, 'latest.pth')
        if self.args.resume_from is not None:
            self._load(name=self.args.resume_from)

    def _build_method(self):
        self.steps_per_epoch = len(self.train_loader)
        self.method = method_maps[self.args.method](self.args, self.device, self.steps_per_epoch)
        self.method.rank = self._rank
        self.method.world_size = self._world_size
        if self._distributed:
            self.method.model = DDP(
                self.method.model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=getattr(self.args, 'find_unused_parameters', False),
                broadcast_buffers=getattr(self.args, 'broadcast_buffers', False),
            )
        self.method.model.eval()

    def _get_data(self, dataloaders=None):
        """Prepare datasets and dataloaders"""
        if dataloaders is None:
            self.train_loader, self.vali_loader, self.test_loader = \
                get_dataset(self.args.dataname, self.config)
        else:
            self.train_loader, self.vali_loader, self.test_loader = dataloaders

        if self.vali_loader is None:
            self.vali_loader = self.test_loader
        self._max_iters = self._max_epochs * len(self.train_loader)

    def _save(self, name=''):
        """Saving models and meta data to checkpoints"""
        if self._rank != 0:
            return
        model_to_save = self.method.model.module if hasattr(self.method.model, 'module') else self.method.model
        checkpoint = {
            'epoch': self._epoch + 1,
            'optimizer': self.method.model_optim.state_dict(),
            'state_dict': weights_to_cpu(model_to_save.state_dict()),
            'scheduler': self.method.scheduler.state_dict()}
        torch.save(checkpoint, osp.join(self.checkpoints_path, name + '.pth'))

    def _load(self, name=''):
        """Loading models from the checkpoint"""
        filename = name if osp.isfile(name) else osp.join(self.checkpoints_path, name + '.pth')
        try:
            checkpoint = torch.load(filename, map_location=self.device)
        except:
            return
        # OrderedDict is a subclass of dict
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
        self._load_from_state_dict(checkpoint['state_dict'])
        if checkpoint.get('epoch', None) is not None:
            self._epoch = checkpoint['epoch']
            self.method.model_optim.load_state_dict(checkpoint['optimizer'])
            self.method.scheduler.load_state_dict(checkpoint['scheduler'])

    def _load_from_state_dict(self, state_dict):
        model_to_load = self.method.model.module if hasattr(self.method.model, 'module') else self.method.model
        model_to_load.load_state_dict(state_dict)

    def _load_best_checkpoint(self):
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        self._load_from_state_dict(torch.load(best_model_path, map_location=self.device))

    def display_method_info(self):
        """Plot the basic infomation of supported methods"""
        T, C, H, W = self.args.in_shape
        # When spatial tiling is enabled, the model operates on cropped windows
        # rather than the full frame size described by ``in_shape``. Use the
        # configured spatial crop to build the dummy tensor for FLOP/FPS
        # reporting so that positional embeddings match the actual model input
        # shape, while keeping ``in_shape`` available for full-frame metadata
        # (e.g., stitching tiles back together during evaluation).
        crop_h = getattr(self.args, 'space_h', None)
        crop_w = getattr(self.args, 'space_w', None)
        use_crop = crop_h is not None and crop_w is not None
        dummy_h = crop_h if use_crop else H
        dummy_w = crop_w if use_crop else W
        model_ref = self.method.model.module if hasattr(self.method.model, 'module') else self.method.model
        dummy_h = getattr(model_ref, 'image_height', dummy_h)
        dummy_w = getattr(model_ref, 'image_width', dummy_w)

        if self.args.method in ['predformer']:
            input_dummy = torch.ones(1, self.args.pre_seq_length, C, dummy_h, dummy_w).to(self.device)
        else:
            raise ValueError(f'Invalid method name {self.args.method}')

        dash_line = '-' * 80 + '\n'
        info = model_ref.__repr__()
        flops = FlopCountAnalysis(model_ref, input_dummy)
        flops = flop_count_table(flops)
        if self.args.fps:
            fps = measure_throughput(model_ref, input_dummy)
            fps = 'Throughputs of {}: {:.3f}\n'.format(self.args.method, fps)
        else:
            fps = ''
        print_log('Model info:\n' + info+'\n' + flops+'\n' + fps + dash_line)


    def train(self):
        """Training loops of STL methods"""
        recorder = Recorder(verbose=True, early_stop_time=min(self._max_epochs // 10, 10))
        num_updates = self._epoch * self.steps_per_epoch
        early_stop = False
        for epoch in range(self._epoch, self._max_epochs):
            if self._distributed and hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            num_updates, loss_mean = self.method.train_one_epoch(
                self, self.train_loader, epoch, num_updates
            )

            self._epoch = epoch
            if epoch % self.args.log_step == 0:
                if self._rank == 0:
                    cur_lr = self.method.current_lr()
                    cur_lr = sum(cur_lr) / len(cur_lr)
                    with torch.no_grad():
                        vali_loss = self.vali()

                    print_log('Epoch: {0}, Steps: {1} | Lr: {2:.7f} | Train Loss: {3:.7f} | Vali Loss: {4:.7f}\n'.format(
                        epoch + 1, len(self.train_loader), cur_lr, loss_mean.avg, vali_loss))
                    early_stop = recorder(vali_loss, self.method.model, self.path)
                    self._save(name='latest')

                if self._distributed:
                    stop_tensor = torch.tensor(int(early_stop), device=self.device)
                    dist.broadcast(stop_tensor, src=0)
                    early_stop = bool(stop_tensor.item())

                    
            if self._use_gpu and self.args.empty_cache:
                torch.cuda.empty_cache()
            if epoch > self._early_stop and early_stop:  # early stop training
                #print_log('Early stop training at f{} epoch'.format(epoch))
                print_log('Early stop training at {} epoch'.format(epoch + 1))
                break
            
        if self._distributed:
            barrier()
        if not check_dir(self.path):  # exit training when work_dir is removed
            assert False and "Exit training because work_dir is removed"
        if self._distributed and hasattr(self.method.model, 'module'):
            self.method.model = self.method.model.module
        self._load_best_checkpoint()
        time.sleep(1)  # wait for asynchronous loggers to flush

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

        return results['loss'].mean()

    def test(self):
        """A testing loop of STL methods"""
        if self.args.test:
            self._load_best_checkpoint()

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

        return eval_res['mse']
