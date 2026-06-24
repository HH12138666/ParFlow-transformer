import os.path as osp
import time
import logging
import json
from fvcore.nn import FlopCountAnalysis, flop_count_table

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from openstl.core.training import ExperimentTrainingMixin
from openstl.core.checkpoint import (
    extract_state_dict, load_checkpoint, load_model_state, save_training_checkpoint,
)
from openstl.methods import method_maps
from openstl.utils import (set_seed, print_log, output_namespace, check_dir, collect_env,
                           get_dataset, measure_throughput)


class BaseExperiment(ExperimentTrainingMixin):
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
        # resume training / finetune from pretrained weights
        if self.args.auto_resume:
            self.args.resume_from = osp.join(self.checkpoints_path, 'latest.pth')
        if self.args.resume_from is not None:
            self._load(name=self.args.resume_from)
        elif getattr(self.args, 'finetune_from', None) is not None:
            self._load_pretrained_weights(self.args.finetune_from)

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

        if self.vali_loader is None and getattr(self.args, 'use_val', True):
            self.vali_loader = self.test_loader
        self._max_iters = self._max_epochs * len(self.train_loader)

    def _save(self, name=""):
        if self._rank != 0:
            return
        save_training_checkpoint(
            osp.join(self.checkpoints_path, name + ".pth"),
            self.method.model, self.method.model_optim, self.method.scheduler,
            self._epoch + 1,
        )

    def _resolve_checkpoint_path(self, name):
        if osp.isfile(name):
            return name
        filename = osp.join(self.checkpoints_path, name + ".pth")
        if not osp.isfile(filename):
            raise FileNotFoundError(f"Checkpoint not found: {filename}")
        return filename

    def _load(self, name=""):
        filename = self._resolve_checkpoint_path(name)
        checkpoint = load_checkpoint(filename, self.device)
        self._load_from_state_dict(extract_state_dict(checkpoint))
        if checkpoint.get("epoch") is not None:
            self._epoch = checkpoint["epoch"]
            self.method.model_optim.load_state_dict(checkpoint["optimizer"])
            self.method.scheduler.load_state_dict(checkpoint["scheduler"])
        print_log(f"Loaded checkpoint for resume from: {filename}")

    def _load_from_state_dict(self, state_dict):
        load_model_state(self.method.model, state_dict)

    def _load_pretrained_weights(self, name=""):
        filename = self._resolve_checkpoint_path(name)
        checkpoint = load_checkpoint(filename, self.device)
        self._load_from_state_dict(extract_state_dict(checkpoint))
        self._epoch = 0
        print_log(f"Loaded pretrained weights for finetuning from: {filename}")

    def _load_best_checkpoint(self):
        filename = osp.join(self.path, "checkpoint.pth")
        self._load_from_state_dict(extract_state_dict(load_checkpoint(filename, self.device)))

    def _load_latest_checkpoint(self):
        filename = osp.join(self.checkpoints_path, "latest.pth")
        self._load_from_state_dict(extract_state_dict(load_checkpoint(filename, self.device)))


    def display_method_info(self):
        """Plot the basic infomation of supported methods"""
        T, C, H, W = self.args.in_shape
        # When spatial tiling is enabled, the model operates on cropped windows
        # rather than the full frame size described by ``in_shape``.
        crop_h = getattr(self.args, 'space_h', None)
        crop_w = getattr(self.args, 'space_w', None)
        use_crop = crop_h is not None and crop_w is not None
        dummy_h = crop_h if use_crop else H
        dummy_w = crop_w if use_crop else W
        model_ref = self.method.model.module if hasattr(self.method.model, 'module') else self.method.model

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
