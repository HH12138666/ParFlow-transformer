import os
import os.path as osp
import time
import logging
import json
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table

import torch
import torch.distributed as dist
from openstl.core import metric, Recorder
from openstl.methods import method_maps
from openstl.utils import (set_seed, print_log, output_namespace, check_dir, collect_env,
                           init_dist, init_random_seed,
                           get_dataset, get_dist_info, measure_throughput, weights_to_cpu)

try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False


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
        self._rank = 0
        self._world_size = 1
        self._dist = self.args.dist
        self._early_stop = self.args.early_stop_epoch
        self._preparation(dataloaders)
        if self._rank == 0:
            print_log(output_namespace(self.args))
            if not self.args.no_display_method_info:
                self.display_method_info()

    def _acquire_device(self):
        """Setup devices"""
        if self.args.use_gpu:
            self._use_gpu = True
            if self.args.dist:
                device = f'cuda:{self._rank}'
                torch.cuda.set_device(self._rank)
                print_log(f'Use distributed mode with GPUs: local rank={self._rank}')
            else:
                device = torch.device('cuda:0')
                print_log(f'Use non-distributed mode with GPU: {device}')
        else:
            self._use_gpu = False
            device = torch.device('cpu')
            print_log('Use CPU')
            if self.args.dist:
                assert False, "Distributed training requires GPUs"
        return device

    def _preparation(self, dataloaders=None):
        """Preparation of environment and basic experiment setups"""
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(self.args.local_rank)

        # init distributed env first, since logger depends on the dist info.
        if self.args.launcher != 'none' or self.args.dist:
            self._dist = True
        if self._dist:
            assert self.args.launcher != 'none'
            dist_params = dict(backend='nccl', init_method='env://')
            if self.args.launcher == 'slurm':
                dist_params['port'] = self.args.port
            init_dist(self.args.launcher, **dist_params)
            self._rank, self._world_size = get_dist_info()
            # re-set gpu_ids with distributed training mode
            self._gpu_ids = range(self._world_size)
        self.device = self._acquire_device()
        if self._early_stop < 0:
            self._early_stop = self._max_epochs * 2

        # log and checkpoint
        base_dir = self.args.res_dir if self.args.res_dir is not None else 'work_dirs'
        self.path = osp.join(base_dir, self.args.ex_name if not self.args.ex_name.startswith(self.args.res_dir) \
            else self.args.ex_name.split(self.args.res_dir+'/')[-1])
        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        if self._rank == 0:
            check_dir(self.path)
            check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        if self._rank == 0:
            with open(sv_param, 'w') as file_obj:
                json.dump(self.args.__dict__, file_obj)

            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            prefix = 'train' if (not self.args.test and not self.args.inference) else 'test'
            logging.basicConfig(level=logging.INFO,
                                filename=osp.join(self.path, '{}_{}.log'.format(prefix, timestamp)),
                                filemode='a', format='%(asctime)s - %(message)s')

        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        if self._rank == 0:
            print_log('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

        # set random seeds
        if self._dist:
            seed = init_random_seed(self.args.seed)
            seed = seed + dist.get_rank() if self.args.diff_seed else seed
        else:
            seed = self.args.seed
        set_seed(seed)

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
        self.method.model.eval()
        # setup ddp training
        if self._dist:
            self.method.model.cuda()
            if self.args.torchscript:
                self.method.model = torch.jit.script(self.method.model)
            self.method._init_distributed()

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
        checkpoint = {
            'epoch': self._epoch + 1,
            'optimizer': self.method.model_optim.state_dict(),
            'state_dict': weights_to_cpu(self.method.model.state_dict()) \
                if not self._dist else weights_to_cpu(self.method.model.module.state_dict()),
            'scheduler': self.method.scheduler.state_dict()}
        torch.save(checkpoint, osp.join(self.checkpoints_path, name + '.pth'))

    def _load(self, name=''):
        """Loading models from the checkpoint"""
        filename = name if osp.isfile(name) else osp.join(self.checkpoints_path, name + '.pth')
        try:
            checkpoint = torch.load(filename)
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
        if self._dist:
            try:
                self.method.model.module.load_state_dict(state_dict)
            except:
                self.method.model.load_state_dict(state_dict)
        else:
            self.method.model.load_state_dict(state_dict)

    def _load_best_checkpoint(self):
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        self._load_from_state_dict(torch.load(best_model_path))

    def _slice_eval_channels(self, preds, trues, save_channels=None):
        if save_channels is None:
            return preds, trues
        if preds.shape[2] < save_channels or trues.shape[2] < save_channels:
            raise ValueError(
                f"Metrics expect at least {save_channels} channels, got "
                f"{preds.shape[2]} (pred) and {trues.shape[2]} (true)."
            )
        return preds[:, :, :save_channels, ...], trues[:, :, :save_channels, ...]

    def _compute_eval_results(self, results, data_loader, save_channels=None):
        metric_list, spatial_norm, channel_names = self.args.metrics, True, None
        preds_eval, trues_eval = self._slice_eval_channels(
            results['preds'], results['trues'], save_channels=save_channels
        )
        eval_res, eval_log = metric(
            preds_eval,
            trues_eval,
            data_loader.dataset.mean,
            data_loader.dataset.std,
            metrics=metric_list,
            channel_names=channel_names,
            spatial_norm=spatial_norm
        )
        results['metrics'] = np.array([eval_res['mae'], eval_res['mse'], eval_res['rmse'], eval_res['mape']])
        return eval_res, eval_log

    def _save_results_arrays(self, results, folder_path, np_data_list,
                             save_channels=10, save_stride=None, epoch_tag=None):
        check_dir(folder_path)
        if save_stride is not None:
            save_stride = max(1, int(save_stride))
        save_indices = slice(None, None, save_stride) if save_stride and save_stride > 1 else None
        for np_data in np_data_list:
            data_to_save = results[np_data]
            if np_data in {'inputs', 'trues', 'preds'}:
                if save_indices is not None:
                    data_to_save = data_to_save[save_indices]
                if data_to_save.ndim >= 3:
                    data_to_save = data_to_save[:, :, :save_channels, ...]
            filename = f'{np_data}_{epoch_tag}.npy' if epoch_tag else f'{np_data}.npy'
            np.save(osp.join(folder_path, filename), data_to_save)



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

        if self.args.method in ['predformer', 'cnn', 'rnn', 'lstm', 'convlstm']:
            input_dummy = torch.ones(1, self.args.pre_seq_length, C, dummy_h, dummy_w).to(self.device)
        else:
            raise ValueError(f'Invalid method name {self.args.method}')

        dash_line = '-' * 80 + '\n'
        info = self.method.model.__repr__()
        flops = FlopCountAnalysis(self.method.model, input_dummy)
        flops = flop_count_table(flops)
        if self.args.fps:
            fps = measure_throughput(self.method.model, input_dummy)
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
            if self._dist and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            num_updates, loss_mean = self.method.train_one_epoch(
                self, self.train_loader, epoch, num_updates
            )

            self._epoch = epoch
            if epoch % self.args.log_step == 0:
                cur_lr = self.method.current_lr()
                cur_lr = sum(cur_lr) / len(cur_lr)
                with torch.no_grad():
                    vali_loss = self.vali()
                    
                if self._rank == 0:
                    print_log('Epoch: {0}, Steps: {1} | Lr: {2:.7f} | Train Loss: {3:.7f} | Vali Loss: {4:.7f}\n'.format(
                        epoch + 1, len(self.train_loader), cur_lr, loss_mean.avg, vali_loss))
                    early_stop = recorder(vali_loss, self.method.model, self.path)
                    self._save(name='latest')

                    
            if self._use_gpu and self.args.empty_cache:
                torch.cuda.empty_cache()
            if epoch > self._early_stop and early_stop:  # early stop training
                #print_log('Early stop training at f{} epoch'.format(epoch))
                print_log('Early stop training at {} epoch'.format(epoch + 1))
                break
            
        if not check_dir(self.path):  # exit training when work_dir is removed
            assert False and "Exit training because work_dir is removed"
        self._load_best_checkpoint()
        time.sleep(1)  # wait for asynchronous loggers to flush

    def vali(self):
        """A validation loop during training"""
        gather_data = getattr(self.args, 'gather_data', True)
        t0 = time.time()
        results, eval_log = self.method.vali_one_epoch(
            self, self.vali_loader, gather_data=gather_data
        )
        t1 = time.time()

        eval_res = None
        if gather_data:
            save_channels = getattr(self.args, "save_channels", None)
            eval_res, eval_log = self._compute_eval_results(
                results, self.vali_loader, save_channels=save_channels
            )
        t2 = time.time()

        if self._rank == 0:
            val_msg = f"val_timing\tforward_collect={t1 - t0:.2f}s"
            if gather_data:
                val_msg += f", eval_metrics={t2 - t1:.2f}s"
            print_log(val_msg)
        results['metrics'] = np.array([eval_res['mae'], eval_res['mse'],eval_res['rmse'], eval_res['mape']])
        
        if self._rank == 0:
            print_log('val\t '+eval_log)
            if has_nni and 'mse' in results:
                nni.report_intermediate_result(results['mse'].mean())
            if gather_data:
                save_stride = getattr(self.args, 'val_save_stride', None)
                if save_stride is not None and int(save_stride) > 0:
                    folder_path = osp.join(self.path, 'val_saved')
                    epoch_tag = f'epoch_{self._epoch + 1:03d}'
                    save_channels = getattr(self.args, "save_channels", 10)
                    self._save_results_arrays(
                        results,
                        folder_path,
                        ['inputs', 'trues', 'preds', 'metrics'],
                        save_channels=save_channels,
                        save_stride=save_stride,
                        epoch_tag=epoch_tag
                    )

        return results['loss'].mean()

    def test(self):
        """A testing loop of STL methods"""
        if self.args.test:
            self._load_best_checkpoint()

        t0 = time.time()
        results = self.method.test_one_epoch(self, self.test_loader, gather_data=True)
        t1 = time.time()

        save_channels = getattr(self.args, "save_channels", None)
        eval_res, eval_log = self._compute_eval_results(
            results, self.test_loader, save_channels=save_channels
        )
        t2 = time.time()

        if self._rank == 0:
            print_log(f"test_timing\tforward_collect={t1 - t0:.2f}s, eval_metrics={t2 - t1:.2f}s")
            print_log(eval_log)
            # Saving test outputs is intentionally disabled.

        return eval_res['mse']
