import argparse

def parse_list(arg):
    return [int(x) for x in arg.strip('[]').split(',')]


def create_parser():
    parser = argparse.ArgumentParser(
        description='OpenSTL train/test a model')
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--dist', action='store_true', default=False,
                        help='Whether to use distributed training (DDP)')
    parser.add_argument('--display_step', default=10, type=int,
                        help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='work_dirs', type=str)
    parser.add_argument('--ex_name', '-ex', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Whether to use Native AMP for mixed precision training (PyTorch=>1.6.0)')
    parser.add_argument('--torchscript', action='store_true', default=False,
                        help='Whether to use torchscripted model')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--diff_seed', action='store_true', default=False,
                        help='Whether to set different seeds for different ranks')
    parser.add_argument('--fps', action='store_true', default=False,
                        help='Whether to measure inference speed (FPS)')
    parser.add_argument('--empty_cache', action='store_true', default=True,
                        help='Whether to empty cuda cache after GPU training')
    parser.add_argument('--find_unused_parameters', action='store_true', default=False,
                        help='Whether to find unused parameters in forward during DDP training')
    parser.add_argument('--broadcast_buffers', action='store_false', default=True,
                        help='Whether to set broadcast_buffers to false during DDP training')
    parser.add_argument('--resume_from', type=str, default=None, help='the checkpoint file to resume from')
    parser.add_argument('--auto_resume', action='store_true', default=False,
                        help='When training was interupted, resume from the latest checkpoint')
    parser.add_argument('--test', action='store_true', default=False, help='Only performs testing')
    parser.add_argument('--inference', '-i', action='store_true', default=False, help='Only performs inference')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='whether to set deterministic options for CUDNN backend (reproducable)')
    parser.add_argument('--launcher', default='none', type=str,
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        help='job launcher for distributed training')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500,
                        help='port only works when launcher=="slurm"')

    # dataset parameters
    parser.add_argument('--batch_size', '-b', default=10, type=int, help='Training batch size')
    parser.add_argument('--val_batch_size', '-vb', default=10, type=int, help='Validation batch size')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--data_root', default='./data/parflow', type=str,)
    parser.add_argument('--dataname', '-d', default='parflow', type=str,
                        help='Dataset name (default: "parflow")')
    parser.add_argument('--patch_size', default=None, type=int, help='Patch size for training')
    parser.add_argument('--pre_seq_length', default=None, type=int, help='Sequence length before prediction')
    parser.add_argument('--aft_seq_length', default=None, type=int, help='Sequence length after prediction')
    parser.add_argument('--total_length', default=None, type=int, help='Total Sequence length for prediction')
    parser.add_argument('--use_augment', action='store_true', default=False,
                        help='Whether to use image augmentations for training')
    parser.add_argument('--use_prefetcher', action='store_true', default=False,
                        help='Whether to use prefetcher for faster data loading')
    parser.add_argument('--drop_last', action='store_true', default=False,
                        help='Whether to drop the last batch in the val data loading')
    parser.add_argument('--static_data', type=str, default=None,
                        help='Comma-separated keywords to select static .pfb files (case-insensitive)')
    parser.add_argument('--align_by_hour_id', action='store_true', default=True,
                        help='Align press/evap files by hour id parsed from filenames')
    parser.add_argument('--use_true_evap', action='store_true', default=True,
                        help='Use true evap channels during autoregressive rollout')
    parser.add_argument('--loss_channels', type=int, default=10,
                        help='Number of leading channels used to compute loss (e.g., 10 for press, 14 for press+evap)')
    parser.add_argument('--save_channels', type=int, default=10,
                        help='Number of leading channels saved for inputs/trues/preds during training/validation')
    
    # ParFlow tiling parameters
    parser.add_argument('--space_h', default=None, type=int, help='Spatial crop height for ParFlow tiling')
    parser.add_argument('--space_w', default=None, type=int, help='Spatial crop width for ParFlow tiling')
    parser.add_argument('--space_stride_h', default=None, type=int, help='Vertical stride for ParFlow tiling')
    parser.add_argument('--space_stride_w', default=None, type=int, help='Horizontal stride for ParFlow tiling')
    parser.add_argument('--val_save_stride', default=None, type=int,
                        help='Subsample stride for saving merged validation samples (saving every N samples)')
    parser.add_argument('--pad_to_patch', action='store_true', default=False,
                        help='Pad full-frame inputs to the next multiple of patch_size before model forward')
    parser.add_argument('--split_mode', default='ratio', type=str,
                        help="Data split mode: 'ratio' or 'year'")
    parser.add_argument('--train_years', type=parse_list, default=None,
                        help='Training years, e.g. [2019]')
    parser.add_argument('--holdout_years', type=parse_list, default=None,
                        help='Holdout years split into val/test, e.g. [2020]')
    parser.add_argument('--val_ratio_in_holdout', default=0.25, type=float,
                        help='Validation fraction inside each holdout year')
    parser.add_argument('--var_name', default='press', type=str,
                        help='Primary dynamic variable folder name under data_root (e.g., press or wtd)')
    parser.add_argument('--use_evap', default=True, type=bool,
                        help='Whether to read and concatenate evaptrans channels')
    parser.add_argument('--use_static_input', default=True, type=bool,
                        help='Whether to read and concatenate static channels')

    # method parameters
    parser.add_argument('--method', '-m', default='predformer', type=str,
                        choices=['predformer'],
                        help='Name of video prediction method to train')
    parser.add_argument('--config_file', '-c', default='configs/parflow/PredFormer.py', type=str,
                        help='Path to the default config file')
    
    #parser.add_argument('--model_type', default=None, type=str,
    #                    help='Name of model for predformer (default: None)')
    
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate(default: 0.)')
    
    # parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate for SimVP (default: 0.)')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Whether to allow overwriting the provided config file with args')
    
    # Training parameters (optimizer)
    parser.add_argument('--epoch', '-e', default=None, type=int, help='end epochs (default: 200)')
    parser.add_argument('--log_step', default=1, type=int, help='Log interval by step')
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--opt_eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer epsilon (default: None, use opt default)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer sgd momentum (default: 0.9)')
    parser.add_argument('--weight_decay', default=0., type=float, help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip_mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--early_stop_epoch', default=-1, type=int,
                        help='Check to early stop after this epoch')
    parser.add_argument('--no_display_method_info', action='store_true', default=False,
                        help='Do not display method info')

    # Training parameters (scheduler)
    parser.add_argument('--sched', default='onecycle', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "onecycle"')
    parser.add_argument('--lr', default=None, type=float, help='Learning rate (default: 1e-3)')
    parser.add_argument('--lr_k_decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--final_div_factor', type=float, default=1e4,
                        help='min_lr = initial_lr/final_div_factor for onecycle scheduler')
    parser.add_argument('--warmup_epoch', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay_epoch', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--multi_decay_epoch', type=parse_list, default=[20, 30, 40], help='Epochs to decay learning rate')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--filter_bias_and_bn', type=bool, default=False,
                        help='Whether to set the weight decay of bias and bn to 0')
    parser.add_argument('--patience', type=int, default=5, metavar='N',
                        help='epochs val loss do not decay, then decay learing rate')

    # Drop Path parameters
    parser.add_argument('--drop_path', default=0.0, type=float, help='Drop path rate (stochastic depth rate)')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate(default: 0.)')
    parser.add_argument('--cutoff', default=0, type=int, help='Epoch at which to cut off the drop path rate (used in early or late modes)')
    parser.add_argument('--cutmode', default='standard', type=str, choices=['standard', 'early', 'late'],
                        help='Mode for applying drop path rate scheduling: "standard", "early", or "late"')
    parser.add_argument('--drop_schedule', default='constant', type=str, choices=['constant', 'linear'],
                        help='Schedule for drop path rate in early mode: "constant" or "linear"')

    return parser


def default_parser():
    default_values = {
        # Set-up parameters
        'device': 'cuda',
        'dist': False,
        'display_step': 10,
        'res_dir': 'work_dirs',
        'ex_name': 'Debug',
        'use_gpu': True,
        'fp16': False,
        'torchscript': False,
        'seed': 42,
        'diff_seed': False,
        'fps': False,
        'empty_cache': True,
        'find_unused_parameters': False,
        'broadcast_buffers': True,
        'resume_from': None,
        'auto_resume': False,
        'test': False,
        'inference': False,
        'deterministic': False,
        'launcher': 'pytorch',
        'local_rank': 0,
        'port': 29500,
        
        # dataset parameters
        'batch_size': 16,
        'val_batch_size': 16,
        'num_workers': 4,
        'data_root': './data/parflow',
        'dataname': 'parflow',
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20,
        'use_augment': False,
        'use_prefetcher': False,
        'drop_last': False,
        'static_data': None,
        'align_by_hour_id': True,
        'use_true_evap': True,
        'loss_channels': 10,
        'save_channels': 10,
        
        # ParFlow tiling parameters
        'space_h': None,  
        'space_w': None,
        'space_stride_h': None,
        'space_stride_w': None,
        'val_save_stride': None,
        'pad_to_patch': False,
        'split_mode': 'ratio',
        'train_years': None,
        'holdout_years': None,
        'val_ratio_in_holdout': 0.25,
        'var_name': 'press',
        'use_evap': True,
        'use_static_input': True,
             
        # method parameters
        'method': 'predformer',
        'config_file': 'configs/parflow/PredFormer.py',
        #'model_type': 'gSTA',
        'drop': 0,
        'drop_path': 0,
        'overwrite': False,
        
        # Training parameters (optimizer)
        'epoch': 200,
        'log_step': 1,
        'opt': 'adam',
        'opt_eps': None,
        'opt_betas': None,
        'momentum': 0.9,
        'weight_decay': 0,
        'clip_grad': None,
        'clip_mode': 'norm',
        'early_stop_epoch': -1,
        'no_display_method_info': False,
        # Training parameters (scheduler)
        'sched': 'onecycle',
        'lr': 1e-3,
        'lr_k_decay': 1.0,
        'warmup_lr': 1e-5,
        'min_lr': 1e-6,
        'final_div_factor': 1e4,
        'warmup_epoch': 0,
        'decay_epoch': 100,
        'decay_rate': 0.1,
        'filter_bias_and_bn': False,

    }
    return default_values
