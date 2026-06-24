import cv2
import os
import logging
import platform
import random
import subprocess
import sys
import warnings
import numpy as np
from collections import defaultdict

import torch
import torchvision
import torch.multiprocessing as mp

import openstl
from .config_utils import Config
from .dist_utils import is_main_process


def set_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def setup_multi_processes(cfg):
    """Setup multi-processing environment variables."""
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != 'Windows':
        mp_start_method = cfg.get('mp_start_method', 'fork')
        current_method = mp.get_start_method(allow_none=True)
        if current_method is not None and current_method != mp_start_method:
            warnings.warn(
                f'Multi-processing start method `{mp_start_method}` is '
                f'different from the previous setting `{current_method}`.'
                f'It will be force set to `{mp_start_method}`. You can change '
                f'this behavior by changing `mp_start_method` in your config.')
        mp.set_start_method(mp_start_method, force=True)

    # disable opencv multithreading to avoid system being overloaded
    opencv_num_threads = cfg.get('opencv_num_threads', 0)
    cv2.setNumThreads(opencv_num_threads)

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    if 'OMP_NUM_THREADS' not in os.environ and cfg['num_workers'] > 1:
        omp_num_threads = 1
        warnings.warn(
            f'Setting OMP_NUM_THREADS environment variable for each process '
            f'to be {omp_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ and cfg['num_workers'] > 1:
        mkl_num_threads = 1
        warnings.warn(
            f'Setting MKL_NUM_THREADS environment variable for each process '
            f'to be {mkl_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)


def collect_env():
    """Collect the information of the running environments."""
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available

    if cuda_available:
        from torch.utils.cpp_extension import CUDA_HOME
        env_info['CUDA_HOME'] = CUDA_HOME

        if CUDA_HOME is not None and os.path.isdir(CUDA_HOME):
            try:
                nvcc = os.path.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(
                    '"{}" -V | tail -n1'.format(nvcc), shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            env_info['GPU ' + ','.join(devids)] = name

    gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
    gcc = gcc.decode('utf-8').strip()
    env_info['GCC'] = gcc

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = torch.__config__.show()
    env_info['TorchVision'] = torchvision.__version__
    env_info['OpenCV'] = cv2.__version__

    env_info['openstl'] = openstl.__version__

    return env_info


def print_log(message):
    if not is_main_process():
        return
    print(message)
    logging.info(message)


def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message


def check_dir(path):
    os.makedirs(path, exist_ok=True)
    return True


def get_dataset(dataname, config):
    from openstl.datasets import load_data
    return load_data(**config)



def measure_throughput(model, input_dummy):

    def get_batch_size(H, W):
        max_side = max(H, W)
        # if max_side >= 128:
        if max_side >= 64:
            bs = 4
            repetitions = 1000
        else:
            bs = 100
            repetitions = 100
        return bs, repetitions
    
    model.eval()
    
    if isinstance(input_dummy, tuple):
        input_dummy = list(input_dummy)
        _, T, C, H, W = input_dummy[0].shape
        bs, repetitions = get_batch_size(H, W)
        _input = torch.rand(bs, T, C, H, W).to(input_dummy[0].device)
        _output = torch.rand(bs, T, C, H, W).to(input_dummy[1].device)
        input_dummy[0],input_dummy[1] = _input,_output
        input_dummy = tuple(input_dummy)
    else:
        _, T, C, H, W = input_dummy.shape
        bs, repetitions = get_batch_size(H, W)
        input_dummy = torch.rand(bs, T, C, H, W).to(input_dummy.device)
    total_time = 0
    with torch.no_grad():
        for _ in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            if isinstance(input_dummy, tuple):
                _ = model(*input_dummy)
            else:
                _ = model(input_dummy)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (repetitions * bs) / total_time
    return Throughput


def load_config(filename):
    """Load a required Python configuration file."""
    print(f"loading config from {filename} ...")
    return dict(Config(filename=filename)._cfg_dict)
