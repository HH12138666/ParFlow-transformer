import os.path as osp
import warnings
warnings.filterwarnings('ignore')

from openstl.api import BaseExperiment
from openstl.utils import (create_parser, default_parser, load_config,
                           setup_multi_processes, update_config)


if __name__ == '__main__':
    # 创建参数解析器并解析命令行参数
    args = create_parser().parse_args()
    config = args.__dict__
    
    
    cfg_path = osp.join('./configs', args.dataname, f'{args.method}.py') \
        if args.config_file is None else args.config_file
    if args.overwrite:
        config = update_config(config, load_config(cfg_path),
                               exclude_keys=['method'])
    else:
        loaded_cfg = load_config(cfg_path)
        config = update_config(config, loaded_cfg,
                               exclude_keys=['method', 'batch_size', 'val_batch_size',
                                             'drop_path', 'warmup_epoch'])
        default_values = default_parser()
        for attribute in default_values.keys():
            if config[attribute] is None:
                config[attribute] = default_values[attribute]

    # set multi-process settings
    setup_multi_processes(config)

    print('>'*35 + ' training ' + '<'*35)
    exp = BaseExperiment(args)
    exp.train()

    print('>'*35 + ' testing  ' + '<'*35)
    mse = exp.test()
