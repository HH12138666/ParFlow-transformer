import warnings
warnings.filterwarnings('ignore')

from openstl.api import BaseExperiment
from openstl.utils import (create_parser, default_parser, load_config,
                           setup_multi_processes, update_config)


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    
    assert args.config_file is not None, "Config file is required for testing"
    config = update_config(config, load_config(args.config_file),
                           exclude_keys=['method', 'batch_size', 'val_batch_size'])
    default_values = default_parser()


    for attribute in default_values.keys():
        if config[attribute] is None:
            config[attribute] = default_values[attribute]
    if not config['inference'] and not config['test']:
        config['test'] = True

    # set multi-process settings
    setup_multi_processes(config)
    
    print('>'*35 + ' testing  ' + '<'*35)
    exp = BaseExperiment(args)

    mse = exp.test()
