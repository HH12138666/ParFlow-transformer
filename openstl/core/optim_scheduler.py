import json
from torch import optim

from timm.scheduler.cosine_lr import CosineLRScheduler

from .optim_constant import optim_parameters



def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def get_optim_scheduler(args, epoch, model, steps_per_epoch):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay

    # if weight_decay and filter_bias_and_bn:
    if args.filter_bias_and_bn:
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        else:
            skip = {}
        parameters = get_parameter_groups(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    opt_args = optim_parameters.get(opt_lower, dict())
    opt_args.update(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower != 'adamw':
        raise ValueError(f"Only adamw is supported, got opt={args.opt}")
    optimizer = optim.AdamW(parameters, **opt_args)

    sched_lower = args.sched.lower()
    total_steps = epoch * steps_per_epoch
    by_epoch = True
    if sched_lower == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=epoch,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epoch,
            t_in_epochs=True,  # update lr by_epoch
            k_decay=getattr(args, 'lr_k_decay', 1.0))
    else:
        raise ValueError(f"Only cosine scheduler is supported, got sched={args.sched}")

    return optimizer, lr_scheduler, by_epoch
