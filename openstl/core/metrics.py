import numpy as np


def rescale(x):
    return (x - x.max()) / (x.max() - x.min()) * 2 - 1


def MAE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean(np.abs(pred-true), axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean(np.abs(pred-true) / norm, axis=(0, 1)).sum()


def MSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean((pred-true)**2, axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean((pred-true)**2 / norm, axis=(0, 1)).sum()


def RMSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.sqrt(np.mean((pred-true)**2, axis=(0, 1)).sum())
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.sqrt(np.mean((pred-true)**2 / norm, axis=(0, 1)).sum())

def MAPE(pred, true, spatial_norm=False, eps=1e-8):
    denom = np.where(np.abs(true) < eps, eps, np.abs(true))
    percentage_error = np.abs((pred - true) / denom)
    if not spatial_norm:
        return np.mean(percentage_error, axis=(0, 1)).sum()
    norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
    return np.mean(percentage_error / norm, axis=(0, 1)).sum()


def metric(pred, true, mean=None, std=None, metrics=['mae', 'mse'],
           clip_range=[0, 1], channel_names=None,
           spatial_norm=True, return_log=True):
   
    if mean is not None and std is not None:

        if pred.ndim >= 3:
            
            C_pred = pred.shape[2]
            C_mean = mean.shape[0]

            if C_pred <= C_mean:
                mean = mean[:C_pred]
                std = std[:C_pred]
                mean = mean.reshape(1, 1, -1, 1, 1)
                std = std.reshape(1, 1, -1, 1, 1)
            else:
                raise ValueError(f"Channel count mismatch: pred has {C_pred} channels, mean/std has {C_mean} channels.")

        
        pred = pred * std + mean
        true = true * std + mean
    '''
    if mean is not None and std is not None:
        pred = pred * std + mean
        true = true * std + mean
    '''
    eval_res = {}
    eval_log = ""
    allowed_metrics = ['mae', 'mse', 'rmse', 'mape']
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')
    if isinstance(channel_names, list):
        assert pred.shape[2] % len(channel_names) == 0 and len(channel_names) > 1
        c_group = len(channel_names)
        c_width = pred.shape[2] // c_group
    else:
        channel_names, c_group, c_width = None, None, None

    if 'mse' in metrics:
        if channel_names is None:
            eval_res['mse'] = MSE(pred, true, spatial_norm)
        else:
            mse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mse_{str(c_name)}'] = MSE(pred[:, :, i*c_width: (i+1)*c_width, ...],
                                                     true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
                mse_sum += eval_res[f'mse_{str(c_name)}']
            eval_res['mse'] = mse_sum / c_group

    if 'mae' in metrics:
        if channel_names is None:
            eval_res['mae'] = MAE(pred, true, spatial_norm)
        else:
            mae_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mae_{str(c_name)}'] = MAE(pred[:, :, i*c_width: (i+1)*c_width, ...],
                                                     true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
                mae_sum += eval_res[f'mae_{str(c_name)}']
            eval_res['mae'] = mae_sum / c_group

    if 'rmse' in metrics:
        if channel_names is None:
            eval_res['rmse'] = RMSE(pred, true, spatial_norm)
        else:
            rmse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'rmse_{str(c_name)}'] = RMSE(pred[:, :, i*c_width: (i+1)*c_width, ...],
                                                       true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
                rmse_sum += eval_res[f'rmse_{str(c_name)}']
            eval_res['rmse'] = rmse_sum / c_group
                
    if 'mape' in metrics:
        if channel_names is None:
            eval_res['mape'] = MAPE(pred, true, spatial_norm)
        else:
            mape_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mape_{str(c_name)}'] = MAPE(pred[:, :, i*c_width: (i+1)*c_width, ...],
                                                       true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
                mape_sum += eval_res[f'mape_{str(c_name)}']
            eval_res['mape'] = mape_sum / c_group
    
    if return_log:
        for k, v in eval_res.items():
            eval_str = f"{k}:{v}" if len(eval_log) == 0 else f", {k}:{v}"
            eval_log += eval_str

    return eval_res, eval_log
