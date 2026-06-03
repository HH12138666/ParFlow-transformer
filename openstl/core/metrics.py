import numpy as np


def MAE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean(np.abs(pred - true), axis=(0, 1)).sum()
    norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
    return np.mean(np.abs(pred - true) / norm, axis=(0, 1)).sum()


def MSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean((pred - true) ** 2, axis=(0, 1)).sum()
    norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
    return np.mean((pred - true) ** 2 / norm, axis=(0, 1)).sum()


def RMSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.sqrt(np.mean((pred - true) ** 2, axis=(0, 1)).sum())
    norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
    return np.sqrt(np.mean((pred - true) ** 2 / norm, axis=(0, 1)).sum())


def _denormalize(pred, true, mean, std, progress_hook=None):
    if mean is None or std is None:
        return pred, true
    if pred.ndim < 3:
        raise ValueError(f"Expected pred with channel dimension, got shape {pred.shape}")

    c_pred = pred.shape[2]
    c_mean = mean.shape[0]
    if c_pred > c_mean:
        raise ValueError(
            f"Channel count mismatch: pred has {c_pred} channels, mean/std has {c_mean} channels."
        )

    mean = mean[:c_pred].reshape(1, 1, -1, 1, 1)
    std = std[:c_pred].reshape(1, 1, -1, 1, 1)
    if progress_hook is not None:
        progress_hook()
    return pred * std + mean, true * std + mean


def metric(pred, true, mean=None, std=None, metrics=['mae', 'mse'],
           spatial_norm=True, return_log=True, progress_hook=None):
    pred, true = _denormalize(pred, true, mean, std, progress_hook=progress_hook)
    eval_res = {}
    allowed_metrics = ['mae', 'mse', 'rmse']
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if invalid_metrics:
        raise ValueError(f'metric {invalid_metrics} is not supported.')

    if 'mse' in metrics:
        eval_res['mse'] = MSE(pred, true, spatial_norm)
        if progress_hook is not None:
            progress_hook()

    if 'mae' in metrics:
        eval_res['mae'] = MAE(pred, true, spatial_norm)
        if progress_hook is not None:
            progress_hook()

    if 'rmse' in metrics:
        eval_res['rmse'] = RMSE(pred, true, spatial_norm)
        if progress_hook is not None:
            progress_hook()

    eval_log = ""
    if return_log:
        for key, value in eval_res.items():
            eval_log += f"{', ' if eval_log else ''}{key}:{value}"
    return eval_res, eval_log
