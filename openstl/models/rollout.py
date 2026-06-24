"""Shared autoregressive rollout helpers."""

import torch


def merge_pred_with_aux(pred, prev_seq, input_channels, out_channels):
    if input_channels == out_channels:
        return pred
    if prev_seq.shape[2] < input_channels:
        raise ValueError(
            f"Input has {prev_seq.shape[2]} channels, expected {input_channels}"
        )
    if out_channels > input_channels:
        raise ValueError(
            f"Output channels {out_channels} cannot exceed input channels {input_channels}"
        )
    aux = prev_seq[:, :, out_channels:input_channels]
    return torch.cat([pred, aux], dim=2)
