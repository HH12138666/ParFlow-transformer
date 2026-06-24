"""Checkpoint serialization shared by training and finetuning."""

from pathlib import Path

import torch


def save_training_checkpoint(path, model, optimizer, scheduler, epoch):
    model_ref = model.module if hasattr(model, "module") else model
    checkpoint = {
        "epoch": int(epoch),
        "optimizer": optimizer.state_dict(),
        "state_dict": _state_to_cpu(model_ref.state_dict()),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(checkpoint, Path(path))


def load_checkpoint(path, device):
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def load_model_state(model, state_dict):
    model_ref = model.module if hasattr(model, "module") else model
    model_ref.load_state_dict(state_dict)


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint must be a dictionary")
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise RuntimeError("Checkpoint does not contain a model state_dict")


def _state_to_cpu(state_dict):
    result = state_dict.__class__((key, value.cpu()) for key, value in state_dict.items())
    result._metadata = getattr(state_dict, "_metadata", {})
    return result
