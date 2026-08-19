"""Checkpoint saving and loading utilities."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    model: nn.Module,
    path: str,
    epoch: Optional[int] = None,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    history: Optional[Any] = None,
) -> None:
    """Saves a training checkpoint (both model state and optimizer/scheduler metadata)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if history is not None:
        checkpoint["history"] = history

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Optional[int], Optional[Any]]:
    """Loads model weights and optional optimizer/scheduler state from a checkpoint.

    Supports both full checkpoint dictionaries and raw state dicts.

    Returns:
        tuple: (epoch, history) if available in the checkpoint, else (None, None).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at '{path}'")

    checkpoint = torch.load(path, map_location=device)

    # Check if this is a standard packaged checkpoint or raw state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint.get("epoch", None), checkpoint.get("history", None)
    else:
        # Raw state dict
        model.load_state_dict(checkpoint)
        return None, None
