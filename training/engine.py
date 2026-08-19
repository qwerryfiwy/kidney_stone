"""Core train/eval loops: AdamW + reduce-on-plateau scheduler + AMP +
early stopping, exactly matching the paper's training recipe
(Section 3.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import compute_all_metrics


@dataclass
class TrainConfig:
    epochs: int = 20
    early_stopping_patience: int = 5
    amp: bool = True
    grad_clip_norm: Optional[float] = None
    lr: float = 5e-5
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2


@dataclass
class TrainHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_metrics: list = field(default_factory=list)
    best_epoch: int = -1
    best_val_auc: float = -1.0


def build_optimizer_and_scheduler(model: nn.Module, cfg: TrainConfig):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
    )
    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip_norm: Optional[float] = None,
) -> float:
    model.train()
    running_loss, n_samples = 0.0, 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip_norm:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

    return running_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running_loss, n_samples = 0.0, 0
    all_probs, all_preds, all_labels = [], [], []

    for images, labels in tqdm(loader, desc="eval", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    metrics = compute_all_metrics(
        y_true=np.concatenate(all_labels),
        y_pred=np.concatenate(all_preds),
        y_prob=np.concatenate(all_probs),
    )
    metrics["loss"] = running_loss / max(n_samples, 1)
    return metrics


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    checkpoint_path: Optional[str] = None,
) -> TrainHistory:
    """Full training loop with early stopping on validation AUC and a
    reduce-on-plateau scheduler monitoring validation loss."""

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    scaler = torch.cuda.amp.GradScaler() if (cfg.amp and device.type == "cuda") else None

    history = TrainHistory()
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, cfg.grad_clip_norm
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        history.train_loss.append(train_loss)
        history.val_loss.append(val_metrics["loss"])
        history.val_metrics.append(val_metrics)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["auc"] > history.best_val_auc:
            history.best_val_auc = val_metrics["auc"]
            history.best_epoch = epoch
            epochs_without_improvement = 0
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (best epoch {history.best_epoch}).")
            break

    return history
