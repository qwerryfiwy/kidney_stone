"""Dataset loading utilities.

Expects an ImageFolder-style directory:

    data_root/
    ├── stone/       (label 1)
    └── non_stone/   (label 0)

Provides:
  - CTStoneDataset: thin wrapper around torchvision.datasets.ImageFolder
    that guarantees a fixed class-to-index mapping.
  - build_stratified_split_loaders: the 80/10/10 protocol used for the
    Colour 2D Mixed Data (Subset 1) experiments (Section 3.2).
  - build_kfold_splits: the 5-fold CV protocol used for the Colour 2D
    Large and Grayscale Axial CT datasets (Section 3.2).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold, train_test_split

from data.transforms import build_train_transforms, build_eval_transforms

CLASS_TO_IDX = {"non_stone": 0, "stone": 1}


class CTStoneDataset(ImageFolder):
    def __init__(self, root: str, transform=None):
        super().__init__(root=root, transform=transform)
        # Enforce a fixed, known label ordering regardless of folder order.
        if set(self.class_to_idx) == set(CLASS_TO_IDX):
            self.class_to_idx = CLASS_TO_IDX
            self.samples = [
                (path, CLASS_TO_IDX[self.classes[label]])
                for path, label in self.samples
            ]
            self.targets = [s[1] for s in self.samples]


def _get_labels(dataset: ImageFolder) -> np.ndarray:
    return np.array([label for _, label in dataset.samples])


def build_stratified_split_loaders(
    root: str,
    image_size: int = 224,
    batch_size: int = 64,
    val_split: float = 0.10,
    test_split: float = 0.10,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """80/10/10 stratified split, evaluated across independent random seeds
    (Section 3.2, Colour 2D Mixed Data protocol)."""

    base = CTStoneDataset(root=root, transform=None)
    labels = _get_labels(base)
    indices = np.arange(len(base))

    train_idx, temp_idx, train_y, temp_y = train_test_split(
        indices, labels, test_size=(val_split + test_split),
        stratify=labels, random_state=seed,
    )
    rel_test_size = test_split / (val_split + test_split)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=rel_test_size, stratify=temp_y, random_state=seed,
    )

    train_ds = CTStoneDataset(root=root, transform=build_train_transforms(image_size))
    eval_ds = CTStoneDataset(root=root, transform=build_eval_transforms(image_size))

    train_loader = DataLoader(
        Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        Subset(eval_ds, val_idx), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(eval_ds, test_idx), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def build_kfold_splits(
    root: str,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Returns a list of (train_idx, test_idx) pairs for stratified k-fold
    CV (Section 3.2, repeated 5-fold protocol)."""

    base = CTStoneDataset(root=root, transform=None)
    labels = _get_labels(base)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(labels)), labels))


def build_fold_loaders(
    root: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    val_fraction_of_train: float = 0.1111,  # ~10% of total -> 1/9 of train fold
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Given one fold's train/test indices, carves a small validation set
    out of the training fold (for early stopping) and builds loaders."""

    base = CTStoneDataset(root=root, transform=None)
    labels = _get_labels(base)

    sub_train_idx, sub_val_idx = train_test_split(
        train_idx,
        test_size=val_fraction_of_train,
        stratify=labels[train_idx],
        random_state=seed,
    )

    train_ds = CTStoneDataset(root=root, transform=build_train_transforms(image_size))
    eval_ds = CTStoneDataset(root=root, transform=build_eval_transforms(image_size))

    train_loader = DataLoader(
        Subset(train_ds, sub_train_idx), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        Subset(eval_ds, sub_val_idx), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(eval_ds, test_idx), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
