"""Augmentation pipelines matching Section 3.2's preprocessing:

- resize to 224x224, replicate grayscale to 3 channels
- train: random horizontal flip (p=0.5), rotation (+-10 deg),
  brightness/contrast jitter, then ImageNet normalization
- val/test: deterministic resize + normalization only
"""
from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _ensure_three_channels():
    return transforms.Lambda(
        lambda img: img.convert("RGB") if img.mode != "RGB" else img
    )


def build_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            _ensure_three_channels(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            _ensure_three_channels(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
