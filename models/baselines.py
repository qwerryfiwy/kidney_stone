"""Standard baseline classifiers used for comparison in Tables 2, 5, 8:
EfficientNet-B0, EfficientNet-V2S, MobileNetV3, ResNet50.

Each is a plain ImageNet-pretrained backbone with its classifier head
replaced by a 2-class linear layer, so all models are trained/evaluated
through the exact same pipeline as ENFM.
"""
from __future__ import annotations

import torch.nn as nn
import timm


def _replace_head(model: nn.Module, num_classes: int) -> nn.Module:
    """timm models expose reset_classifier for this purpose."""
    model.reset_classifier(num_classes=num_classes)
    return model


BASELINE_TIMM_NAMES = {
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_v2s": "tf_efficientnetv2_s",
    "mobilenetv3": "mobilenetv3_large_100",
    "resnet50": "resnet50",
}


def build_baseline(name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    if name not in BASELINE_TIMM_NAMES:
        raise ValueError(
            f"Unknown baseline '{name}'. Choices: {list(BASELINE_TIMM_NAMES)}"
        )
    timm_name = BASELINE_TIMM_NAMES[name]
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    return model
