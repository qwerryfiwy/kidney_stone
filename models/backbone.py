"""EfficientNet-B0 backbone that exposes multi-scale feature maps C3, C4, C5.

Corresponds to Section 3.3.1 of the paper: features are taken after the
3rd, 5th, and final inverted-residual blocks (block3b_add, block5c_add,
top_activation in the Keras naming used by the paper; here we use the
equivalent timm feature-extraction indices).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:  # pragma: no cover
    raise ImportError("timm is required: pip install timm") from e


class EfficientNetB0Backbone(nn.Module):
    """Wraps timm's efficientnet_b0 as a feature pyramid source.

    Returns a dict {"C3": ..., "C4": ..., "C5": ...} with channel counts
    matching the paper (40, 112, 1280 for stock EfficientNet-B0 at these
    stages).
    """

    # timm feature_info indices for efficientnet_b0 stages (0-indexed,
    # out_indices selects which reduction stages to return).
    OUT_INDICES = (2, 3, 4)  # -> strides 8, 16, 32 roughly matching C3,C4,C5

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            features_only=True,
            out_indices=self.OUT_INDICES,
        )
        self.out_channels: List[int] = self.encoder.feature_info.channels()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.encoder(x)  # list of 3 tensors, shallow -> deep
        return {"C3": feats[0], "C4": feats[1], "C5": feats[2]}


def build_backbone(pretrained: bool = True) -> EfficientNetB0Backbone:
    return EfficientNetB0Backbone(pretrained=pretrained)
