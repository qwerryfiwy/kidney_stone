"""Assembles the full EfficientNet-FPN-Max (ENFM) model and its ablations:

    ENFM        = backbone -> top-down FPN -> attention-gated bottom-up
                   -> multi-level max pooling -> classifier   (Algorithm 1)
    FPN-Avg     = backbone -> top-down FPN -> attention-gated bottom-up
                   -> multi-level AVERAGE pooling -> classifier
    FPN-Dual    = backbone -> top-down FPN -> attention-gated bottom-up
                   -> multi-level DUAL (avg+max) pooling -> classifier
    FPN-GMP     = backbone -> top-down FPN only (no bottom-up refinement)
                   -> single-level global max pooling on P3 -> classifier
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.backbone import build_backbone
from models.fpn import TopDownFPN
from models.attention_gate import AttentionGatedBottomUp
from models.pooling import build_pooling
from models.classifier import ClassificationHead


class EfficientNetFPN(nn.Module):
    """Generic wrapper: backbone + top-down FPN + optional bottom-up gate
    + configurable pooling head + classifier. All variants in the paper's
    ablation study (Max / Avg / Dual / GMP) share this skeleton.
    """

    def __init__(
        self,
        pyramid_channels: int = 256,
        num_classes: int = 2,
        pretrained: bool = True,
        pooling: str = "max",           # "max" | "avg" | "dual" | "gmp_single"
        use_bottom_up: bool = True,
        dropout_fc1: float = 0.3,
        dropout_fc2: float = 0.2,
    ):
        super().__init__()
        self.use_bottom_up = use_bottom_up

        self.backbone = build_backbone(pretrained=pretrained)
        self.fpn = TopDownFPN(self.backbone.out_channels, out_channels=pyramid_channels)

        if use_bottom_up:
            self.bottom_up = AttentionGatedBottomUp(channels=pyramid_channels)
            n_levels = 4  # n3, n4, n5, n6
        else:
            self.bottom_up = None
            n_levels = 1  # single P3 level for the GMP ablation

        self.pool = build_pooling(pooling)

        if pooling == "dual":
            fused_dim = pyramid_channels * 2 * n_levels
        else:
            fused_dim = pyramid_channels * n_levels

        self.head = ClassificationHead(
            in_features=fused_dim,
            num_classes=num_classes,
            dropout1=dropout_fc1,
            dropout2=dropout_fc2,
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        c_feats = self.backbone(x)
        p_feats = self.fpn(c_feats)

        if self.use_bottom_up:
            pyramid = self.bottom_up(p_feats)
        else:
            pyramid = p_feats  # pooling head reads "P3" directly

        z = self.pool(pyramid)
        logits = self.head(z)

        if return_features:
            return logits, {"pyramid": pyramid, "fused": z}
        return logits


def build_enfm(
    variant: str = "enfm",
    pyramid_channels: int = 256,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout_fc1: float = 0.3,
    dropout_fc2: float = 0.2,
) -> EfficientNetFPN:
    """Factory for the ENFM family.

    variant:
        "enfm"     -> proposed model (attention-gated bottom-up + multi-level max)
        "fpn_avg"  -> ablation, multi-level average pooling
        "fpn_dual" -> ablation, multi-level dual (avg+max) pooling
        "fpn_gmp"  -> ablation, single-level global max pooling, no bottom-up
    """
    variant_cfg = {
        "enfm":     dict(pooling="max", use_bottom_up=True),
        "fpn_avg":  dict(pooling="avg", use_bottom_up=True),
        "fpn_dual": dict(pooling="dual", use_bottom_up=True),
        "fpn_gmp":  dict(pooling="gmp_single", use_bottom_up=False),
    }
    if variant not in variant_cfg:
        raise ValueError(f"Unknown ENFM variant '{variant}'. Choices: {list(variant_cfg)}")

    return EfficientNetFPN(
        pyramid_channels=pyramid_channels,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_fc1=dropout_fc1,
        dropout_fc2=dropout_fc2,
        **variant_cfg[variant],
    )
