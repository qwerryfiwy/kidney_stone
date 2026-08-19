"""Attention-gated bottom-up path-aggregation module (Section 3.3.3, Eqs. 6-9).

Re-fuses the top-down pyramid {P3, P4, P5} bottom-up using learned
channel-wise attention gates, then appends an extra coarsest level n6
by plain downsampling (no lateral connection / gate) to widen the
receptive field before pooling.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class AttentionGateBlock(nn.Module):
    """Computes n_k from n_{k-1} and P_k~ (Eqs. 6-8)."""

    def __init__(self, channels: int):
        super().__init__()
        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Eq. 6
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1)       # Eq. 7
        self.gate_act = nn.Sigmoid()
        self.fuse_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # Eq. 8
        self.fuse_act = nn.ReLU(inplace=True)

    def forward(self, n_prev: torch.Tensor, p_top_down: torch.Tensor) -> torch.Tensor:
        d_k = self.downsample(n_prev)                      # Eq. 6
        # match spatial size defensively (odd input sizes / rounding)
        if d_k.shape[-2:] != p_top_down.shape[-2:]:
            d_k = nn.functional.interpolate(
                d_k, size=p_top_down.shape[-2:], mode="nearest"
            )
        a_k = self.gate_act(self.gate_conv(d_k))            # Eq. 7
        gated = p_top_down * a_k
        fused = gated + p_top_down + d_k
        n_k = self.fuse_act(self.fuse_conv(fused))          # Eq. 8
        return n_k


class AttentionGatedBottomUp(nn.Module):
    """Produces the 4-level pyramid {n3, n4, n5, n6} from {P3, P4, P5}."""

    def __init__(self, channels: int = 256):
        super().__init__()
        self.gate4 = AttentionGateBlock(channels)
        self.gate5 = AttentionGateBlock(channels)
        self.downsample6 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Eq. 9

    def forward(self, pyramid: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        p3, p4, p5 = pyramid["P3"], pyramid["P4"], pyramid["P5"]

        n3 = p3                                  # carried forward unchanged
        n4 = self.gate4(n3, p4)
        n5 = self.gate5(n4, p5)
        n6 = self.downsample6(n5)                # Eq. 9, no gate

        return {"n3": n3, "n4": n4, "n5": n5, "n6": n6}
