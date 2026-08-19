"""Top-down Feature Pyramid Network (Section 3.3.2, Eqs. 2-5).

Lateral 1x1 convolutions project {C3, C4, C5} to a common channel width D,
a top-down pathway with nearest-neighbour upsampling fuses coarse semantic
features into finer levels, and a 3x3 conv smooths each fused map.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopDownFPN(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        """
        Args:
            in_channels: channel counts for [C3, C4, C5] in that order.
            out_channels: common pyramid width D.
        """
        super().__init__()
        c3, c4, c5 = in_channels

        # Eq. 2: lateral 1x1 projections
        self.lat3 = nn.Conv2d(c3, out_channels, kernel_size=1)
        self.lat4 = nn.Conv2d(c4, out_channels, kernel_size=1)
        self.lat5 = nn.Conv2d(c5, out_channels, kernel_size=1)

        # Eq. 5: post-fusion smoothing convs
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    @staticmethod
    def _upsample_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Up(P_{k+1}) + L_k with nearest-neighbour upsampling (Eq. 4)."""
        up = F.interpolate(x, size=y.shape[-2:], mode="nearest")
        return up + y

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        c3, c4, c5 = features["C3"], features["C4"], features["C5"]

        l3, l4, l5 = self.lat3(c3), self.lat4(c4), self.lat5(c5)

        p5 = l5                                   # Eq. 3
        p4 = self._upsample_add(p5, l4)            # Eq. 4, k=4
        p3 = self._upsample_add(p4, l3)            # Eq. 4, k=3

        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)

        return {"P3": p3, "P4": p4, "P5": p5}
