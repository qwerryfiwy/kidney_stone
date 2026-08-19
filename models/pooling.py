"""Feature-aggregation / pooling strategies (Section 3.3.4 and the four
pooling variants compared in Table 2/5/8: Avg, GMP, Dual, and the proposed
multi-level attention-gated Max).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


def _global_max(x: torch.Tensor) -> torch.Tensor:
    return torch.amax(x, dim=(-2, -1))  # Eq. 10


def _global_avg(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x, dim=(-2, -1))


class MultiLevelMaxPool(nn.Module):
    """Applies global max pooling independently to each pyramid level and
    concatenates the results (Eq. 10-11). Used by the full ENFM model on
    the 4-level attention-gated pyramid {n3, n4, n5, n6}.
    """

    def __init__(self, level_keys: List[str] = ("n3", "n4", "n5", "n6")):
        super().__init__()
        self.level_keys = list(level_keys)

    def forward(self, pyramid: Dict[str, torch.Tensor]) -> torch.Tensor:
        pooled = [_global_max(pyramid[k]) for k in self.level_keys]
        return torch.cat(pooled, dim=1)  # Eq. 11


class MultiLevelAvgPool(nn.Module):
    """Ablation: average pooling in place of max pooling at each level."""

    def __init__(self, level_keys: List[str] = ("n3", "n4", "n5", "n6")):
        super().__init__()
        self.level_keys = list(level_keys)

    def forward(self, pyramid: Dict[str, torch.Tensor]) -> torch.Tensor:
        pooled = [_global_avg(pyramid[k]) for k in self.level_keys]
        return torch.cat(pooled, dim=1)


class MultiLevelDualPool(nn.Module):
    """Ablation: concatenated average + max pooling at each level."""

    def __init__(self, level_keys: List[str] = ("n3", "n4", "n5", "n6")):
        super().__init__()
        self.level_keys = list(level_keys)

    def forward(self, pyramid: Dict[str, torch.Tensor]) -> torch.Tensor:
        pooled = []
        for k in self.level_keys:
            x = pyramid[k]
            pooled.append(torch.cat([_global_avg(x), _global_max(x)], dim=1))
        return torch.cat(pooled, dim=1)


class SingleLevelGlobalMaxPool(nn.Module):
    """Ablation: global max pooling applied only to the finest fused
    top-down level (no bottom-up refinement, no multi-level fusion) —
    the "EfficientNet-FPN-GMP" baseline in the paper.
    """

    def __init__(self, level_key: str = "P3"):
        super().__init__()
        self.level_key = level_key

    def forward(self, pyramid: Dict[str, torch.Tensor]) -> torch.Tensor:
        return _global_max(pyramid[self.level_key])


POOLING_REGISTRY = {
    "max": MultiLevelMaxPool,
    "avg": MultiLevelAvgPool,
    "dual": MultiLevelDualPool,
    "gmp_single": SingleLevelGlobalMaxPool,
}


def build_pooling(name: str, **kwargs) -> nn.Module:
    if name not in POOLING_REGISTRY:
        raise ValueError(f"Unknown pooling '{name}'. Choices: {list(POOLING_REGISTRY)}")
    return POOLING_REGISTRY[name](**kwargs)
