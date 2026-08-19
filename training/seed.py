"""Seeding helper for reproducible multi-seed protocols (Section 3.5:
20 seeds for Subset 1, seeds 42/123/999 for the CV protocol)."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
