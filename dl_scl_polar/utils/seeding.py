"""Deterministic seeding helpers."""

from __future__ import annotations

import os

# Limit OpenMP usage before importing heavy libraries.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import random

import torch

# Import numpy after torch to keep Intel OMP happy in constrained envs.
import numpy as np

# Force single-threaded torch to avoid shared-memory issues.
torch.set_num_threads(1)


def seed_all(seed: int, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True)


__all__ = ["seed_all"]
