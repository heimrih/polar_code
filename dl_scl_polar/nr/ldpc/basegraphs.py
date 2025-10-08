"""Base graph definitions for 5G NR LDPC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class BaseGraph:
    name: str
    m: int  # rows in base graph
    n: int  # columns in base graph
    shifts: np.ndarray  # shape (m, n) with -1 meaning zero block


def _create_demo_bg(name: str) -> BaseGraph:
    # Minimal 3x6 base graph: first 3 columns payload; last 3 parity (identity)
    shifts = np.array(
        [
            [0,  1,  2,  0, -1, -1],
            [1,  0,  3, -1,  0, -1],
            [2,  3,  0, -1, -1,  0],
        ],
        dtype=np.int32,
    )
    return BaseGraph(name=name, m=3, n=6, shifts=shifts)


_BG_CACHE: Dict[int, BaseGraph] = {
    1: _create_demo_bg("BG_demo1"),
    2: _create_demo_bg("BG_demo2"),
}


def load_base_graph(bg: int) -> BaseGraph:
    if bg not in _BG_CACHE:
        raise ValueError(f"Unknown base graph: {bg}")
    return _BG_CACHE[bg]


__all__ = ["BaseGraph", "load_base_graph"]
