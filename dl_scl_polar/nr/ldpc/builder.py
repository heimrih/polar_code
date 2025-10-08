"""Build lifted parity-check matrices from base graphs."""

from __future__ import annotations

import numpy as np

from .basegraphs import BaseGraph


def _circulant(size: int, shift: int) -> np.ndarray:
    mat = np.zeros((size, size), dtype=np.int8)
    if shift < 0:
        return mat
    shift = shift % size
    for i in range(size):
        mat[i, (i + shift) % size] = 1
    return mat


def build_h_matrix(base_graph: BaseGraph, Z: int) -> np.ndarray:
    rows = []
    for r in range(base_graph.m):
        row_blocks = []
        for c in range(base_graph.n):
            shift = base_graph.shifts[r, c]
            block = _circulant(Z, shift)
            row_blocks.append(block)
        rows.append(np.hstack(row_blocks))
    H = np.vstack(rows)
    return H


__all__ = ["build_h_matrix"]
