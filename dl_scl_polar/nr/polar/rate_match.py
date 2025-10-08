"""NR polar rate matching utilities."""

from __future__ import annotations

import numpy as np


def rate_match_polar(bits: np.ndarray, E: int, mode: str = "puncture") -> np.ndarray:
    if bits.ndim != 1:
        raise ValueError("bits must be 1D")
    N = bits.size
    if E <= N:
        return bits[:E]
    reps = (E + N - 1) // N
    tiled = np.tile(bits, reps)
    return tiled[:E]


def derate_match_polar(bits_E: np.ndarray, N: int, mode: str = "puncture") -> np.ndarray:
    if bits_E.ndim != 1:
        raise ValueError("bits_E must be 1D")
    if bits_E.size <= N:
        result = np.full(N, fill_value=-1.0, dtype=np.float64)
        result[: bits_E.size] = bits_E
        return result
    reps = bits_E.size // N
    remainder = bits_E.size % N
    accum = np.zeros(N, dtype=np.float64)
    counts = np.zeros(N, dtype=np.int32)
    if reps > 0:
        shaped = bits_E[: reps * N].reshape(reps, N)
        accum += shaped.sum(axis=0)
        counts += reps
    if remainder:
        start = reps * N
        accum[:remainder] += bits_E[start : start + remainder]
        counts[:remainder] += 1
    counts[counts == 0] = 1
    return accum / counts


__all__ = ["rate_match_polar", "derate_match_polar"]
