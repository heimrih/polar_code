"""Simplified NR LDPC rate matching (puncture/repeat)."""

from __future__ import annotations

import numpy as np


def rate_match_ldpc(codeword: np.ndarray, E: int) -> np.ndarray:
    if codeword.ndim != 1:
        raise ValueError("codeword must be 1D")
    N = codeword.size
    if E <= N:
        return codeword[:E]
    reps = (E + N - 1) // N
    return np.tile(codeword, reps)[:E]


def derate_match_ldpc(llr: np.ndarray, N: int) -> np.ndarray:
    if llr.ndim != 1:
        raise ValueError("llr must be 1D")
    if llr.size <= N:
        result = np.zeros(N, dtype=np.float64)
        result[: llr.size] = llr
        return result
    reps = llr.size // N
    remainder = llr.size % N
    accum = np.zeros(N, dtype=np.float64)
    count = np.zeros(N, dtype=np.int32)
    if reps > 0:
        shaped = llr[: reps * N].reshape(reps, N)
        accum += shaped.sum(axis=0)
        count += reps
    if remainder:
        start = reps * N
        accum[:remainder] += llr[start : start + remainder]
        count[:remainder] += 1
    count[count == 0] = 1
    return accum / count


__all__ = ["rate_match_ldpc", "derate_match_ldpc"]
