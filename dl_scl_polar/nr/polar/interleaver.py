"""5G NR sub-block interleaver implementation."""

from __future__ import annotations

import numpy as np

_INTERLEAVER_BLOCK = 32


def subblock_interleave(bits: np.ndarray, mode: str = "default") -> np.ndarray:
    """Apply NR sub-block interleaver."""

    if bits.ndim != 1:
        raise ValueError("bits must be 1D")
    block = _INTERLEAVER_BLOCK
    num_blocks = (bits.size + block - 1) // block
    padded = np.full(num_blocks * block, fill_value=-1, dtype=bits.dtype)
    padded[: bits.size] = bits

    order = np.array([((i % block) * num_blocks) + (i // block) for i in range(padded.size)], dtype=np.int32)
    interleaved = padded[order]

    return interleaved


def subblock_deinterleave(bits: np.ndarray, original_len: int, mode: str = "default") -> np.ndarray:
    if bits.ndim != 1:
        raise ValueError("bits must be 1D")
    block = _INTERLEAVER_BLOCK
    num_blocks = (original_len + block - 1) // block
    order = np.array([((i % block) * num_blocks) + (i // block) for i in range(num_blocks * block)], dtype=np.int32)
    total = num_blocks * block
    padded = np.zeros(total, dtype=bits.dtype)
    padded[: bits.size] = bits
    deinterleaved = np.zeros(total, dtype=bits.dtype)
    deinterleaved[:total] = padded[np.argsort(order)]
    return deinterleaved[:original_len]


__all__ = ["subblock_interleave", "subblock_deinterleave"]
