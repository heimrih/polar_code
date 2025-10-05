"""Polar code core primitives: construction, encoding, and SC decoding."""

from __future__ import annotations

import functools
import math
from typing import Iterable

import numpy as np

from .. import config

# ------------------------------
# Helper functions
# ------------------------------

def _polar_transform(u: np.ndarray) -> np.ndarray:
    """Apply the Arikan polar transform (non-systematic)."""

    n = int(math.log2(u.size))
    x = u.copy()
    for stage in range(n):
        step = 1 << stage
        block = step << 1
        for start in range(0, x.size, block):
            left = slice(start, start + step)
            right = slice(start + step, start + block)
            x[left] ^= x[right]
    return x


def _check_power_of_two(n: int) -> None:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("N must be a power of two")


def _polarization_weights(N: int) -> np.ndarray:
    n = int(math.log2(N))
    weights = np.zeros(N, dtype=float)
    for idx in range(N):
        w = 0.0
        bits = idx
        for j in range(n):
            if bits & 1:
                w += 2 ** (j / 4.0)
            bits >>= 1
        weights[idx] = w
    return weights


def _phi_inv(x: float) -> float:
    if x > 12.0:
        return 0.9861 * x - 2.3152
    if x > 3.5:
        return x * (0.009005 * x + 0.7694) - 0.9507
    if x > 1.0:
        return x * (0.062883 * x + 0.3678) - 0.1627
    return x * (0.2202 * x + 0.06448)


def _gaussian_pe(N: int, K: int, design_snr_db: float) -> np.ndarray:
    rate = K / N
    snr = 10 ** (design_snr_db / 10.0)
    sigma_sq = 1.0 / (2.0 * rate * snr)

    m = np.zeros(N, dtype=float)
    m[0] = 2.0 / sigma_sq
    stages = int(math.log2(N))
    for level in range(1, stages + 1):
        B = 1 << level
        half = B >> 1
        for j in range(half):
            T = m[j]
            m[j] = _phi_inv(T)
            m[half + j] = 2.0 * T

    # Convert mean LLR to error probability via Q-function approximation.
    pe = np.zeros_like(m)
    for i in range(N):
        val = max(m[i], 1e-12)
        pe[i] = 0.5 - 0.5 * math.erf(math.sqrt(val) / 2.0)
    return pe


@functools.lru_cache(maxsize=None)
def construct_info_set(N: int, K: int, method: str = "gaussian", design_snr_db: float = 2.5) -> np.ndarray:
    """Return sorted indices of the information set for an (N, K) polar code."""

    _check_power_of_two(N)
    if not (0 < K <= N):
        raise ValueError("K must satisfy 0 < K <= N")

    if method == "polarization":
        metric = _polarization_weights(N)
        order = np.argsort(metric, kind="stable")
    elif method == "gaussian":
        pe = _gaussian_pe(N, K, design_snr_db)
        order = np.argsort(pe, kind="stable")
    else:
        raise ValueError(f"Unsupported construction method: {method}")

    info_idx = np.sort(order[:K])
    return info_idx.astype(np.int32)


def encode(msg_bits: np.ndarray) -> np.ndarray:
    """Encode an information vector using the default polar code."""

    cfg = config.DEFAULTS
    if msg_bits.ndim != 1:
        raise ValueError("msg_bits must be 1D")
    if msg_bits.size != cfg.K:
        raise ValueError(f"msg_bits must have length {cfg.K}")

    info_set = construct_info_set(cfg.N, cfg.K)
    u = np.zeros(cfg.N, dtype=np.int8)
    u[info_set] = msg_bits.astype(np.int8) & 1
    x = _polar_transform(u)
    return x


def _f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def _g(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return b + (1 - 2 * c) * a


def sc_decode(llr: np.ndarray, info_set: np.ndarray) -> np.ndarray:
    """Perform SC decoding and return the estimated information bits."""

    if llr.ndim != 1:
        raise ValueError("llr must be 1D")
    N = llr.size
    _check_power_of_two(N)
    if info_set.ndim != 1:
        raise ValueError("info_set must be 1D")
    if np.any(info_set < 0) or np.any(info_set >= N):
        raise ValueError("info_set indices out of range")

    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_set] = False
    u_hat = np.zeros(N, dtype=np.int8)

    def _decode_segment(segment_llr: np.ndarray, depth: int, start: int) -> np.ndarray:
        if depth == 0:
            idx = start
            if frozen_mask[idx]:
                bit = 0
            else:
                bit = np.int8(segment_llr[0] < 0)
            u_hat[idx] = bit
            return np.array([bit], dtype=np.int8)

        half = segment_llr.size // 2
        left = _f(segment_llr[:half], segment_llr[half:])
        left_bits = _decode_segment(left, depth - 1, start)

        right = _g(segment_llr[:half], segment_llr[half:], left_bits)
        right_bits = _decode_segment(right, depth - 1, start + half)

        combined = np.concatenate([(left_bits ^ right_bits), right_bits])
        return combined

    n = int(math.log2(N))
    _decode_segment(llr.astype(float), n, 0)
    return u_hat[info_set]


__all__ = [
    "construct_info_set",
    "encode",
    "sc_decode",
]
