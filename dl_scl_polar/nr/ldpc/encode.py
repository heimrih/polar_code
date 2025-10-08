"""Systematic LDPC encoder for small NR-esque matrices."""

from __future__ import annotations

import numpy as np


def _gauss_solve_gf2(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = (A.copy() % 2).astype(np.uint8)
    b = (b.copy() % 2).astype(np.uint8)
    m, n = A.shape
    x = np.zeros(n, dtype=np.uint8)
    row = 0
    pivot_rows = {}
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if A[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
            b[[row, pivot]] = b[[pivot, row]]
        pivot_rows[col] = row
        for r in range(m):
            if r != row and A[r, col]:
                A[r] ^= A[row]
                b[r] ^= b[row]
        row += 1
        if row == m:
            break

    for r in range(row, m):
        if not A[r].any() and b[r]:
            raise ValueError("Linear system over GF(2) has no solution")

    for col, r in sorted(pivot_rows.items(), reverse=True):
        val = b[r]
        for c in range(col + 1, n):
            if A[r, c]:
                val ^= x[c]
        x[col] = val
    return x


def encode_ldpc(payload: np.ndarray, H: np.ndarray) -> np.ndarray:
    if payload.ndim != 1:
        raise ValueError("payload must be 1D")
    m, n = H.shape
    k = payload.size
    if n <= k:
        raise ValueError("Parity-check matrix too small for payload length")

    systematic = payload.astype(np.uint8)
    H_sys = (H[:, :k] % 2).astype(np.uint8)
    H_par = (H[:, k:] % 2).astype(np.uint8)

    syndrome = (H_sys @ systematic) % 2
    parity = _gauss_solve_gf2(H_par, syndrome)
    codeword = np.concatenate([systematic, parity]).astype(np.int8)
    return codeword


__all__ = ["encode_ldpc"]
