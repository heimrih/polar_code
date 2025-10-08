"""Layered normalized min-sum LDPC decoder."""

from __future__ import annotations

import numpy as np


def decode_ldpc_nms(
    llr: np.ndarray,
    H: np.ndarray,
    max_iter: int = 20,
    alpha: float = 0.8,
    early_stop: bool = True,
) -> dict:
    m, n = H.shape
    if llr.size != n:
        raise ValueError("llr length mismatch")

    llr = llr.astype(np.float64)
    hard = (llr < 0).astype(np.int8)
    msg = np.zeros_like(H, dtype=np.float64)

    for it in range(1, max_iter + 1):
        for r in range(m):
            idx = np.where(H[r] == 1)[0]
            if idx.size == 0:
                continue
            llr_ext = llr[idx] - msg[r, idx]
            sign = np.prod(np.sign(llr_ext))
            magnitude = np.min(np.abs(llr_ext))
            update = alpha * sign * magnitude
            msg[r, idx] = update
            llr[idx] = llr_ext + update

        hard = (llr < 0).astype(np.int8)
        syndrome = (H @ hard) % 2
        if early_stop and not syndrome.any():
            return {"hard": hard, "iters_used": it, "parity_ok": True}

    return {"hard": hard, "iters_used": max_iter, "parity_ok": not (H @ hard % 2).any()}


__all__ = ["decode_ldpc_nms"]
