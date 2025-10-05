"""Bit-flip retry mechanics for DL-SCL."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..polar.crc import check_crc
from ..polar.scl import decode_scl


def choose_flip_index(abs_l0: np.ndarray, beta: Optional[np.ndarray]) -> int:
    """Choose flip index using β metric (fallback to |L0| ordering)."""

    if abs_l0.ndim != 1:
        raise ValueError("abs_l0 must be a 1D array")
    if abs_l0.size == 0:
        raise ValueError("abs_l0 cannot be empty")

    if beta is not None:
        if beta.ndim != 2 or beta.shape[0] != beta.shape[1] or beta.shape[0] != abs_l0.size:
            raise ValueError("beta must be a square matrix matching abs_l0 length")
        q = abs_l0 @ beta
        return int(np.argmin(q))

    return int(np.argmin(abs_l0))


def _force_vector(best_path_bits: np.ndarray, flip_index: int) -> np.ndarray:
    forced = np.full(best_path_bits.size, -1, dtype=np.int8)
    forced[:flip_index] = best_path_bits[:flip_index]
    forced[flip_index] = 1 - best_path_bits[flip_index]
    return forced


def retry_with_flip(
    llr_root: np.ndarray,
    info_set: np.ndarray,
    M: int,
    best_path_bits: np.ndarray,
    flip_index: int,
    crc: Optional[str] = None,
) -> dict:
    """Retry SCL decoding by flipping the specified info bit."""

    if best_path_bits.ndim != 1:
        raise ValueError("best_path_bits must be 1D")
    if flip_index < 0 or flip_index >= best_path_bits.size:
        raise IndexError("flip_index out of range")

    forced = _force_vector(best_path_bits, flip_index)
    result = decode_scl(
        llr_root,
        info_set,
        M,
        crc=crc,
        force_info_bits=forced,
    )
    result["forced_info_bits"] = forced
    result["flip_index"] = flip_index
    return result


def decode_with_retries(
    llr_root: np.ndarray,
    info_set: np.ndarray,
    M: int,
    retries: int,
    *,
    crc: Optional[str] = None,
    beta: Optional[np.ndarray] = None,
) -> dict:
    """Run baseline SCL followed by up to ``retries`` flip attempts."""

    attempts: List[dict] = []
    baseline = decode_scl(llr_root, info_set, M, crc=crc)
    attempts.append({**baseline, "attempt_type": "baseline"})

    best_output = baseline

    def _passes(output: dict) -> bool:
        if crc is None:
            return output.get("best_path_bits") is not None
        bits = output.get("best_path_bits")
        if bits is None:
            return False
        return check_crc(bits, crc)

    if _passes(baseline) or retries <= 0:
        final = {**best_output}
        final["attempts"] = attempts
        final["tried_indices"] = []
        final["success"] = _passes(best_output)
        return final

    reference_bits = baseline.get("best_path_bits")
    reference_llrs = baseline.get("best_path_info_llrs")
    if reference_bits is None or reference_llrs is None:
        raise ValueError("Baseline decode did not produce candidate bits/LLRs")

    abs_l0 = np.abs(np.asarray(reference_llrs, dtype=float))

    def rank_indices(abs_l0_vec: np.ndarray, beta_mat: Optional[np.ndarray]) -> List[int]:
        if beta_mat is not None:
            q = abs_l0_vec @ beta_mat
            return list(np.argsort(q))
        return list(np.argsort(abs_l0_vec))

    tried: List[int] = []
    while len(tried) < retries and len(tried) < abs_l0.size:
        order = rank_indices(abs_l0, beta)
        idx = next((i for i in order if i not in tried), None)
        if idx is None:
            break
        tried.append(idx)
        retry_result = retry_with_flip(
            llr_root,
            info_set,
            M,
            reference_bits,
            flip_index=idx,
            crc=crc,
        )
        attempts.append({**retry_result, "attempt_type": "flip"})
        best_output = retry_result
        new_bits = retry_result.get("best_path_bits")
        new_llrs = retry_result.get("best_path_info_llrs")
        if new_bits is not None:
            reference_bits = new_bits
        if new_llrs is not None:
            reference_llrs = new_llrs
        abs_l0 = np.abs(np.asarray(reference_llrs, dtype=float))
        if _passes(retry_result):
            break

    final = {**best_output}
    final["attempts"] = attempts
    final["tried_indices"] = tried
    final["success"] = _passes(best_output)
    return final


__all__ = ["choose_flip_index", "retry_with_flip", "decode_with_retries"]
