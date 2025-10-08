"""Rate-matched polar SCL wrappers following 5G NR flow (simplified)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ...polar.crc import attach_crc, check_crc
from ...polar.polar import construct_info_set
from ...polar.scl import decode_scl
from ...polar import polar as polar_core
from .interleaver import subblock_interleave, subblock_deinterleave
from .rate_match import rate_match_polar, derate_match_polar


def _polar_encode(info_bits: np.ndarray, info_set: np.ndarray, N: int) -> np.ndarray:
    u = np.zeros(N, dtype=np.int8)
    u[info_set] = info_bits
    return polar_core._polar_transform(u)


def encode_rate_matched(
    payload_bits: np.ndarray,
    crc_poly: str,
    N: int,
    E: int,
    info_set: np.ndarray,
    ilv_mode: str = "default",
) -> np.ndarray:
    msg = attach_crc(payload_bits, crc_poly)
    codeword = _polar_encode(msg, info_set, N)
    ilv = subblock_interleave(codeword, mode=ilv_mode)
    rate_matched = rate_match_polar(ilv, E)
    return rate_matched


def decode_rate_matched_scl(
    llr_E: np.ndarray,
    crc_poly: str,
    N: int,
    E: int,
    info_set: np.ndarray,
    M: int,
    ilv_mode: str = "default",
) -> Dict[str, np.ndarray]:
    llr_internal = derate_match_polar(llr_E, N)
    llr_internal = subblock_deinterleave(llr_internal, N, mode=ilv_mode)

    result = decode_scl(llr_internal, info_set, M=M, crc=crc_poly)
    bits = result.get("best_path_bits")
    payload = bits[: len(info_set)] if bits is not None else None
    return {
        "payload": payload,
        "crc_pass": bits is not None and check_crc(bits, crc_poly),
        "best_path_bits": bits,
    }


__all__ = ["encode_rate_matched", "decode_rate_matched_scl"]
