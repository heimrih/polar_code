"""CRC-24 utilities for polar coding."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _poly_to_bits(poly: str) -> np.ndarray:
    if not poly:
        raise ValueError("CRC polynomial string must be non-empty")
    value = int(poly, 16)
    bit_length = value.bit_length()
    bits = [(value >> i) & 1 for i in reversed(range(bit_length))]
    return np.array(bits, dtype=np.int8)


def attach_crc(msg_bits: np.ndarray, poly: str) -> np.ndarray:
    """Append CRC parity bits to `msg_bits` using the given hex polynomial."""

    if msg_bits.ndim != 1:
        raise ValueError("msg_bits must be a 1D array")
    msg_bits = (msg_bits.astype(np.int8) & 1)
    poly_bits = _poly_to_bits(poly)
    degree = poly_bits.size - 1
    if degree <= 0:
        raise ValueError("Polynomial degree must be positive")

    buffer = np.concatenate([msg_bits, np.zeros(degree, dtype=np.int8)])
    poly_vec = poly_bits
    for i in range(msg_bits.size):
        if buffer[i] == 0:
            continue
        buffer[i : i + degree + 1] ^= poly_vec
    remainder = buffer[-degree:]
    return np.concatenate([msg_bits, remainder])


def check_crc(msg_with_crc: np.ndarray, poly: str) -> bool:
    """Return True if `msg_with_crc` satisfies the CRC checksum."""

    if msg_with_crc.ndim != 1:
        raise ValueError("msg_with_crc must be a 1D array")
    msg_with_crc = (msg_with_crc.astype(np.int8) & 1)
    poly_bits = _poly_to_bits(poly)
    degree = poly_bits.size - 1
    if msg_with_crc.size <= degree:
        raise ValueError("Message too short for the provided CRC polynomial")

    buffer = msg_with_crc.copy()
    for i in range(msg_with_crc.size - degree):
        if buffer[i] == 0:
            continue
        buffer[i : i + degree + 1] ^= poly_bits
    return not buffer[-degree:].any()


__all__ = ["attach_crc", "check_crc"]
