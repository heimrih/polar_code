"""Successive-cancellation list (SCL) decoder with optional CRC filtering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .polar import _f as f_fn  # reuse combine helpers
from .polar import _g as g_fn
from .crc import check_crc


@dataclass
class _PathState:
    llr_layers: List[List[np.ndarray]]
    bit_layers: List[List[np.ndarray]]
    u: np.ndarray
    metric: float
    n: int
    info_llrs: List[float]

    @classmethod
    def create(cls, channel_llr: np.ndarray) -> "_PathState":
        N = channel_llr.size
        n = int(math.log2(N))
        if 1 << n != N:
            raise ValueError("Channel LLR length must be a power of two")

        llr_layers: List[List[np.ndarray]] = []
        bit_layers: List[List[np.ndarray]] = []
        for level in range(n + 1):
            nodes = 1 << level
            block = 1 << (n - level)
            llr_level = [np.zeros(block, dtype=float) for _ in range(nodes)]
            bit_level = [np.zeros(block, dtype=np.int8) for _ in range(nodes)]
            llr_layers.append(llr_level)
            bit_layers.append(bit_level)
        llr_layers[0][0][:] = channel_llr.astype(float)
        u = np.zeros(N, dtype=np.int8)
        return cls(
            llr_layers=llr_layers,
            bit_layers=bit_layers,
            u=u,
            metric=0.0,
            n=n,
            info_llrs=[],
        )

    def clone(self) -> "_PathState":
        llr_layers = [[node.copy() for node in level] for level in self.llr_layers]
        bit_layers = [[node.copy() for node in level] for level in self.bit_layers]
        return _PathState(
            llr_layers=llr_layers,
            bit_layers=bit_layers,
            u=self.u.copy(),
            metric=self.metric,
            n=self.n,
            info_llrs=self.info_llrs.copy(),
        )

    def _ensure_alpha(self, level: int, node: int) -> None:
        if level == 0:
            return
        parent = node // 2
        self._ensure_alpha(level - 1, parent)
        parent_vec = self.llr_layers[level - 1][parent]
        half = parent_vec.size // 2
        left = parent_vec[:half]
        right = parent_vec[half:]
        target = self.llr_layers[level][node]
        if node % 2 == 0:
            np.copyto(target, f_fn(left, right))
        else:
            left_bits = self.bit_layers[level][node - 1]
            np.copyto(target, g_fn(left, right, left_bits))

    def llr_for_phase(self, phase: int) -> float:
        self._ensure_alpha(self.n, phase)
        return float(self.llr_layers[self.n][phase][0])

    def set_bit(self, phase: int, bit: int) -> None:
        bit = int(bit) & 1
        self.u[phase] = bit
        self.bit_layers[self.n][phase][0] = bit
        level = self.n
        node = phase
        while level > 0 and node % 2 == 1:
            parent = node // 2
            left = self.bit_layers[level][node - 1]
            right = self.bit_layers[level][node]
            parent_vec = self.bit_layers[level - 1][parent]
            half = parent_vec.size // 2
            parent_vec[:half] = np.bitwise_xor(left, right)
            parent_vec[half:] = right
            node = parent
            level -= 1


def _update_metric(metric: float, llr_value: float, bit: int) -> float:
    if bit:
        return metric + float(np.logaddexp(0.0, llr_value))
    return metric + float(np.logaddexp(0.0, -llr_value))


def decode_scl(
    llr: np.ndarray,
    info_set: np.ndarray,
    M: int,
    crc: Optional[str] = None,
    *,
    force_info_bits: Optional[np.ndarray] = None,
) -> dict:
    """Decode using SCL and optionally filter candidates with CRC."""

    if M <= 0:
        raise ValueError("List size M must be positive")
    N = llr.size
    if info_set.ndim != 1:
        raise ValueError("info_set must be a 1D array")
    info_mask = np.zeros(N, dtype=bool)
    info_mask[info_set] = True

    if force_info_bits is not None:
        if force_info_bits.ndim != 1:
            raise ValueError("force_info_bits must be 1D when provided")
        if force_info_bits.size != info_set.size:
            raise ValueError("force_info_bits length must match info_set")
        force_info_bits = force_info_bits.astype(np.int8)

    info_index = 0

    paths: List[_PathState] = [_PathState.create(llr)]
    for phase in range(N):
        frozen = not info_mask[phase]
        forced_choice: Optional[int] = None
        if info_mask[phase] and force_info_bits is not None:
            val = int(force_info_bits[info_index])
            if val in (0, 1):
                forced_choice = val
            elif val != -1:
                raise ValueError("force_info_bits entries must be -1, 0, or 1")

        new_paths: List[_PathState] = []
        for path in paths:
            llr_value = path.llr_for_phase(phase)
            if frozen:
                path.metric = _update_metric(path.metric, llr_value, 0)
                path.set_bit(phase, 0)
                new_paths.append(path)
                continue

            choices = (forced_choice,) if forced_choice is not None else (0, 1)
            if len(choices) == 1:
                bit = choices[0]
                path.metric = _update_metric(path.metric, llr_value, bit)
                path.set_bit(phase, bit)
                path.info_llrs = path.info_llrs + [float(llr_value)]
                new_paths.append(path)
            else:
                for bit in choices:
                    branch = path.clone()
                    branch.metric = _update_metric(branch.metric, llr_value, bit)
                    branch.set_bit(phase, bit)
                    branch.info_llrs.append(float(llr_value))
                    new_paths.append(branch)
        if info_mask[phase]:
            info_index += 1
        if len(new_paths) == 0:
            raise RuntimeError("All paths pruned during decoding")
        new_paths.sort(key=lambda p: p.metric)
        paths = new_paths[:M]

    candidates: List[np.ndarray] = []
    metrics: List[float] = []
    info_llrs_list: List[np.ndarray] = []
    best_path_bits: Optional[np.ndarray] = None
    best_path_info_llrs: Optional[np.ndarray] = None
    best_index: Optional[int] = None

    for idx, path in enumerate(paths):
        info_bits = path.u[info_set].copy()
        candidates.append(info_bits)
        metrics.append(path.metric)
        info_llrs = np.asarray(path.info_llrs, dtype=float)
        info_llrs_list.append(info_llrs)

    if crc is not None:
        for idx, info_bits in enumerate(candidates):
            if check_crc(info_bits, crc):
                best_index = idx
                break

    if best_index is None and candidates:
        best_index = 0

    if best_index is not None:
        best_path_bits = candidates[best_index]
        best_path_info_llrs = info_llrs_list[best_index]

    return {
        "candidates": candidates,
        "metrics": metrics,
        "best_path_bits": best_path_bits,
        "info_llrs": info_llrs_list,
        "best_path_info_llrs": best_path_info_llrs,
    }


__all__ = ["decode_scl"]
