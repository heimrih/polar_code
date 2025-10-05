"""Central configuration defaults for dl_scl_polar."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PolarConfig:
    N: int = 128
    K: int = 64
    crc_poly: str = "0x1864CFB"  # 5G CRC-24
    crc_bits: int = 24
    list_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    retries: int = 8
    ebno_sweep: List[float] = field(default_factory=lambda: [4.0, 6.5, 0.5])
    seed: int = 0


DEFAULTS = PolarConfig()


def get_config() -> PolarConfig:
    """Return a copy of the default configuration."""

    return PolarConfig(**DEFAULTS.__dict__)
