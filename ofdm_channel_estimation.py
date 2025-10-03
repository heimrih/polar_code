"""Simple OFDM channel estimation demo using least-squares pilots.

The module mirrors the lightweight configuration approach used elsewhere in
this repository: tweak the ``CONFIG`` object below and run the file directly
to execute a small Monte-Carlo experiment.  The simulation generates random
BPSK OFDM symbols, passes them through a frequency-selective Rayleigh fading
channel, and recovers the data using least-squares channel estimation on
comb-type pilots with linear interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class OFDMSimulationConfig:
    """Container for the OFDM channel estimation experiment parameters."""

    num_subcarriers: int = 64
    pilot_spacing: int = 4
    num_ofdm_symbols: int = 1000
    snr_db: float = 15.0
    channel_taps: int = 8
    seed: int | None = 0

    def pilot_indices(self) -> np.ndarray:
        """Return the pilot subcarrier indices for a comb-type pattern."""

        pilots = np.arange(0, self.num_subcarriers, self.pilot_spacing)
        if pilots[-1] != self.num_subcarriers - 1:
            pilots = np.append(pilots, self.num_subcarriers - 1)
        return pilots


CONFIG = OFDMSimulationConfig()


def generate_bpsk_symbols(size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate equiprobable BPSK symbols with unit average energy."""

    bits = rng.integers(0, 2, size=size)
    return 1 - 2 * bits  # Maps {0, 1} -> {+1, -1}


def rayleigh_frequency_response(
    num_subcarriers: int, channel_taps: int, rng: np.random.Generator
) -> np.ndarray:
    """Create a random frequency-selective channel frequency response."""

    taps = (
        rng.normal(size=channel_taps) + 1j * rng.normal(size=channel_taps)
    ) / np.sqrt(2 * channel_taps)
    impulse_response = np.zeros(num_subcarriers, dtype=np.complex128)
    impulse_response[:channel_taps] = taps
    return np.fft.fft(impulse_response)


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex additive white Gaussian noise to the input signal."""

    symbol_energy = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_variance = symbol_energy / snr_linear
    noise = (
        rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape)
    ) * np.sqrt(noise_variance / 2.0)
    return signal + noise


def ls_channel_estimate(
    transmitted: np.ndarray,
    received: np.ndarray,
    pilot_indices: np.ndarray,
) -> np.ndarray:
    """Perform least-squares channel estimation and interpolate pilots."""

    eps = 1e-12
    tx_pilots = transmitted[pilot_indices]
    rx_pilots = received[pilot_indices]
    safe_tx = np.where(np.abs(tx_pilots) < eps, eps, tx_pilots)
    pilot_estimates = rx_pilots / safe_tx

    all_indices = np.arange(transmitted.size)
    real_interp = np.interp(all_indices, pilot_indices, pilot_estimates.real)
    imag_interp = np.interp(all_indices, pilot_indices, pilot_estimates.imag)
    return real_interp + 1j * imag_interp


def simulate(config: OFDMSimulationConfig) -> Tuple[float, float]:
    """Run the OFDM channel estimation Monte-Carlo and return metrics.

    Returns
    -------
    Tuple[float, float]
        ``(channel_mse, ber)`` averaged over all simulated OFDM symbols.
    """

    if config.num_subcarriers < 2:
        raise ValueError("num_subcarriers must be at least 2")
    if config.pilot_spacing < 1:
        raise ValueError("pilot_spacing must be positive")

    rng = np.random.default_rng(config.seed)
    pilot_indices = config.pilot_indices()

    mse_accum = 0.0
    bit_errors = 0
    total_bits = 0

    for _ in range(config.num_ofdm_symbols):
        transmitted = generate_bpsk_symbols(config.num_subcarriers, rng)
        transmitted = transmitted.astype(np.complex128)

        transmitted[pilot_indices] = generate_bpsk_symbols(pilot_indices.size, rng)

        channel = rayleigh_frequency_response(
            config.num_subcarriers, config.channel_taps, rng
        )
        received = channel * transmitted
        received = add_awgn(received, config.snr_db, rng)

        estimated_channel = ls_channel_estimate(transmitted, received, pilot_indices)
        mse_accum += np.mean(np.abs(estimated_channel - channel) ** 2)

        safe_est = np.where(np.abs(estimated_channel) < 1e-12, 1e-12, estimated_channel)
        equalized = received / safe_est
        decisions = np.sign(equalized.real)
        bits = (transmitted.real < 0).astype(int)
        detected_bits = (decisions < 0).astype(int)

        bit_errors += np.count_nonzero(bits != detected_bits)
        total_bits += bits.size

    channel_mse = mse_accum / config.num_ofdm_symbols
    ber = bit_errors / total_bits
    return channel_mse, ber


def main(config: OFDMSimulationConfig = CONFIG) -> None:
    channel_mse, ber = simulate(config)
    print("OFDM LS Channel Estimation Results")
    print(f"  Num subcarriers       : {config.num_subcarriers}")
    print(f"  Pilot spacing         : {config.pilot_spacing}")
    print(f"  OFDM symbols simulated: {config.num_ofdm_symbols}")
    print(f"  SNR (dB)              : {config.snr_db}")
    print(f"  Channel taps          : {config.channel_taps}")
    print(f"  Average channel MSE   : {channel_mse:.4e}")
    print(f"  Bit error rate        : {ber:.4e}")


if __name__ == "__main__":
    main()

