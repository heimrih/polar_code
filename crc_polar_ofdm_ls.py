"""CRC-polar performance over OFDM with LS channel estimation vs. perfect CSI.

The script reuses the configuration-driven style used by the other simulation
helpers in this repository.  Edit the :data:`CONFIG` object at the bottom of the
file to adjust the polar-code parameters, OFDM layout, or SNR sweep, then run
``python crc_polar_ofdm_ls.py`` to launch the experiment.  For each SNR point it
reports frame and bit error rates for the LS-estimated receiver together with a
baseline that assumes perfect channel knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - matplotlib is optional at runtime
    plt = None

import numpy as np

import polar_coding_functions as pcf
from crclib import crc
from ofdm_channel_estimation import add_awgn, ls_channel_estimate, rayleigh_frequency_response
from polar_code import PolarCode
from rate_profile import rateprofile

DEFAULT_SNR_POINTS = tuple(float(f"{x:.1f}") for x in np.arange(-2.0, 6.5, 0.5))


@dataclass
class SimulationResult:
    """Container for the performance metrics obtained at a single SNR point."""

    snr_db: float
    ls_ber: float
    ls_fer: float
    perfect_ber: float
    perfect_fer: float
    avg_channel_mse: float
    frames_run: int


@dataclass
class SimulationConfig:
    """Tunable parameters for the CRC-polar OFDM simulation."""

    n: int = 128
    k_info: int = 64
    crc_length: int = 16
    crc_poly: int = 0x1021
    list_size: int = 16
    design_snr_db: float = 2.0
    profile_name: str = "dega"
    snr_points: Sequence[float] = field(default_factory=lambda: DEFAULT_SNR_POINTS)
    target_frame_errors: int = 30
    max_frames: int = 5000
    min_frames_per_snr: int = 50
    stop_when_error_free: bool = True
    seed: int | None = None

    # OFDM-specific parameters
    num_subcarriers: int = 128
    pilot_spacing: int = 8
    channel_taps: int = 8
    ofdm_symbols_per_frame: int | None = None  # ``None`` -> computed automatically
    pilot_value: complex = 1 + 0j

    # Plotting controls
    plot_results: bool = True
    plot_file: str | None = None


# Modify the values below to customise the simulation without needing command-line flags.
CONFIG = SimulationConfig()


def _pilot_indices(num_subcarriers: int, spacing: int) -> np.ndarray:
    if num_subcarriers < 2:
        raise ValueError("num_subcarriers must be at least 2")
    if spacing < 1:
        raise ValueError("pilot_spacing must be positive")

    pilots = np.arange(0, num_subcarriers, spacing)
    if pilots[-1] != num_subcarriers - 1:
        pilots = np.append(pilots, num_subcarriers - 1)
    return pilots


def _bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    return 1 - 2 * bits


def _compute_bpsk_llr(equalized: np.ndarray, channel_mag_sq: np.ndarray, noise_variance: float) -> np.ndarray:
    safe_noise = max(noise_variance, 1e-12)
    safe_mag_sq = np.maximum(channel_mag_sq, 1e-12)
    return 4.0 * equalized.real * (safe_mag_sq / safe_noise)


def simulate(config: SimulationConfig) -> List[SimulationResult]:
    rng = np.random.default_rng(config.seed)

    non_frozen_bits = config.k_info + config.crc_length
    if non_frozen_bits > config.n:
        raise ValueError("k_info + crc_length must not exceed n")

    rprofile = rateprofile(config.n, non_frozen_bits, config.design_snr_db, 0)
    polar = PolarCode(
        config.n,
        non_frozen_bits,
        config.profile_name,
        L=config.list_size,
        rprofile=rprofile,
    )
    polar.list_size_max = config.list_size
    polar.m = 0
    polar.gen = [1]
    polar.cur_state = []

    crc_obj = crc(config.crc_length, config.crc_poly) if config.crc_length > 0 else None
    crc_for_decoder = crc_obj if crc_obj is not None else crc(0, 0)

    pilot_indices = _pilot_indices(config.num_subcarriers, config.pilot_spacing)
    data_indices = np.setdiff1d(np.arange(config.num_subcarriers), pilot_indices)
    if data_indices.size == 0:
        raise ValueError("No data subcarriers remain after placing pilots")

    min_symbols = int(np.ceil(config.n / data_indices.size))
    if config.ofdm_symbols_per_frame is None:
        num_symbols = min_symbols
    else:
        if config.ofdm_symbols_per_frame < min_symbols:
            raise ValueError(
                "ofdm_symbols_per_frame is insufficient for the requested block length"
            )
        num_symbols = config.ofdm_symbols_per_frame

    results: List[SimulationResult] = []

    if config.min_frames_per_snr < 1:
        raise ValueError("min_frames_per_snr must be at least 1")

    for snr in config.snr_points:
        snr_linear = 10 ** (snr / 10.0)

        ls_bit_errors = 0
        ls_frame_errors = 0
        perfect_bit_errors = 0
        perfect_frame_errors = 0
        bits_total = 0
        frames = 0
        channel_mse_accum = 0.0
        channel_mse_samples = 0

        while frames < config.max_frames and ls_frame_errors < config.target_frame_errors:
            info_bits = rng.integers(0, 2, size=config.k_info, dtype=int)
            if crc_obj is not None:
                crc_bits = np.array(crc_obj.crcCalc(info_bits), dtype=int)
                message = np.concatenate([info_bits, crc_bits])
            else:
                message = info_bits

            codeword = polar.pac_encode(message, conv_gen=[1], mem=0, issystematic=False)
            codeword = np.array(codeword, dtype=int)

            ls_llrs: list[float] = []
            perfect_llrs: list[float] = []

            bits_consumed = 0
            for _ in range(num_symbols):
                transmitted = np.full(config.num_subcarriers, config.pilot_value, dtype=np.complex128)

                bits_remaining = config.n - bits_consumed
                bits_this_symbol = min(data_indices.size, bits_remaining)

                if bits_this_symbol > 0:
                    symbol_bits = codeword[bits_consumed : bits_consumed + bits_this_symbol]
                    transmitted[data_indices[:bits_this_symbol]] = _bpsk_modulate(symbol_bits)
                    bits_consumed += bits_this_symbol

                if bits_this_symbol < data_indices.size:
                    transmitted[data_indices[bits_this_symbol:]] = 1.0

                channel = rayleigh_frequency_response(
                    config.num_subcarriers, config.channel_taps, rng
                )
                noiseless = channel * transmitted
                noisy = add_awgn(noiseless, snr, rng)
                noise_variance = np.mean(np.abs(noiseless) ** 2) / snr_linear

                safe_channel = np.where(np.abs(channel) < 1e-12, 1e-12, channel)
                perfect_equalized = noisy / safe_channel
                perfect_mag_sq = np.abs(safe_channel) ** 2

                est_channel = ls_channel_estimate(transmitted, noisy, pilot_indices)
                channel_mse_accum += float(np.mean(np.abs(est_channel - channel) ** 2))
                channel_mse_samples += 1
                safe_est = np.where(np.abs(est_channel) < 1e-12, 1e-12, est_channel)
                ls_equalized = noisy / safe_est
                ls_mag_sq = np.abs(safe_est) ** 2

                perfect_llr_symbol = _compute_bpsk_llr(
                    perfect_equalized[data_indices], perfect_mag_sq[data_indices], noise_variance
                )
                ls_llr_symbol = _compute_bpsk_llr(
                    ls_equalized[data_indices], ls_mag_sq[data_indices], noise_variance
                )

                if bits_this_symbol > 0:
                    perfect_llrs.extend(perfect_llr_symbol[:bits_this_symbol])
                    ls_llrs.extend(ls_llr_symbol[:bits_this_symbol])

            if bits_consumed != config.n:
                raise RuntimeError("Failed to map all coded bits onto OFDM symbols")

            ls_llrs_arr = np.array(ls_llrs, dtype=float)
            perfect_llrs_arr = np.array(perfect_llrs, dtype=float)

            ls_decoded = polar.pac_list_crc_decoder(
                ls_llrs_arr,
                issystematic=False,
                isCRCinc=crc_obj is not None,
                crc1=crc_for_decoder,
                L=config.list_size,
            )
            ls_decoded = np.array(ls_decoded, dtype=int)

            perfect_decoded = polar.pac_list_crc_decoder(
                perfect_llrs_arr,
                issystematic=False,
                isCRCinc=crc_obj is not None,
                crc1=crc_for_decoder,
                L=config.list_size,
            )
            perfect_decoded = np.array(perfect_decoded, dtype=int)

            ls_errors = int(pcf.fails(message, ls_decoded))
            perfect_errors = int(pcf.fails(message, perfect_decoded))

            ls_bit_errors += ls_errors
            perfect_bit_errors += perfect_errors
            bits_total += message.size

            if ls_errors > 0:
                ls_frame_errors += 1
            if perfect_errors > 0:
                perfect_frame_errors += 1

            frames += 1

            if (
                config.stop_when_error_free
                and frames >= config.min_frames_per_snr
                and ls_frame_errors == 0
                and perfect_frame_errors == 0
            ):
                break

        ls_ber = ls_bit_errors / bits_total if bits_total else 0.0
        ls_fer = ls_frame_errors / frames if frames else 0.0
        perfect_ber = perfect_bit_errors / bits_total if bits_total else 0.0
        perfect_fer = perfect_frame_errors / frames if frames else 0.0
        avg_mse = channel_mse_accum / channel_mse_samples if channel_mse_samples else 0.0

        results.append(
            SimulationResult(
                snr_db=snr,
                ls_ber=ls_ber,
                ls_fer=ls_fer,
                perfect_ber=perfect_ber,
                perfect_fer=perfect_fer,
                avg_channel_mse=avg_mse,
                frames_run=frames,
            )
        )

    return results


def _format_results(results: Iterable[SimulationResult]) -> str:
    header = (
        "SNR (dB) |   LS BER  |   LS FER  | Perfect BER | Perfect FER | Avg Ch. MSE | Frames\n"
        "---------+----------+----------+------------+------------+------------+-------"
    )
    rows = [
        f"{res.snr_db:8.2f} | {res.ls_ber:8.3e} | {res.ls_fer:8.3e} | "
        f"{res.perfect_ber:10.3e} | {res.perfect_fer:10.3e} | "
        f"{res.avg_channel_mse:10.3e} | {res.frames_run:6d}"
        for res in results
    ]
    return "\n".join([header, *rows])


def main(config: SimulationConfig = CONFIG) -> None:
    results = simulate(config)
    print(_format_results(results))

    if config.plot_results:
        if plt is None:
            raise RuntimeError(
                "matplotlib is required for plotting but is not installed. "
                "Either install matplotlib or set config.plot_results to False."
            )

        snrs = [res.snr_db for res in results]
        ls_ber = [res.ls_ber for res in results]
        ls_fer = [res.ls_fer for res in results]
        perfect_ber = [res.perfect_ber for res in results]
        perfect_fer = [res.perfect_fer for res in results]

        def _safe(values: Sequence[float]) -> np.ndarray:
            arr = np.asarray(values, dtype=float)
            return np.maximum(arr, 1e-12)

        plt.figure(figsize=(8, 6))
        plt.semilogy(snrs, _safe(ls_ber), marker="o", label="LS BER")
        plt.semilogy(snrs, _safe(perfect_ber), marker="o", label="Perfect CSI BER")
        plt.semilogy(snrs, _safe(ls_fer), marker="s", label="LS FER")
        plt.semilogy(snrs, _safe(perfect_fer), marker="s", label="Perfect CSI FER")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Error Rate")
        plt.title("CRC-Polar over OFDM: LS vs. Perfect CSI")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        if config.plot_file:
            plt.savefig(config.plot_file, dpi=150)

        if config.plot_file is None:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    main()
