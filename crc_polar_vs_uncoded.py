"""Simulation comparing CRC-aided polar coding against an uncoded BPSK baseline.

The script reuses the existing polar code implementation with the convolutional
precoder disabled (conv_gen=[1]) and measures bit/frame error rate under an
AWGN channel. Results for both the coded and uncoded schemes are reported for
each SNR point.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

import polar_coding_functions as pcf
from channel import channel
from crclib import crc
from polar_code import PolarCode
from rate_profile import rateprofile


try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional dependency for plotting
    plt = None


DEFAULT_SNR_POINTS = [float(f"{x:.1f}") for x in np.arange(-2.0, 6.5, 0.5)]


@dataclass
class SimulationResult:
    """Container for the performance metrics obtained at a single SNR point."""

    snr_db: float
    coded_ber: float
    coded_fer: float
    uncoded_ber: float
    uncoded_fer: float
    frames_run: int


def simulate(
    n: int,
    k_info: int,
    crc_length: int,
    crc_poly: int,
    list_size: int,
    design_snr_db: float,
    profile_name: str,
    snr_points: Sequence[float],
    target_frame_errors: int,
    max_frames: int,
    snr_mode: str = "SNRb",
    modulation: str = "BPSK",
    seed: int | None = None,
) -> List[SimulationResult]:
    """Run a Monte-Carlo simulation for several SNR points.

    Parameters
    ----------
    n: int
        Polar code block length (must be a power of two).
    k_info: int
        Number of information bits (CRC bits are appended to these).
    crc_length: int
        Length of the CRC appended to the information bits. Set to zero for no CRC.
    crc_poly: int
        Generator polynomial corresponding to ``crc_length``.
    list_size: int
        SCL decoder list size.
    design_snr_db: float
        Design SNR used for the rate-profile construction.
    profile_name: str
        Name of the rate profile to use (e.g., "dega", "rm-polar").
    snr_points: Sequence[float]
        Iterable with SNR points (in dB) at which to evaluate performance.
    target_frame_errors: int
        Stop the simulation for a given SNR once this many frame errors are
        accumulated (for the coded scheme).
    max_frames: int
        Upper bound on the number of frames to simulate per SNR point.
    snr_mode: str
        Either "SNR" (Es/N0) or "SNRb" (Eb/N0). Default is "SNRb".
    modulation: str
        Modulation scheme supported by :class:`~channel.channel`. Only BPSK is
        meaningful for the uncoded comparison.
    seed: int | None
        Optional NumPy random generator seed for reproducibility.
    """

    rng = np.random.default_rng(seed)

    non_frozen_bits = k_info + crc_length
    rate = k_info / n

    rprofile = rateprofile(n, non_frozen_bits, design_snr_db, 0)
    polar = PolarCode(n, non_frozen_bits, profile_name, L=list_size, rprofile=rprofile)
    polar.list_size_max = list_size
    polar.m = 0  # No convolutional precoding when conv_gen = [1]
    polar.gen = [1]
    polar.cur_state = []

    crc_obj = crc(crc_length, crc_poly) if crc_length > 0 else None
    crc_for_decoder = crc_obj if crc_obj is not None else crc(0, 0)

    results: List[SimulationResult] = []

    for snr in snr_points:
        ch_coded = channel(modulation, snr, snr_mode, rate if rate > 0 else 1.0)
        ch_uncoded = channel(modulation, snr, snr_mode, 1.0)

        coded_bit_errors = 0
        coded_frame_errors = 0
        uncoded_bit_errors = 0
        uncoded_frame_errors = 0
        coded_bits_total = 0
        uncoded_bits_total = 0
        frames = 0

        while frames < max_frames and coded_frame_errors < target_frame_errors:
            info_bits = rng.integers(0, 2, size=k_info, dtype=int)

            if crc_obj is not None:
                crc_bits = np.array(crc_obj.crcCalc(info_bits), dtype=int)
                message = np.concatenate([info_bits, crc_bits])
            else:
                message = info_bits

            codeword = polar.pac_encode(message, conv_gen=[1], mem=0, issystematic=False)

            coded_bits_total += len(message)

            modulated = np.array(ch_coded.modulate(codeword))
            noisy = ch_coded.add_noise(modulated)
            llr = ch_coded.calc_llr3(noisy)

            decoded = polar.pac_list_crc_decoder(
                llr,
                issystematic=False,
                isCRCinc=crc_obj is not None,
                crc1=crc_for_decoder,
                L=list_size,
            )
            decoded = np.array(decoded, dtype=int)

            bit_errors = pcf.fails(message, decoded)
            coded_bit_errors += int(bit_errors)
            if bit_errors > 0:
                coded_frame_errors += 1

            # Uncoded transmission of the raw information bits
            uncoded_mod = np.array(ch_uncoded.modulate(info_bits))
            uncoded_noisy = ch_uncoded.add_noise(uncoded_mod)
            hard_decision = (np.array(uncoded_noisy) < 0).astype(int)
            uncoded_bit_errors += int(pcf.fails(info_bits, hard_decision))
            if np.any(info_bits != hard_decision):
                uncoded_frame_errors += 1
            uncoded_bits_total += len(info_bits)

            frames += 1

        coded_ber = coded_bit_errors / coded_bits_total if coded_bits_total else 0.0
        coded_fer = coded_frame_errors / frames if frames else 0.0
        uncoded_ber = uncoded_bit_errors / uncoded_bits_total if uncoded_bits_total else 0.0
        uncoded_fer = uncoded_frame_errors / frames if frames else 0.0

        results.append(
            SimulationResult(
                snr_db=snr,
                coded_ber=coded_ber,
                coded_fer=coded_fer,
                uncoded_ber=uncoded_ber,
                uncoded_fer=uncoded_fer,
                frames_run=frames,
            )
        )

    return results


def _format_results(results: Iterable[SimulationResult]) -> str:
    header = (
        "SNR (dB) | Coded BER | Coded FER | Uncoded BER | Uncoded FER | Frames\n"
        "---------+-----------+-----------+-------------+-------------+-------"
    )
    rows = [
        f"{res.snr_db:8.2f} | {res.coded_ber:9.3e} | {res.coded_fer:9.3e} | "
        f"{res.uncoded_ber:11.3e} | {res.uncoded_fer:11.3e} | {res.frames_run:6d}"
        for res in results
    ]
    return "\n".join([header, *rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CRC-aided polar coding with an uncoded baseline over AWGN."
    )
    parser.add_argument("--n", type=int, default=128, help="Code length (must be power of two).")
    parser.add_argument("--k-info", type=int, default=64, help="Number of information bits per block.")
    parser.add_argument("--crc-length", type=int, default=16, help="CRC length in bits.")
    parser.add_argument(
        "--crc-poly",
        type=lambda x: int(x, 0),
        default="0x1021",
        help="Generator polynomial for the CRC (accepts hex literals).",
    )
    parser.add_argument("--list-size", type=int, default=16, help="List size for the SCL decoder.")
    parser.add_argument(
        "--design-snr",
        type=float,
        default=2.0,
        help="Design SNR (in dB) used when building the rate profile.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="dega",
        help="Rate profile identifier (see rate_profile.py for supported names).",
    )
    parser.add_argument(
        "--snr",
        type=float,
        nargs="*",
        default=DEFAULT_SNR_POINTS,
        help="SNR points (in dB) at which to evaluate performance.",
    )
    parser.add_argument(
        "--target-frame-errors",
        type=int,
        default=30,
        help="Stop simulation at an SNR point once this many coded frame errors are observed.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=5000,
        help="Maximum number of frames to simulate per SNR point.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random number generator seed for reproducibility.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting of BER/FER curves.",
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        default=None,
        help="Optional path to save the BER/FER figure (implies plotting).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = simulate(
        n=args.n,
        k_info=args.k_info,
        crc_length=args.crc_length,
        crc_poly=args.crc_poly,
        list_size=args.list_size,
        design_snr_db=args.design_snr,
        profile_name=args.profile,
        snr_points=args.snr,
        target_frame_errors=args.target_frame_errors,
        max_frames=args.max_frames,
        seed=args.seed,
    )
    print(_format_results(results))

    if args.no_plot:
        return

    if plt is None:
        print("matplotlib is not installed; skipping plot generation.")
        return

    _plot_results(results, save_path=args.plot_file, show=args.plot_file is None)


def _plot_results(results: Sequence[SimulationResult], save_path: str | None, show: bool) -> None:
    snr = [res.snr_db for res in results]
    coded_ber = [res.coded_ber for res in results]
    uncoded_ber = [res.uncoded_ber for res in results]
    coded_fer = [res.coded_fer for res in results]
    uncoded_fer = [res.uncoded_fer for res in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    axes[0].semilogy(snr, coded_ber, marker="o", label="Coded BER")
    axes[0].semilogy(snr, uncoded_ber, marker="s", label="Uncoded BER")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Bit Error Rate")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.6)
    axes[0].legend()

    axes[1].semilogy(snr, coded_fer, marker="o", label="Coded FER")
    axes[1].semilogy(snr, uncoded_fer, marker="s", label="Uncoded FER")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Frame Error Rate")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.6)
    axes[1].legend()

    fig.suptitle("CRC-Polar vs. Uncoded Performance over AWGN")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
