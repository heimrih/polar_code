"""Run FER sweeps comparing baseline SCL and DL-SCL with β-guided flips."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import config
from ..utils.seeding import seed_all
from ..polar.polar import construct_info_set, encode
from ..polar.crc import attach_crc, check_crc
from ..polar.scl import decode_scl
from ..dlscl.flip import decode_with_retries


def _bpsk(codeword: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * codeword


def simulate_frame(
    llr: np.ndarray,
    info_set: np.ndarray,
    M: int,
    crc_poly: str,
    retries: int,
    beta: np.ndarray | None,
) -> Tuple[bool, bool]:
    base = decode_scl(llr, info_set, M, crc=crc_poly)
    scl_bits = base.get("best_path_bits")
    scl_pass = scl_bits is not None and check_crc(scl_bits, crc_poly)

    dl = decode_with_retries(llr, info_set, M, retries, crc=crc_poly, beta=beta)
    dl_bits = dl.get("best_path_bits")
    dl_pass = dl_bits is not None and check_crc(dl_bits, crc_poly)

    return scl_pass, dl_pass


def run_sweep(args: argparse.Namespace) -> None:
    cfg = config.get_config()
    seed_all(args.seed)

    info_set = construct_info_set(cfg.N, cfg.K)
    payload_bits = cfg.K - cfg.crc_bits

    snr_points = (
        np.arange(args.snr_lo, args.snr_hi + 1e-9, args.snr_step)
        if args.snr_step > 0
        else np.array([args.snr_lo])
    )

    beta = None
    if args.beta:
        beta = np.load(args.beta)

    rate = cfg.K / cfg.N

    results: List[Dict[str, float]] = []

    for snr_db in snr_points:
        rng = np.random.default_rng(args.seed + int(snr_db * 10))
        noise_var = 1.0 / (2.0 * rate * 10 ** (snr_db / 10.0))
        noise_sigma = math.sqrt(noise_var)

        scl_errors = 0
        dl_errors = 0
        total_frames = args.frames

        for frame in range(total_frames):
            payload = rng.integers(0, 2, size=payload_bits, dtype=np.int8)
            msg = attach_crc(payload, cfg.crc_poly)
            code = encode(msg)
            symbols = _bpsk(code)

            noise = rng.normal(0.0, noise_sigma, size=symbols.shape)
            received = symbols + noise
            llr = 2.0 * received / noise_var

            scl_pass, dl_pass = simulate_frame(llr, info_set, args.M, cfg.crc_poly, args.retries, beta)
            if not scl_pass:
                scl_errors += 1
            if not dl_pass:
                dl_errors += 1

        scl_fer = scl_errors / total_frames
        dl_fer = dl_errors / total_frames
        results.append({"snr_db": snr_db, "fer_scl": scl_fer, "fer_dl": dl_fer})
        print(f"SNR={snr_db:.2f} dB -> SCL FER={scl_fer:.3e}, DL FER={dl_fer:.3e}")

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"fer_M{args.M}.csv"
    with csv_path.open("w") as f:
        f.write("snr_db,fer_scl,fer_dl\n")
        for row in results:
            f.write(f"{row['snr_db']:.3f},{row['fer_scl']:.6e},{row['fer_dl']:.6e}\n")
    print(f"Saved FER table to {csv_path}")

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"fer_M{args.M}.png"
    plt.figure(figsize=(6, 4))
    snrs = [row["snr_db"] for row in results]
    plt.semilogy(snrs, [row["fer_scl"] for row in results], "o-", label="SCL")
    plt.semilogy(snrs, [row["fer_dl"] for row in results], "s-", label="DL-SCL")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Frame Error Rate")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved FER plot to {plot_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FER sweep for DL-SCL")
    parser.add_argument("--M", type=int, required=True, help="List size")
    parser.add_argument("--frames", type=int, default=10000, help="Frames per SNR point")
    parser.add_argument("--snr_lo", type=float, default=4.0)
    parser.add_argument("--snr_hi", type=float, default=6.5)
    parser.add_argument("--snr_step", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--beta", type=str, help="Path to trained β matrix (.npy)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--plot_dir", type=str, default="plots")
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    run_sweep(args)


if __name__ == "__main__":
    main()
