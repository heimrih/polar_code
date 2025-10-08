"""Unified BER/FER sweeps across coding schemes."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import config as global_config
from ..utils.seeding import seed_all
from ..polar.crc import attach_crc, check_crc
from ..polar.polar import construct_info_set
from ..polar import polar as polar_core
from ..polar.scl import decode_scl
from ..dlscl.flip import decode_with_retries
from ..nr.polar.scl_nr import encode_rate_matched as encode_nr_polar
from ..nr.polar.scl_nr import decode_rate_matched_scl as decode_nr_polar
from ..nr.ldpc import (
    load_base_graph,
    build_h_matrix,
    encode_ldpc,
    rate_match_ldpc,
    derate_match_ldpc,
    decode_ldpc_nms,
)


@dataclass
class SimulationStats:
    bits_total: int = 0
    bit_errors: int = 0
    frame_errors: int = 0
    work_sum: float = 0.0
    frames: int = 0

    def update(self, bit_err: int, work: float, frame_error: bool, payload_len: int) -> None:
        self.bits_total += payload_len
        self.bit_errors += bit_err
        self.work_sum += work
        self.frames += 1
        if frame_error:
            self.frame_errors += 1

    def row(self) -> Dict[str, float]:
        ber = self.bit_errors / self.bits_total if self.bits_total > 0 else float("nan")
        fer = self.frame_errors / self.frames if self.frames > 0 else float("nan")
        avg_work = self.work_sum / self.frames if self.frames > 0 else 0.0
        return {
            "bits_total": self.bits_total,
            "bit_errors": self.bit_errors,
            "ber": ber,
            "fer": fer,
            "avg_work": avg_work,
        }


def _polar_encode(info_bits: np.ndarray, info_set: np.ndarray, N: int) -> np.ndarray:
    if info_bits.size != info_set.size:
        raise ValueError("info_bits length must match info_set size")
    u = np.zeros(N, dtype=np.int8)
    u[info_set] = info_bits
    return polar_core._polar_transform(u)


def _bpsk(bits: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * bits.astype(np.float64)


def _payload_bit_errors(payload: np.ndarray, candidate: Optional[np.ndarray], K_payload: int) -> int:
    if candidate is None:
        return int(K_payload)
    if candidate.size < K_payload:
        raise ValueError("Candidate bits shorter than payload")
    return int(np.count_nonzero(payload != candidate[:K_payload]))


def _decode_polar_scl(
    llr: np.ndarray,
    info_set: np.ndarray,
    M: int,
    crc_poly: str,
) -> Dict[str, np.ndarray]:
    return decode_scl(llr, info_set, M=M, crc=crc_poly)


def _decode_dl_scl(
    llr: np.ndarray,
    info_set: np.ndarray,
    M: int,
    retries: int,
    crc_poly: str,
    beta: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    return decode_with_retries(llr, info_set, M=M, retries=retries, crc=crc_poly, beta=beta)


def _noise_params(EbN0_dB: float, payload_bits: int, coded_bits: int) -> float:
    ebno_lin = 10 ** (EbN0_dB / 10.0)
    rate = payload_bits / coded_bits
    esn0_lin = ebno_lin * rate
    return 1.0 / (2.0 * esn0_lin)


def run_scheme(
    rng: np.random.Generator,
    EbN0_dB: float,
    args: argparse.Namespace,
    info_set: Optional[np.ndarray],
    encode_fn: Callable[[np.ndarray], np.ndarray],
    decode_fn: Callable[[np.ndarray], Dict[str, np.ndarray]],
    coded_len: int,
    payload_len: int,
    params_label: str,
) -> Dict[str, float]:
    stats = SimulationStats()
    noise_var = _noise_params(EbN0_dB, payload_len, coded_len)
    noise_sigma = math.sqrt(noise_var)

    while stats.bit_errors < args.err_cap and stats.bits_total < args.bits_cap:
        payload = rng.integers(0, 2, size=payload_len, dtype=np.int8)
        if args.scheme in {"polar_scl", "dl_scl"}:
            info_bits = payload if args.K_crc == 0 else attach_crc(payload, args.crc_poly)
            codeword = encode_fn(info_bits)
        elif args.scheme == "nr_polar_scl":
            codeword = encode_fn(payload)
        elif args.scheme == "nr_ldpc":
            message_bits = payload if args.K_crc == 0 else attach_crc(payload, args.crc_poly)
            codeword = encode_fn(message_bits)
        else:
            raise ValueError(f"Unsupported scheme: {args.scheme}")
        symbols = _bpsk(codeword)
        noise = rng.normal(0.0, noise_sigma, size=symbols.shape)
        received = symbols + noise
        llr = 2.0 * received / noise_var

        decode_result = decode_fn(llr)

        if args.scheme == "nr_ldpc":
            hard = decode_result.get("hard")
            k_total = payload_len + args.K_crc
            candidate = hard[:k_total] if hard is not None else None
        else:
            best_bits = decode_result.get("best_path_bits")
            if best_bits is None:
                best_bits = decode_result.get("payload")
            candidate = best_bits

        bit_err = _payload_bit_errors(payload, candidate, payload_len)
        frame_error = bit_err > 0

        work = 0.0
        if args.scheme == "dl_scl":
            attempts = decode_result.get("attempts")
            work = float(len(attempts) - 1) if attempts else 0.0
        elif args.scheme == "nr_ldpc":
            work = float(decode_result.get("iters_used", 0.0))

        stats.update(bit_err, work, frame_error, payload_len)

    row = stats.row()
    row.update(
        {
            "scheme": args.scheme,
            "code": args.scheme,
            "N_or_E": coded_len,
            "K_payload": payload_len,
            "K_crc": args.K_crc,
            "rate": payload_len / coded_len,
            "params": params_label,
            "EbN0_dB": EbN0_dB,
        }
    )
    return row


def build_runner(args: argparse.Namespace, info_set: np.ndarray, N: int) -> Callable[[np.ndarray], Dict[str, np.ndarray]]:
    if args.scheme == "polar_scl":
        return lambda llr: _decode_polar_scl(llr, info_set, args.M, args.crc_poly)
    if args.scheme == "dl_scl":
        beta = np.load(args.beta) if args.beta else None
        return lambda llr: _decode_dl_scl(llr, info_set, args.M, args.retries, args.crc_poly, beta)
    raise ValueError(f"Unsupported scheme: {args.scheme}")


def build_encoder(info_set: np.ndarray, N: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda info_bits: _polar_encode(info_bits, info_set, N)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BER/FER sweep across schemes")
    parser.add_argument("--scheme", required=True, choices=["polar_scl", "dl_scl", "nr_polar_scl", "nr_ldpc"], help="Coding scheme")
    parser.add_argument("--K_payload", type=int, required=True, help="Payload bits per frame")
    parser.add_argument("--K_crc", type=int, required=True, help="CRC bits per frame")
    parser.add_argument("--E", type=int, required=True, help="Coded bits transmitted")
    parser.add_argument("--N", type=int, help="Polar length before rate match (defaults to E)")
    parser.add_argument("--crc_poly", type=str, default=global_config.DEFAULTS.crc_poly)
    parser.add_argument("--M", type=int, default=4, help="List size for polar decoders")
    parser.add_argument("--retries", type=int, default=8, help="Retries for DL-SCL")
    parser.add_argument("--beta", type=str, help="Path to beta matrix (DL-SCL)")
    parser.add_argument("--ilv_mode", type=str, default="default")
    parser.add_argument("--bg", type=int, default=2, help="LDPC base graph")
    parser.add_argument("--Z", type=int, default=2, help="LDPC lifting size")
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--EbN0_lo", type=float, required=True)
    parser.add_argument("--EbN0_hi", type=float, required=True)
    parser.add_argument("--EbN0_step", type=float, default=0.5)
    parser.add_argument("--bits_cap", type=float, default=1e7)
    parser.add_argument("--err_cap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True, help="CSV output path")
    parser.add_argument("--plot", type=str, help="Optional plot path")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.scheme == "dl_scl" and not args.beta:
        raise ValueError("--beta is required for dl_scl scheme")
    return args


def run(args: argparse.Namespace) -> List[Dict[str, float]]:
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    N = args.N if args.N is not None else args.E
    K_total = args.K_payload + args.K_crc
    info_set = None
    encoder = None
    decoder = None
    params_label = ""

    if args.scheme in {"polar_scl", "dl_scl", "nr_polar_scl"}:
        info_set = construct_info_set(N, K_total)

    if args.scheme == "polar_scl":
        encoder = build_encoder(info_set, N)
        decoder = build_runner(args, info_set, N)
        params_label = f"M={args.M}"
    elif args.scheme == "dl_scl":
        encoder = build_encoder(info_set, N)
        decoder = build_runner(args, info_set, N)
        params_label = f"M={args.M},retries={args.retries}"
    elif args.scheme == "nr_polar_scl":
        encoder = lambda info_bits: encode_nr_polar(info_bits[:args.K_payload], args.crc_poly, N, args.E, info_set, args.ilv_mode)

        def _decode_nr(llr: np.ndarray) -> Dict[str, np.ndarray]:
            return decode_nr_polar(llr, args.crc_poly, N, args.E, info_set, args.M, args.ilv_mode)

        decoder = _decode_nr
        params_label = f"M={args.M},ilv={args.ilv_mode}"
    elif args.scheme == "nr_ldpc":
        bg = load_base_graph(args.bg)
        H = build_h_matrix(bg, args.Z)
        k = H.shape[1] - H.shape[0]
        if k != K_total:
            raise ValueError("LDPC payload+CRC size mismatch with base graph")
        encoder = lambda info_bits: rate_match_ldpc(encode_ldpc(info_bits[:k], H), args.E)

        def _decode_ldpc(llr: np.ndarray) -> Dict[str, np.ndarray]:
            internal = derate_match_ldpc(llr, H.shape[1])
            result = decode_ldpc_nms(internal, H, max_iter=args.max_iter, alpha=args.alpha)
            return result

        decoder = _decode_ldpc
        params_label = f"bg={args.bg},Z={args.Z},iter={args.max_iter},alpha={args.alpha}"
    else:
        raise ValueError(f"Unsupported scheme: {args.scheme}")

    EbN0_values = np.arange(args.EbN0_lo, args.EbN0_hi + 1e-12, args.EbN0_step)
    rows: List[Dict[str, float]] = []

    for EbN0_dB in EbN0_values:
        row = run_scheme(
            rng=rng,
            EbN0_dB=float(EbN0_dB),
            args=args,
            info_set=info_set,
            encode_fn=encoder,
            decode_fn=decoder,
            coded_len=args.E,
            payload_len=args.K_payload,
            params_label=params_label,
        )
        rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    header = [
        "scheme",
        "code",
        "N_or_E",
        "K_payload",
        "K_crc",
        "rate",
        "params",
        "EbN0_dB",
        "bits_total",
        "bit_errors",
        "ber",
        "fer",
        "avg_work",
    ]
    with path.open("w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[col]) for col in header) + "\n")


def plot_rows(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: r["EbN0_dB"])
    snrs = [r["EbN0_dB"] for r in rows_sorted]
    bers = [r["ber"] for r in rows_sorted]
    fers = [r["fer"] for r in rows_sorted]
    plt.figure(figsize=(6, 4))
    plt.semilogy(snrs, bers, "o-", label="BER")
    plt.semilogy(snrs, fers, "s-", label="FER")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Error Rate")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    rows = run(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_path)
    if args.plot:
        plot_rows(rows, Path(args.plot))


if __name__ == "__main__":
    main()
