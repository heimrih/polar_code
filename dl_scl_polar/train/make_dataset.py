"""Dataset generation for β flip-metric training (Milestone M3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from .. import config
from ..utils.seeding import seed_all
from ..polar.polar import construct_info_set, encode
from ..polar.crc import attach_crc, check_crc
from ..polar.scl import decode_scl
from ..dlscl.flip import retry_with_flip


def _bpsk(codeword: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * codeword


def generate_samples(args: argparse.Namespace) -> None:
    cfg = config.get_config()
    seed_all(args.seed)

    info_set = construct_info_set(cfg.N, cfg.K)
    payload_bits = cfg.K - cfg.crc_bits

    payload = np.zeros(payload_bits, dtype=np.int8)
    info = attach_crc(payload, cfg.crc_poly)
    code = encode(info)
    symbols = _bpsk(code)

    rate = cfg.K / cfg.N
    ebno = 10 ** (args.snr_db / 10.0)
    noise_var = 1.0 / (2.0 * rate * ebno)
    noise_sigma = np.sqrt(noise_var)

    rng = np.random.default_rng(args.seed)

    abs_l0_samples: List[np.ndarray] = []
    labels: List[int] = []
    failures = 0

    for frame in range(args.frames):
        noise = rng.normal(0.0, noise_sigma, size=symbols.shape)
        received = symbols + noise
        llr = 2.0 * received / noise_var

        baseline = decode_scl(llr, info_set, args.M, crc=cfg.crc_poly)
        best_bits = baseline.get("best_path_bits")
        best_llrs = baseline.get("best_path_info_llrs")

        if best_bits is None or best_llrs is None:
            failures += 1
            continue
        if check_crc(best_bits, cfg.crc_poly):
            continue  # no failure to repair

        abs_l0 = np.abs(np.asarray(best_llrs, dtype=np.float32))
        if abs_l0.shape[0] != cfg.K:
            failures += 1
            continue

        order = np.argsort(abs_l0)
        success_label = None
        max_attempts = min(8, order.size)
        for idx in order[:max_attempts]:
            retry = retry_with_flip(
                llr,
                info_set,
                args.M,
                best_path_bits=best_bits,
                flip_index=int(idx),
                crc=cfg.crc_poly,
            )
            candidate = retry.get("best_path_bits")
            if candidate is None:
                continue
            if check_crc(candidate, cfg.crc_poly) and np.array_equal(candidate, info):
                success_label = int(idx)
                break

        if success_label is None:
            failures += 1
            continue

        abs_l0_samples.append(abs_l0)
        labels.append(success_label)

    if not abs_l0_samples:
        raise RuntimeError("No samples collected; consider increasing frames or SNR")

    abs_array = np.stack(abs_l0_samples).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int32)
    meta = {
        "M": args.M,
        "EbN0_dB": args.snr_db,
        "seed": args.seed,
        "frames": args.frames,
        "crc_poly": cfg.crc_poly,
        "crc_bits": cfg.crc_bits,
        "samples": int(label_array.size),
        "failures": int(failures),
    }

    out_path = Path(args.out)
    out_dir = out_path.parent if out_path.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_path.name
    shard = out_dir / f"{prefix}_part0.npz"
    np.savez_compressed(
        shard,
        abs_l0=abs_array,
        flip_idx=label_array,
        meta=json.dumps(meta),
    )

    print(f"Saved {label_array.size} samples to {shard}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DL-SCL flip dataset")
    parser.add_argument("--M", type=int, required=True, help="SCL list size")
    parser.add_argument("--snr_db", type=float, default=5.0, help="AWGN Eb/N0 in dB")
    parser.add_argument("--frames", type=int, default=100000, help="Number of frames to simulate")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, required=True, help="Output prefix for dataset shards")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    generate_samples(args)


if __name__ == "__main__":
    main()
