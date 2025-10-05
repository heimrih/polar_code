"""Operation counting for β-based correlation metric."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import numpy as np


def _count_ops(beta: np.ndarray) -> Tuple[int, int, int]:
    if beta.ndim != 2 or beta.shape[0] != beta.shape[1]:
        raise ValueError("beta must be a square matrix")
    n = beta.shape[0]
    mask = beta != 0.0
    nonzero = int(mask.sum())
    multiplies = nonzero
    adds = 0
    for col in range(n):
        col_nnz = int(mask[:, col].sum())
        if col_nnz > 0:
            adds += col_nnz - 1
    return nonzero, multiplies, adds


def run(args: argparse.Namespace) -> None:
    beta = np.load(args.beta)

    nonzero_full, mult_full, add_full = _count_ops(beta)

    pruned = beta.copy()
    pruned[np.abs(pruned) <= args.prune] = 0.0
    nonzero_pruned, mult_pruned, add_pruned = _count_ops(pruned)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "nonzero", "multiplies", "adds"])
        writer.writerow(["full", nonzero_full, mult_full, add_full])
        writer.writerow(["pruned", nonzero_pruned, mult_pruned, add_pruned])

    print(f"Saved opcount report to {report_path}")

    if args.save_pruned:
        pruned_path = Path(args.save_pruned)
        pruned_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(pruned_path, pruned)
        print(f"Saved pruned β to {pruned_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Count operations for β metric")
    parser.add_argument("--beta", required=True, help="Path to β matrix (.npy)")
    parser.add_argument("--prune", type=float, default=1e-4, help="Threshold for pruning")
    parser.add_argument("--report", required=True, help="CSV output path")
    parser.add_argument("--save_pruned", help="Optional path to save pruned matrix")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
