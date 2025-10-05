"""Train symmetric β matrices for DL-SCL flip ranking."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from glob import glob
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .. import config
from ..utils.seeding import seed_all
from ..dlscl.beta import SymmetricBeta


def _load_dataset(paths: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
    abs_l0_list: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for pattern in paths:
        matches = sorted(glob(pattern))
        if not matches:
            file_path = Path(pattern)
            if file_path.exists():
                matches = [str(file_path)]
        for file_str in matches:
            data = np.load(file_str)
            abs_l0_list.append(data["abs_l0"])
            labels.append(data["flip_idx"])
    if not abs_l0_list:
        raise FileNotFoundError("No dataset shards found for the provided --data patterns")
    abs_l0 = np.concatenate(abs_l0_list, axis=0)
    labels_arr = np.concatenate(labels, axis=0)
    return abs_l0.astype(np.float32), labels_arr.astype(np.int64)


def _split_train_val(abs_l0: np.ndarray, labels: np.ndarray, val_frac: float, seed: int) -> Tuple[TensorDataset, TensorDataset]:
    rng = np.random.default_rng(seed)
    num_samples = abs_l0.shape[0]
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    split = int(num_samples * (1.0 - val_frac))
    train_idx = indices[:split]
    val_idx = indices[split:]

    def make_dataset(idx: np.ndarray) -> TensorDataset:
        tensors = (
            torch.from_numpy(abs_l0[idx]),
            torch.from_numpy(labels[idx]),
        )
        return TensorDataset(*tensors)

    return make_dataset(train_idx), make_dataset(val_idx)


def train_beta(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    cfg = config.get_config()

    abs_l0, labels = _load_dataset(args.data)
    dim = abs_l0.shape[1]

    train_ds, val_ds = _split_train_val(abs_l0, labels, args.val_frac, args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")

    model = SymmetricBeta(dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_M{args.M}.csv"

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"beta_M{args.M}.npy"

    best_val = float("inf")
    best_beta = None

    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            for batch_abs, batch_lbl in train_loader:
                batch_abs = batch_abs.to(device)
                batch_lbl = batch_lbl.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = -model(batch_abs)
                loss = criterion(logits, batch_lbl)
                if args.lambda_l2 > 0:
                    off_diag = model.off_diag
                    l2_term = off_diag.pow(2).sum() / (dim * dim)
                    loss = loss + args.lambda_l2 * l2_term
                loss.backward()
                model.clamp_diagonal()
                optimizer.step()

                total_loss += loss.item() * batch_abs.size(0)
                total_correct += (logits.argmax(dim=1) == batch_lbl).sum().item()
                total_samples += batch_abs.size(0)

            train_loss = total_loss / max(total_samples, 1)
            train_acc = total_correct / max(total_samples, 1)

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_samples = 0
            with torch.no_grad():
                for batch_abs, batch_lbl in val_loader:
                    batch_abs = batch_abs.to(device)
                    batch_lbl = batch_lbl.to(device)
                    logits = -model(batch_abs)
                    loss = criterion(logits, batch_lbl)
                    val_loss += loss.item() * batch_abs.size(0)
                    val_correct += (logits.argmax(dim=1) == batch_lbl).sum().item()
                    val_samples += batch_abs.size(0)

            if val_samples > 0:
                val_loss /= val_samples
                val_acc = val_correct / val_samples
            else:
                val_loss = float("nan")
                val_acc = float("nan")

            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
            f.flush()

            if val_samples > 0 and val_loss < best_val:
                best_val = val_loss
                beta_matrix = model.beta_matrix().detach().cpu().numpy()
                best_beta = beta_matrix
        if best_beta is None:
            best_beta = model.beta_matrix().detach().cpu().numpy()

    np.save(ckpt_path, best_beta)
    print(f"Saved β checkpoint to {ckpt_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train symmetric β for DL-SCL")
    parser.add_argument("--M", type=int, required=True, help="SCL list size")
    parser.add_argument("--data", nargs="+", required=True, help="Glob(s) to dataset shards")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lambda_l2", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    train_beta(args)


if __name__ == "__main__":
    main()
