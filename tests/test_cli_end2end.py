import os
from argparse import Namespace

import numpy as np

from dl_scl_polar.train.make_dataset import generate_samples
from dl_scl_polar.train.train_beta import train_beta
from dl_scl_polar.eval.run_fer_sweep import run_sweep


def test_cli_end2end(tmp_path):
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    data_prefix = tmp_path / "data" / "train_M2_small"
    args = Namespace(M=2, snr_db=0.0, frames=80, seed=1234, out=str(data_prefix))
    generate_samples(args)

    shard = next((data_prefix.parent).glob(f"{data_prefix.name}_part*.npz"))
    data = np.load(shard)
    assert data["abs_l0"].size > 0

    ckpt_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"
    train_args = Namespace(
        M=2,
        data=[str(shard)],
        epochs=1,
        lr=1e-4,
        batch=32,
        lambda_l2=0.1,
        seed=1234,
        val_frac=0.5,
        checkpoint_dir=str(ckpt_dir),
        log_dir=str(log_dir),
        cpu=True,
    )
    train_beta(train_args)

    ckpt_path = ckpt_dir / "beta_M2.npy"
    assert ckpt_path.exists()

    sweep_args = Namespace(
        M=2,
        frames=200,
        snr_lo=4.5,
        snr_hi=4.5,
        snr_step=0,
        retries=2,
        beta=str(ckpt_path),
        seed=4321,
        out_dir=str(tmp_path / "results"),
        plot_dir=str(tmp_path / "plots"),
        include_uncoded=False,
    )
    run_sweep(sweep_args)

    assert (tmp_path / "results" / "fer_M2.csv").exists()
    assert (tmp_path / "plots" / "fer_M2.png").exists()

    with (tmp_path / "results" / "fer_M2.csv").open() as f:
        header = f.readline().strip().split(",")
        assert header == ["snr_db", "fer_scl", "ber_scl", "fer_dl", "ber_dl"]
        row = f.readline().strip().split(",")
        assert len(row) == len(header)

    sweep_args_uncoded = Namespace(
        M=2,
        frames=100,
        snr_lo=4.5,
        snr_hi=4.5,
        snr_step=0,
        retries=2,
        beta=str(ckpt_path),
        seed=9876,
        out_dir=str(tmp_path / "results_uncoded"),
        plot_dir=str(tmp_path / "plots_uncoded"),
        include_uncoded=True,
    )
    run_sweep(sweep_args_uncoded)

    assert (tmp_path / "results_uncoded" / "fer_M2.csv").exists()
    assert (tmp_path / "plots_uncoded" / "fer_M2.png").exists()

    with (tmp_path / "results_uncoded" / "fer_M2.csv").open() as f:
        header = f.readline().strip().split(",")
        assert header == ["snr_db", "fer_uncoded", "ber_uncoded", "fer_scl", "ber_scl", "fer_dl", "ber_dl"]
        row = f.readline().strip().split(",")
        assert len(row) == len(header)
