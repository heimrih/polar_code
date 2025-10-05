# DL-SCL Polar Codes

This repository reproduces a deep-learning-guided successive-cancellation list (DL-SCL) decoder for the 
Polar code P(N=128, K=64) with CRC-24 over the AWGN channel. The workflow follows the milestones in
`AGENTS.md`: baseline polar/CRC/SCL implementation, flip-retry semantics, β-metric training data,
β training, FER evaluation, and complexity accounting.

## Repository Layout

```
dl_scl_polar/
  config.py              # Global defaults (N, K, CRC, Eb/N0 grid, retries, seeds)
  utils/seeding.py       # Deterministic seeding helpers
  polar/
    polar.py             # Encoder, SC decoder, info-set construction
    scl.py               # SCL decoder with CRC filtering and forced-bit support
    crc.py               # CRC attach/check utilities
  dlscl/
    beta.py              # Symmetric β module (learnable metric)
    flip.py              # Baseline + DL-SCL retry logic
  train/
    make_dataset.py      # Collect |L0| vectors + oracle flip labels
    train_beta.py        # Train β for each list size
  eval/
    run_fer_sweep.py     # Compare SCL vs DL-SCL FER curves
    opcount.py           # Count β adds/mults before/after pruning
results/, plots/, data/, checkpoints/, logs/ store generated artifacts.
```

Unit tests live in `tests/`, including an end-to-end CLI test that exercises
`make_dataset.py`, `train_beta.py`, and `run_fer_sweep.py` on a tiny configuration.

## Environment

- Python 3.10+
- NumPy, PyTorch, Matplotlib (listed in `pyproject.toml`)
- CPU execution only; set `OMP_NUM_THREADS=1` in constrained environments to avoid
  OpenMP shared-memory issues.

Install the project in editable mode (optional but convenient):

```
pip install -e .[dev]
```

## Key Commands

### 1. Generate Training Data (per list size M ∈ {1,2,4,8})

```
python -m dl_scl_polar.train.make_dataset --M 4 --snr_db 5 --frames 300000 \
    --seed 0 --out data/train_M4_snr5_seed0
```

The script writes shards such as `data/train_M4_snr5_seed0_part0.npz` containing
`abs_l0`, `flip_idx`, and a JSON `meta` field documenting the run. Only SCL failures
with a successful flip (within 8 attempts) are kept.

### 2. Train β Metrics

```
python -m dl_scl_polar.train.train_beta --M 4 \
    --data 'data/train_M4_snr5_seed0_part*.npz' \
    --epochs 8 --lr 1e-4 --batch 128 --lambda_l2 0.25
```

Artifacts:
- `checkpoints/beta_M4.npy`
- `logs/train_M4.csv`

Repeat for each list size (M = 1, 2, 4, 8).

### 3. FER Sweep (SCL vs DL-SCL)

```
python -m dl_scl_polar.eval.run_fer_sweep --M 4 --frames 10000 --retries 8 \
    --snr_lo 4.0 --snr_hi 6.5 --snr_step 0.5 \
    --beta checkpoints/beta_M4.npy \
    --out_dir results --plot_dir plots
```

Outputs:
- `results/fer_M4.csv`
- `plots/fer_M4.png`

Adjust M, β checkpoint, SNR grid, and frame count as needed.

### 4. Complexity Accounting

```
python -m dl_scl_polar.eval.opcount --beta checkpoints/beta_M4.npy \
    --prune 1e-4 --report results/opcount_M4.csv
```

This reports the number of non-zero β entries, multiplies, and adds before/after
pruning |β| ≤ 1e-4. Optionally save the pruned matrix with `--save_pruned`.

## Testing

```
pytest -q
```

`tests/test_cli_end2end.py` performs a miniature dataset generation, β training,
and FER sweep to ensure the CLIs remain compatible.

## Generated Artifacts (example run)

- `data/train_M{1,2,4,8}_snr5_seed0_part0.npz`
- `checkpoints/beta_M{1,2,4,8}.npy`
- `logs/train_M{1,2,4,8}.csv`
- `results/fer_M{1,2,4,8}.csv`
- `plots/fer_M{1,2,4,8}.png`
- `results/opcount_M4.csv`

Regenerate these artifacts using the commands above to reproduce the DL-SCL experiments.
