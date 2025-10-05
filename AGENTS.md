# AGENTS.md

A practical, step‑by‑step orchestration guide for building and reproducing the **Deep‑Learning‑Aided SCL (DL‑SCL)** polar decoding experiment with small, focused agents. This document favors a conservative, incremental path so each step is verifiable before moving on.

---

## 0) Scope & Non‑Goals

* **Scope**: Polar code P(N=128, K=64) with CRC‑24, BPSK over AWGN, baseline SC/SCL decoders, and a learned **β** correlation matrix that guides up to 8 **bit‑flip SCL** retries after a failed SCL attempt.
* **Non‑Goals**: Channel estimation, fading channels, 5G rate‑matching, hardware acceleration, non‑BPSK modulations.

---

## 1) Repo Layout (target)

```
polar_code/
  README.md
  AGENTS.md  ← this file
  pyproject.toml
  dl_scl_polar/
    __init__.py
    config.py
    utils/seeding.py
    polar/polar.py
    polar/scl.py
    polar/crc.py
    dlscl/beta.py
    dlscl/flip.py
    train/make_dataset.py
    train/train_beta.py
    eval/run_fer_sweep.py
    eval/opcount.py
  tests/
    test_polar_basics.py
    test_scl_crc.py
    test_flip_logic.py
    test_beta_symmetry.py
    test_cli_end2end.py
```

---

## 2) Global Conventions

* **Language/stack**: Python 3.10+, NumPy, PyTorch. No GPU required.
* **Determinism**: Set seeds via `utils/seeding.py` and per‑script CLI `--seed`.
* **Config**: Central defaults in `config.py`; all scripts accept CLI overrides.
* **CRC**: 24‑bit poly configurable; default to a 5G‑standard CRC‑24.
* **List sizes**: train/eval for M ∈ {1, 2, 4, 8}.
* **Retries**: Allow up to 8 extra flip attempts on SCL failure.

---

## 3) Milestones (strict order)

### M0 — Scaffold & Minimal Proof

**Goal**: Encode→BPSK→AWGN→LLR→SC decode (no CRC, no SCL).
**DoD**: Round‑trip bit‑perfect at high SNR; unit tests pass.

Tasks:

1. `polar.py`: implement generator, info/frozen sets, `encode(bits)`, `f/g` LLR combines, and `sc_decode(llr)`.
2. `test_polar_basics.py`: noiseless round‑trip and simple AWGN sanity.

### M1 — CRC & SCL Baseline

**Goal**: Add CRC‑24 and list decoding.
**DoD**: `scl.decode(...)` returns a CRC‑passing candidate when SNR is moderate.

Tasks:

1. `crc.py`: functions `crc_attach(msg)`, `crc_check(msg_crc)`.
2. `scl.py`: standard LLR‑based SCL with path metric; integrate CRC filter.
3. `test_scl_crc.py`: unit tests for CRC correctness and SCL basic behavior.

### M2 — Flip Mechanics (Metric‑Agnostic)

**Goal**: Implement the **fix/flip** retry mechanism using a placeholder metric (e.g., absolute LLR ascending).
**DoD**: `flip.decode_with_retries(...)` can recover a fraction of CRC failures.

Tasks:

1. `flip.py`: given best path and an index i*, fix all previous info bits to prior decisions, flip bit i*, continue SCL from i*+1 using the same M.
2. `test_flip_logic.py`: verifies fix/flip semantics and equivalence to baseline when 0 retries.

### M3 — β Metric & Training Data

**Goal**: Implement symmetric β and dataset maker.
**DoD**: `beta.forward(abs_L0)` returns Q; dataset contains only SCL‑fail samples.

Tasks:

1. `beta.py`: class `SymmetricBeta(dim)`, enforce symmetry, diag==1.0.
2. `make_dataset.py`: generate all‑zero codewords at Eb/N0=5 dB, run SCL once; save samples that fail CRC; store `(abs_L0, label)` where **label is the oracle flip index** that first yields a CRC pass when applying flip logic (search limited to at most 8 attempts). Discard samples with no successful flip.
3. `test_beta_symmetry.py`: checks β symmetry and diagonal constraints.

### M4 — Train β (Per M)

**Goal**: Train one β per list size.
**DoD**: Training converges; validation top‑1 accuracy on oracle flip index ≥ threshold (e.g., 30–50%).

Tasks:

1. `train_beta.py`: RMSprop, lr=1e‑4, batch=128, L2 on off‑diagonals (λ=0.25). Initialize off‑diagonals U[−0.2,0.2]. Loss: cross‑entropy to oracle index.
2. Save β as `.npy`; log metrics.

### M5 — Integrate DL‑SCL & Evaluate FER

**Goal**: Use trained β to rank flip indices; run FER sweep.
**DoD**: FER curve within ~0.1–0.2 dB of the paper’s DL‑SCL trend; SCL baseline acts as sanity reference.

Tasks:

1. `run_fer_sweep.py`: compare (a) SCL only, (b) DL‑SCL (β) with 8 retries; sweep Eb/N0 ∈ [4.0, 6.5] dB.
2. `test_cli_end2end.py`: small sweep (fewer frames) executes without error and produces plots.

### M6 — Complexity Accounting & Pruning

**Goal**: Show β’s low‑complexity benefits.
**DoD**: Report add/mul counts before/after pruning |β|≤1e‑4.

Tasks:

1. `opcount.py`: count operations for Q = |L0| @ β.
2. Prune, recount, and write a small table (CSV/Markdown).

---

## 4) Agent Roster

Each agent has a **single responsibility**, clear inputs/outputs, and a Definition of Done. Agents communicate only via the filesystem and test suite.

### A. Project Manager (PM)

* **Goal**: Keep the milestone order, open/close issues, enforce DoD.
* **Inputs**: This AGENTS.md, issue tracker.
* **Outputs**: Created issues, milestone checklists, release tags.
* **DoD**: All milestones closed; `tests/` passing; `README.md` updated.

### B. Spec/Config Agent

* **Goal**: Centralize defaults in `config.py` and CLIs.
* **Inputs**: Milestones.
* **Outputs**: `config.py`, argparse wiring across scripts.
* **DoD**: One‑source‑of‑truth for N, K, CRC poly, M, Eb/N0 grid, retries, seeds.

### C. Polar Core Agent

* **Goal**: Implement `polar.py` and unit tests.
* **Inputs**: N, K, frozen set construction method.
* **Outputs**: Encoder/SC LLR combines.
* **DoD**: `test_polar_basics.py` passes; SC decodes noiseless frames.

### D. CRC Agent

* **Goal**: Add CRC‑24 attach/check with configurable polynomial.
* **DoD**: `test_scl_crc.py::test_crc_roundtrip` passes for random payloads.

### E. SCL Agent

* **Goal**: Implement LLR‑based SCL, path metrics, and CRC filtering.
* **DoD**: `test_scl_crc.py::test_scl_recovers_over_sc` passes at moderate SNR.

### F. Flip Logic Agent

* **Goal**: Implement fix/flip retry semantics independent of metric.
* **DoD**: `test_flip_logic.py` verifies equivalence to baseline when retries=0 and correct bit‑locking behavior when retries>0.

### G. Dataset Agent

* **Goal**: Produce `{abs_L0, oracle_flip_idx}` samples from SCL failures.
* **DoD**: `make_dataset.py` writes `.npz` shards (e.g., 64k samples each) with schema:

  * `abs_l0`: float32 [num_samples, K+c]
  * `flip_idx`: int32 [num_samples]
  * `meta`: JSON string (M, Eb/N0, seed, CRC poly)

### H. β Metric Agent

* **Goal**: `SymmetricBeta`(dim) with symmetry/diag constraints and forward pass `Q = abs_l0 @ beta`.
* **DoD**: `test_beta_symmetry.py` passes; numerical parity on small fixtures.

### I. Trainer Agent

* **Goal**: Train β as a **classification** to oracle flip index.
* **DoD**: Checkpoint `.npy`, logs (loss/acc), early‑stopping, CLI help.

### J. Evaluation Agent

* **Goal**: FER sweeps for SCL and DL‑SCL(β).
* **DoD**: Generates `plots/fer_M{M}.png` and CSV with results.

### K. Complexity Agent

* **Goal**: Op counts and pruning; produce a Markdown/CSV table.
* **DoD**: `results/opcount_M{M}.csv` and a short summary.

### L. Docs/QA Agent

* **Goal**: Keep `README.md` usage examples, update AGENTS.md if interfaces change, ensure tests are green in CI.
* **DoD**: `pytest -q` green; `README.md` runnable snippets.

---

## 5) Interfaces (APIs to honor)

### `polar/polar.py`

```python
encode(msg_bits: np.ndarray) -> np.ndarray
sc_decode(llr: np.ndarray, info_set: np.ndarray) -> np.ndarray
construct_info_set(N: int, K: int, method: str = "polarization") -> np.ndarray
```

### `polar/scl.py`

```python
decode_scl(llr: np.ndarray, info_set: np.ndarray, M: int,
           crc=None) -> dict  # {"candidates": List[np.ndarray], "metrics": List[float], "best_path_bits": np.ndarray}
```

### `polar/crc.py`

```python
attach_crc(msg_bits: np.ndarray, poly: str) -> np.ndarray
check_crc(msg_with_crc: np.ndarray, poly: str) -> bool
```

### `dlscl/beta.py`

```python
class SymmetricBeta(nn.Module):
    def __init__(self, dim: int, init_range: float = 0.2): ...
    def forward(self, abs_l0: torch.Tensor) -> torch.Tensor  # returns Q
    def clamp_diagonal(self): ...  # force diag to 1.0
```

### `dlscl/flip.py`

```python
def choose_flip_index(abs_l0: np.ndarray, beta: np.ndarray) -> int

def retry_with_flip(llr_root: np.ndarray, info_set: np.ndarray, M: int,
                    best_path_bits: np.ndarray, flip_index: int,
                    crc=None) -> dict
```

### `train/make_dataset.py`

* CLI: `--M 4 --snr_db 5 --frames 300000 --seed 0 --out data/train_M4_snr5_seed0`
* Output: `.npz` shards with schema in G (above). Oracle label is found by iterating candidate flip indices (sorted by placeholder metric) until a CRC pass occurs within 8 attempts.

### `train/train_beta.py`

* CLI: `--M 4 --data data/train_M4_snr5_* --epochs 8 --lr 1e-4 --batch 128 --lambda_l2 0.25`
* Saves: `checkpoints/beta_M4.npy`, `logs/train_M4.csv`.

### `eval/run_fer_sweep.py`

* CLI: `--M 4 --snr_lo 4.0 --snr_hi 6.5 --snr_step 0.5 --frames 10000 --retries 8 --beta checkpoints/beta_M4.npy`
* Output: `plots/fer_M4.png`, `results/fer_M4.csv`.

### `eval/opcount.py`

* CLI: `--beta checkpoints/beta_M4.npy --prune 1e-4 --report results/opcount_M4.csv`

---

## 6) Checklists per Milestone

**M0 Checklist**

* [ ] `encode` implements polar transform; SC works at high SNR.
* [ ] `test_polar_basics.py` green.

**M1 Checklist**

* [ ] CRC attach/check round‑trip on random messages.
* [ ] SCL recovers where SC fails (moderate SNR).
* [ ] `test_scl_crc.py` green.

**M2 Checklist**

* [ ] `retry_with_flip` fixes bits < i* and flips i*.
* [ ] With retries=0, equals baseline SCL output.
* [ ] `test_flip_logic.py` green.

**M3 Checklist**

* [ ] Dataset contains only SCL‑fail samples with successful oracle flip.
* [ ] β forward returns Q with correct shapes; diag==1.

**M4 Checklist**

* [ ] Training log shows decreasing loss; validation top‑1 improves.
* [ ] `.npy` saved per M.

**M5 Checklist**

* [ ] FER sweep runs and saves plot/CSV.
* [ ] DL‑SCL(β) improves over SCL at same M (or matches with retries).

**M6 Checklist**

* [ ] Opcount table before/after pruning saved.
* [ ] Report mentions that metric uses only adds/mults.

---

## 7) Minimal Commands (happy path)

```bash
# M0–M1
pytest -q

# M2 placeholder metric evaluation (no β)
python -m dl_scl_polar.eval.run_fer_sweep --M 4 --frames 2000 --retries 8 --snr_lo 5 --snr_hi 5 --snr_step 0

# M3 dataset
python -m dl_scl_polar.train.make_dataset --M 4 --frames 300000 --snr_db 5 --out data/train_M4_snr5

# M4 training
python -m dl_scl_polar.train.train_beta --M 4 --data data/train_M4_snr5 --epochs 8 --lr 1e-4 --batch 128

# M5 evaluation with β
python -m dl_scl_polar.eval.run_fer_sweep --M 4 --frames 10000 --retries 8 \
  --snr_lo 4.0 --snr_hi 6.5 --snr_step 0.5 --beta checkpoints/beta_M4.npy

# M6 complexity
python -m dl_scl_polar.eval.opcount --beta checkpoints/beta_M4.npy --prune 1e-4 \
  --report results/opcount_M4.csv
```

---

## 8) Guardrails & Tips

* Keep **bit index mapping** consistent among `polar`, `scl`, and `flip`.
* Always compute β on the **root‑stage abs LLR vector of info+CRC bits**.
* If labeling is too costly, cap oracle search to top‑K candidate indices.
* Discard training samples where no flip within 8 attempts passes CRC.
* Train **one β per M**; do not mix list sizes within a model.
* Add `--frames` caps to keep quick runs fast.

---

## 9) Acceptance: What counts as “reproduced”

* Plots show SCL baseline and DL‑SCL(β) curves; DL‑SCL is competitive (within ~0.1–0.2 dB around target FER levels) and uses only adds/mults for its metric.
* Opcount table demonstrates fewer mults/adds after pruning small β entries.

---

## 10) MASTER_PROMPT (for a code‑gen agent)

**Role**: You are a senior engineer implementing the `dl_scl_polar` package to exactly match the APIs and milestones in AGENTS.md.

**Objective**: Produce clean, tested code that passes `pytest` and runs the commands in Section 7. Do not change interfaces.

**Tasks**:

1. Create the repo layout in Section 1.
2. Implement M0→M6 in order, committing at each milestone.
3. Fill each file per the Interfaces in Section 5.
4. Add unit tests in `tests/` as listed; make them pass.
5. Provide a concise `README.md` with the commands from Section 7.

**Constraints**:

* Python 3.10+, NumPy, PyTorch only. No GPUs required.
* Deterministic seeding. No exotic dependencies.
* β metric uses only adds/mults; support pruning.

**Deliverables**:

* Working package + tests + plots + CSVs + opcount table.
* Checkpoints saved as `.npy` and reproducible via provided CLIs.

**Begin now** by scaffolding the repo and implementing M0 with tests. Then proceed milestone by milestone, running `pytest` after each step and fixing any failures before moving on.
