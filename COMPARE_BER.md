# COMPARE_BER.md

A focused plan to extend the **dl_scl_polar** project with **BER comparisons** between:

* **DL‑SCL (your current repo)**
* **Polar SCL (5G‑style, rate‑matched)**
* **5G‑NR LDPC (BG1/BG2, layered normalized min‑sum)**

This file leaves **AGENTS.md** intact (paper replication) and layers comparative experiments on top.

---

## 0) Assumptions & Fairness Rules

* **Channel**: AWGN, **BPSK** with symbols ±1.
* **LLR**: (L = 2y/\sigma^2).
* **Energy normalization**: Given **Eb/N0 (dB)** and **payload rate** (R = K_\text{payload}/E), compute
  [\text{Es/N0}*\text{lin} = (\text{Eb/N0}*\text{lin})\times R,\qquad \sigma^2 = \frac{1}{2,\text{Es/N0}_\text{lin}}.]
  (BPSK has 1 bit/symbol; CRC bits are **overhead** and do not count toward payload.)
* **Payload parity**: Use the **same payload length** (K_\text{payload}) and **CRC length** across schemes unless otherwise noted.
* **Metric**: Primary **BER** (on payload bits only). Secondary **FER** for sanity.
* **Stopping rule per SNR**: simulate until **bit_errors ≥ 1000** or **bits_total ≥ 1e7**.
* **Seeds**: Deterministic seeding across all schemes.

---

## 1) Repo Additions

```
dl_scl_polar/
  nr/
    polar/
      __init__.py
      interleaver.py        # 5G sub‑block interleaver (configurable)
      rate_match.py         # bit selection (puncture/repeat) + inverse
      scl_nr.py             # encode_rate_matched(), decode_rate_matched_scl()
    ldpc/
      __init__.py
      basegraphs.py         # BG1/BG2 base graph description
      builder.py            # Build sparse H with lifting Z (circulant shifts)
      rate_match.py         # puncturing/shortening/repetition + inverse
      encode.py             # systematic LDPC encoding
      decode_nms.py         # layered normalized min‑sum (alpha, max_iter, ET)
  eval/
    run_ber_sweep.py        # unified BER evaluator
    compare_ber.py          # merge CSVs, make plots
  configs/
    ber_suites.yaml         # optional: saved experiment suites
```

---

## 2) Milestones

### M7 — Add BER Evaluation (no new codes yet)

**Goal**: Enable BER sweeps for existing Polar SCL & DL‑SCL.

**Tasks**

1. `eval/run_ber_sweep.py` (or extend existing):

   * `--scheme {polar_scl, dl_scl, nr_polar_scl, nr_ldpc}`
   * Common args: `--K_payload`, `--K_crc`, `--E` (coded length), `--EbN0_lo`, `--EbN0_hi`, `--EbN0_step`, `--bits_cap`, `--err_cap`, `--seed`.
   * DL‑SCL extras: `--M`, `--retries`, `--beta`.
   * Writer: CSV with columns
     `scheme,code,N_or_E,K_payload,K_crc,rate,params,EbN0_dB,bits_total,bit_errors,ber,fer,avg_work`.

     * `params`: e.g., `M=4,retries=8` for Polar; `iters=20,alpha=0.8` for LDPC.
     * `avg_work`: Polar: mean **retries used**; LDPC: mean **iterations**.
2. Sanity: Compare DL‑SCL vs SCL at same M; curves should align at high SNR.

**Acceptance**

* CSV + plot for Polar/DL‑SCL produced; payload‑only BER confirmed by unit test.

---

### M8 — 5G‑Style Polar SCL (rate‑matched)

**Goal**: Add NR‑Polar pre/post processing around your SCL to compare with DL‑SCL.

**Encoder side**

1. `interleaver.py`: sub‑block interleaver function `interleave(x, E, mode)` and `deinterleave(...)`.
2. `rate_match.py`:

   * `rate_match_polar(bits, E, mode) -> bits_E` supports puncturing/repetition;
   * `derate_match_polar(bits_E, N, mode) -> bits_N`.
3. `scl_nr.py`:

   * `encode_rate_matched(msg_payload, crc_type, N, E, info_set, ilv_mode)`:
     attach CRC → polar encode to length `N` → interleave → rate‑match to `E`.
   * `decode_rate_matched_scl(llr_E, crc_type, N, E, info_set, M, ilv_mode)`:
     de‑rate‑match → de‑interleave → SCL → CRC check.

**Tests**

* Noiseless round‑trip (encode→rate‑match→de‑rate→decode) recovers payload.
* With moderate SNR, BER finite and improves with M.

**Acceptance**

* BER curve generated for two `E` values (e.g., 128 and 256) at `K_payload=64, K_crc=24`.

---

### M9 — 5G‑NR LDPC Baseline (BG1/BG2)

**Goal**: Add systematic encoder + layered normalized min‑sum decoder with rate matching.

**Builder**

1. `basegraphs.py`: describe BG1/BG2 as sparse templates (per‑edge shifts).
2. `builder.py`: `build_H(bg, Z)` to generate sparse parity‑check `H` by applying circulant shifts for lifting `Z`.

**Rate match**

* `rate_match_ldpc(sys_bits, parity_bits, E) -> codeword_E` with puncture/shorten/repeat; and inverse.

**Encoder**

* `encode_ldpc(payload, crc, H) -> codeword` using systematic encoding (solve parity via back‑substitution / sparse ops).

**Decoder**

* `decode_ldpc_nms(llr_E, H, max_iter, alpha, early_stop=True)`:
  layered schedule; normalized min‑sum with scaling `alpha` (e.g., 0.8/0.9); parity early termination.

**Tests**

* Noiseless parity check: `H @ codeword^T = 0`.
* At high SNR, decoder converges to zero BER; average iteration count < max.

**Acceptance**

* Two LDPC configs (e.g., BG2@Z=32,E=512 and BG1@Z=24,E=384) produce BER curves.

---

### M10 — Unified Comparison & Plotting

**Goal**: Merge CSVs and render BER plots.

**compare_ber.py**

* Ingest multiple CSVs, normalize labels, compute envelopes.
* Plot `BER vs Eb/N0` (log‑scale y), color by scheme, marker by config.
* Optional: Plot `avg_work vs Eb/N0` on a separate figure.

**Acceptance**

* `plots/ber_all.png`, `plots/work_all.png`, and combined CSV saved.

---

## 3) Public APIs (to keep stable)

### `eval/run_ber_sweep.py`

```bash
python -m dl_scl_polar.eval.run_ber_sweep \
  --scheme {polar_scl,dl_scl,nr_polar_scl,nr_ldpc} \
  --K_payload 64 --K_crc 24 --E 128 \
  --EbN0_lo 1.0 --EbN0_hi 6.5 --EbN0_step 0.5 \
  --bits_cap 10000000 --err_cap 1000 --seed 0 \
  [--M 4 --retries 8 --beta checkpoints/beta_M4.npy] \
  [--bg 2 --Z 32 --max_iter 20 --alpha 0.8] \
  --out results/ber_<name>.csv
```

### `nr/polar/scl_nr.py`

```python
def encode_rate_matched(payload: np.ndarray, crc_type: str, N: int, E: int,
                        info_set: np.ndarray, ilv_mode: str) -> np.ndarray

def decode_rate_matched_scl(llr_E: np.ndarray, crc_type: str, N: int, E: int,
                            info_set: np.ndarray, M: int, ilv_mode: str) -> dict
# returns {"payload": np.ndarray, "crc_pass": bool, "avg_work": float}
```

### `nr/ldpc/decode_nms.py`

```python
def decode_ldpc_nms(llr_E, H, max_iter: int = 20, alpha: float = 0.8,
                    early_stop: bool = True) -> dict
# returns {"hard": np.ndarray, "iters_used": int, "parity_ok": bool}
```

---

## 4) Unit Tests

* **BER counting**: a synthetic payload with forced errors yields the expected BER.
* **NR‑Polar**: noiseless encode→rate‑match→decode recovers payload; interleaver invertible.
* **LDPC**: `H` construction correct for several `(bg,Z)`; parity checks pass after encoding; decoder converges at high SNR.
* **Energy model**: for identical payload and rate, Es/N0 and σ mapping consistent across schemes.
* **DL‑SCL invariants**: β inference path uses only adds/mults; retries==0 equals SCL.

---

## 5) Reference Suites (examples)

**Payload/CRC:** `K_payload=64`, `K_crc=24`.

**Polar (no RM)**: `N=128`, `M∈{1,4}`, DL‑SCL with `retries=8` and trained β.

**NR‑Polar (RM)**: `E∈{128,256}`, same `M`.

**LDPC**: two configs, e.g.,

* `BG2, Z=32, E=512, max_iter=20, alpha=0.8` (higher redundancy)
* `BG1, Z=24, E=384, max_iter=20, alpha=0.9` (closer to Polar rate)

SNR grid: `EbN0_dB in {1.0, 1.5, …, 6.5}`.

---

## 6) Command Cheatsheet

```bash
# Polar SCL (baseline)
python -m dl_scl_polar.eval.run_ber_sweep --scheme polar_scl \
  --K_payload 64 --K_crc 24 --E 128 --M 4 \
  --EbN0_lo 1.0 --EbN0_hi 6.5 --EbN0_step 0.5 \
  --bits_cap 1e7 --err_cap 1000 --out results/ber_polar_M4.csv

# DL‑SCL
python -m dl_scl_polar.eval.run_ber_sweep --scheme dl_scl \
  --K_payload 64 --K_crc 24 --E 128 --M 4 --retries 8 \
  --beta checkpoints/beta_M4.npy \
  --EbN0_lo 1.0 --EbN0_hi 6.5 --EbN0_step 0.5 \
  --bits_cap 1e7 --err_cap 1000 --out results/ber_dlscl_M4.csv

# NR‑Polar SCL (rate‑matched)
python -m dl_scl_polar.eval.run_ber_sweep --scheme nr_polar_scl \
  --K_payload 64 --K_crc 24 --E 256 --M 4 --crc_type 24A --ilv_mode default \
  --EbN0_lo 1.0 --EbN0_hi 6.5 --EbN0_step 0.5 \
  --bits_cap 1e7 --err_cap 1000 --out results/ber_nrpolar_E256_M4.csv

# 5G‑NR LDPC
python -m dl_scl_polar.eval.run_ber_sweep --scheme nr_ldpc \
  --K_payload 64 --K_crc 24 --bg 2 --Z 32 --E 512 \
  --max_iter 20 --alpha 0.8 \
  --EbN0_lo 1.0 --EbN0_hi 6.5 --EbN0_step 0.5 \
  --bits_cap 1e7 --err_cap 1000 --out results/ber_nrldpc_cfgA.csv

# Merge & plot
python -m dl_scl_polar.eval.compare_ber --inputs results/*.csv --out_plot plots/ber_all.png
```

---

## 7) Acceptance Criteria

* All tests green (including new modules).
* For each scheme/config, a BER CSV exists with ≥ two SNR points having ≥1000 bit errors or ≥1e7 bits.
* Plots `ber_all.png` and (optional) `work_all.png` saved.
* `avg_work` behaves plausibly: LDPC iterations drop with SNR; DL‑SCL avg retries finite and ≤ max.

---

## 8) MASTER_PROMPT (for a code‑gen agent)

**Role**: Implement the features in COMPARE_BER.md without changing AGENTS.md interfaces.

**Objective**: Add NR‑Polar and NR‑LDPC baselines and a BER evaluator, then generate CSVs/plots for direct comparison with DL‑SCL.

**Tasks**:

1. Implement M7 (BER evaluator) and unit tests. Keep payload‑only BER.
2. Implement M8 (NR‑Polar): interleaver, rate‑match, wrappers around existing SCL. Add noiseless round‑trip tests.
3. Implement M9 (NR‑LDPC): basegraphs, builder (lifting), encoder, rate‑match, layered normalized min‑sum decoder with early stop. Add tests (parity/decoding sanity).
4. Implement M10 (comparison merging & plotting).
5. Provide CLI commands from Section 6 in README.

**Constraints**:

* Python 3.10+, NumPy, PyTorch only; CPU‑friendly; deterministic seeds.
* Keep β inference adds/mults only; do not alter existing DL‑SCL APIs.

**Deliverables**:

* New modules under `nr/` and `eval/`.
* Passing unit tests.
* BER CSVs and plots for the suites in Section 5.

**Begin now** with M7; run tests; proceed milestone by milestone.
