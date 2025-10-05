import numpy as np

from dl_scl_polar import config
from dl_scl_polar.utils.seeding import seed_all
from dl_scl_polar.polar.polar import construct_info_set, encode
from dl_scl_polar.polar.crc import attach_crc, check_crc
from dl_scl_polar.polar.scl import decode_scl
from dl_scl_polar.dlscl.flip import (
    choose_flip_index,
    decode_with_retries,
    retry_with_flip,
)


def _bpsk(codeword: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * codeword


def _awgn(symbols: np.ndarray, ebno_db: float, *, seed: int) -> np.ndarray:
    seed_all(seed)
    cfg = config.DEFAULTS
    rate = cfg.K / cfg.N
    ebno = 10 ** (ebno_db / 10)
    noise_var = 1 / (2 * rate * ebno)
    noise = np.random.normal(0.0, np.sqrt(noise_var), size=symbols.shape)
    received = symbols + noise
    return 2 * received / noise_var


def test_choose_flip_index_placeholder_metric():
    abs_l0 = np.array([0.8, 0.3, 1.5, 0.2])
    idx = choose_flip_index(abs_l0, beta=None)
    assert idx == 3


def test_retry_with_flip_enforces_prefix_and_flip():
    cfg = config.DEFAULTS
    info_set = construct_info_set(cfg.N, cfg.K)
    payload_bits = cfg.K - cfg.crc_bits

    seed_all(0)
    payload = np.random.randint(0, 2, size=payload_bits, dtype=np.int8)
    info = attach_crc(payload, cfg.crc_poly)
    code = encode(info)
    llr = _awgn(_bpsk(code), ebno_db=6.0, seed=10)

    baseline = decode_scl(llr, info_set, M=4, crc=cfg.crc_poly)
    best_bits = baseline["best_path_bits"]
    assert best_bits is not None
    assert check_crc(best_bits, cfg.crc_poly)

    flip_index = 5
    result = retry_with_flip(
        llr,
        info_set,
        M=4,
        best_path_bits=best_bits,
        flip_index=flip_index,
        crc=cfg.crc_poly,
    )

    forced = result["forced_info_bits"]
    assert forced[flip_index] == 1 - best_bits[flip_index]
    assert np.array_equal(forced[:flip_index], best_bits[:flip_index])

    for candidate in result["candidates"]:
        assert np.array_equal(candidate[:flip_index], best_bits[:flip_index])
        assert candidate[flip_index] == forced[flip_index]


def test_decode_with_retries_matches_baseline_when_zero_retries():
    cfg = config.DEFAULTS
    info_set = construct_info_set(cfg.N, cfg.K)
    payload_bits = cfg.K - cfg.crc_bits

    seed_all(100)
    payload = np.random.randint(0, 2, size=payload_bits, dtype=np.int8)
    info = attach_crc(payload, cfg.crc_poly)
    code = encode(info)
    llr = _awgn(_bpsk(code), ebno_db=2.0, seed=200)

    baseline = decode_scl(llr, info_set, M=4, crc=cfg.crc_poly)
    result = decode_with_retries(llr, info_set, M=4, retries=0, crc=cfg.crc_poly)

    np.testing.assert_array_equal(result["best_path_bits"], baseline["best_path_bits"])
    assert result["tried_indices"] == []
    assert result["attempts"][0]["attempt_type"] == "baseline"


def test_decode_with_retries_can_recover_failure():
    cfg = config.DEFAULTS
    info_set = construct_info_set(cfg.N, cfg.K)
    payload_bits = cfg.K - cfg.crc_bits

    recovered = False
    trials = 0
    for trials in range(1, 201):
        seed_all(1000 + trials)
        payload = np.random.randint(0, 2, size=payload_bits, dtype=np.int8)
        info = attach_crc(payload, cfg.crc_poly)
        code = encode(info)
        llr = _awgn(_bpsk(code), ebno_db=1.0, seed=2000 + trials)

        baseline = decode_scl(llr, info_set, M=2, crc=cfg.crc_poly)
        baseline_bits = baseline.get("best_path_bits")
        baseline_pass = baseline_bits is not None and check_crc(baseline_bits, cfg.crc_poly)

        result = decode_with_retries(llr, info_set, M=2, retries=4, crc=cfg.crc_poly)
        if not baseline_pass and result["success"] and np.array_equal(result["best_path_bits"], info):
            recovered = True
            break

    assert recovered, "Flip retries should eventually recover a CRC failure"
    assert trials <= 200
