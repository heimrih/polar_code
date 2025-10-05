from dl_scl_polar.utils.seeding import seed_all
from dl_scl_polar import config
from dl_scl_polar.polar.polar import construct_info_set, encode, sc_decode

import torch
import numpy as np


def bpsk_mod(codeword: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * codeword


def test_noiseless_round_trip():
    seed_all(0)
    cfg = config.DEFAULTS
    info_set = construct_info_set(cfg.N, cfg.K)
    msg = np.random.randint(0, 2, size=cfg.K, dtype=np.int8)
    code = encode(msg)
    assert code.shape == (cfg.N,)

    llr = np.where(code == 0, 1.0, -1.0) * 1e6
    decoded = sc_decode(llr, info_set)
    np.testing.assert_array_equal(decoded, msg)


def test_sc_decode_high_snr_awgn():
    seed_all(123)
    cfg = config.DEFAULTS
    info_set = construct_info_set(cfg.N, cfg.K)
    msg = np.random.randint(0, 2, size=cfg.K, dtype=np.int8)
    code = encode(msg)
    symbols = bpsk_mod(code)

    ebno_db = 6.0
    rate = cfg.K / cfg.N
    ebno = 10 ** (ebno_db / 10)
    noise_var = 1 / (2 * rate * ebno)
    noise = np.random.normal(0.0, np.sqrt(noise_var), size=symbols.shape)
    received = symbols + noise
    llr = 2 * received / noise_var

    decoded = sc_decode(llr, info_set)
    np.testing.assert_array_equal(decoded, msg)
