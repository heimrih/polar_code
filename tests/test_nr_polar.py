import numpy as np

from dl_scl_polar.nr.polar import (
    subblock_interleave,
    subblock_deinterleave,
    rate_match_polar,
    derate_match_polar,
    encode_rate_matched,
    decode_rate_matched_scl,
)
from dl_scl_polar.polar.polar import construct_info_set
from dl_scl_polar import config
from dl_scl_polar.utils.seeding import seed_all


def test_subblock_interleaver_invertible():
    bits = np.arange(40) % 2
    inter = subblock_interleave(bits)
    deinter = subblock_deinterleave(inter, bits.size)
    np.testing.assert_array_equal(bits, deinter)


def test_rate_match_roundtrip_llr_average():
    bits = np.arange(16) % 2
    rm = rate_match_polar(bits, 24)
    assert rm.size == 24
    llr = np.linspace(-1, 1, 24)
    der = derate_match_polar(llr, 16)
    assert der.size == 16


def test_nr_polar_noiseless_roundtrip():
    cfg = config.DEFAULTS
    payload_len = cfg.K - cfg.crc_bits
    info_set = construct_info_set(cfg.N, cfg.K)
    payload = np.random.randint(0, 2, size=payload_len, dtype=np.int8)
    tx = encode_rate_matched(payload, cfg.crc_poly, cfg.N, cfg.N, info_set)
    llr = np.where(tx == 0, 50.0, -50.0)
    result = decode_rate_matched_scl(llr, cfg.crc_poly, cfg.N, cfg.N, info_set, M=4)
    assert result["crc_pass"]
    np.testing.assert_array_equal(result["payload"][:payload_len], payload)


def test_nr_polar_awgn_recovery():
    seed_all(123)
    cfg = config.DEFAULTS
    payload_len = cfg.K - cfg.crc_bits
    info_set = construct_info_set(cfg.N, cfg.K)

    payload = np.random.randint(0, 2, size=payload_len, dtype=np.int8)
    tx = encode_rate_matched(payload, cfg.crc_poly, cfg.N, cfg.N, info_set)
    symbols = 1.0 - 2.0 * tx
    noise = np.random.normal(0.0, 0.3, size=symbols.shape)
    llr = 2.0 * (symbols + noise) / (0.3 ** 2)
    result = decode_rate_matched_scl(llr, cfg.crc_poly, cfg.N, cfg.N, info_set, M=4)
    assert result["crc_pass"]
