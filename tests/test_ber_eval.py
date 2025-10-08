import numpy as np
import pytest

from dl_scl_polar.eval import run_ber_sweep


def test_payload_bit_errors_ignores_crc_only():
    payload = np.array([0, 1, 1, 0], dtype=np.int8)
    candidate = np.concatenate([payload, np.array([1, 0, 0, 1], dtype=np.int8)])
    candidate[-1] ^= 1  # flip only CRC bit
    assert run_ber_sweep._payload_bit_errors(payload, candidate, payload.size) == 0


def test_payload_bit_errors_all_when_missing_candidate():
    payload = np.array([0, 1, 0], dtype=np.int8)
    assert run_ber_sweep._payload_bit_errors(payload, None, payload.size) == payload.size


def test_run_polar_scl_small_config():
    args = run_ber_sweep.parse_args(
        [
            "--scheme",
            "polar_scl",
            "--K_payload",
            "8",
            "--K_crc",
            "4",
            "--E",
            "16",
            "--crc_poly",
            "0x17",
            "--M",
            "2",
            "--EbN0_lo",
            "6.0",
            "--EbN0_hi",
            "6.0",
            "--EbN0_step",
            "0.5",
            "--bits_cap",
            "64",
            "--err_cap",
            "2",
            "--out",
            "results/tmp_test.csv",
        ]
    )
    rows = run_ber_sweep.run(args)
    assert len(rows) == 1
    row = rows[0]
    assert row["scheme"] == "polar_scl"
    assert row["K_payload"] == 8
    assert row["rate"] == pytest.approx(0.5)
    assert row["bits_total"] > 0
    assert row["ber"] >= 0.0
    assert row["avg_work"] == 0.0


def test_nr_polar_args_parsed():
    args = run_ber_sweep.parse_args(
        [
            "--scheme",
            "nr_polar_scl",
            "--K_payload",
            "8",
            "--K_crc",
            "4",
            "--E",
            "16",
            "--N",
            "16",
            "--M",
            "2",
            "--EbN0_lo",
            "5.0",
            "--EbN0_hi",
            "5.0",
            "--bits_cap",
            "64",
            "--err_cap",
            "2",
            "--out",
            "results/tmp_nrpolar.csv",
            "--crc_poly",
            "0x17",
        ]
    )
    rows = run_ber_sweep.run(args)
    assert rows[0]["scheme"] == "nr_polar_scl"


def test_nr_ldpc_args_parsed():
    args = run_ber_sweep.parse_args(
        [
            "--scheme",
            "nr_ldpc",
            "--K_payload",
            "6",
            "--K_crc",
            "0",
            "--E",
            "12",
            "--bg",
            "2",
            "--Z",
            "2",
            "--EbN0_lo",
            "5.0",
            "--EbN0_hi",
            "5.0",
            "--bits_cap",
            "64",
            "--err_cap",
            "2",
            "--out",
            "results/tmp_nrldpc.csv",
            "--crc_poly",
            "0x1",
        ]
    )
    rows = run_ber_sweep.run(args)
    assert rows[0]["scheme"] == "nr_ldpc"
