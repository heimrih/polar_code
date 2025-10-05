import numpy as np

from dl_scl_polar.utils.seeding import seed_all
from dl_scl_polar import config
from dl_scl_polar.polar.polar import construct_info_set, encode, sc_decode
from dl_scl_polar.polar.crc import attach_crc, check_crc
from dl_scl_polar.polar.scl import decode_scl


def _bpsk(codeword: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * codeword


def _awgn(llr_seed: int, symbols: np.ndarray, ebno_db: float) -> np.ndarray:
    seed_all(llr_seed)
    cfg = config.DEFAULTS
    rate = cfg.K / cfg.N
    ebno = 10 ** (ebno_db / 10)
    noise_var = 1 / (2 * rate * ebno)
    noise = np.random.normal(0.0, np.sqrt(noise_var), size=symbols.shape)
    received = symbols + noise
    return 2 * received / noise_var


def test_crc_roundtrip():
    cfg = config.DEFAULTS
    payload_bits = cfg.K - cfg.crc_bits
    seed_all(7)
    msg = np.random.randint(0, 2, size=payload_bits, dtype=np.int8)
    msg_crc = attach_crc(msg, cfg.crc_poly)
    assert msg_crc.shape[0] == cfg.K
    assert check_crc(msg_crc, cfg.crc_poly)

    # Random bit flip should break CRC
    corrupted = msg_crc.copy()
    corrupted[3] ^= 1
    assert not check_crc(corrupted, cfg.crc_poly)


def test_scl_recovers_over_sc():
    cfg = config.DEFAULTS
    info_set = construct_info_set(cfg.N, cfg.K)
    payload_bits = cfg.K - cfg.crc_bits

    attempts = 0
    successes = 0
    while attempts < 300 and successes == 0:
        seed_all(1000 + attempts)
        payload = np.random.randint(0, 2, size=payload_bits, dtype=np.int8)
        info = attach_crc(payload, cfg.crc_poly)
        code = encode(info)
        symbols = _bpsk(code)
        llr = _awgn(2000 + attempts, symbols, ebno_db=2.0)

        sc_candidate = sc_decode(llr, info_set)
        sc_info = sc_candidate.copy()
        sc_pass = check_crc(sc_info, cfg.crc_poly)

        scl_result = decode_scl(llr, info_set, M=4, crc=cfg.crc_poly)
        scl_info = scl_result["best_path_bits"]
        if scl_info is None:
            attempts += 1
            continue
        scl_pass = check_crc(scl_info, cfg.crc_poly)

        if scl_pass and not sc_pass and np.array_equal(scl_info, info):
            successes += 1
        attempts += 1

    assert successes == 1, "SCL should find a CRC-valid codeword when SC fails"
