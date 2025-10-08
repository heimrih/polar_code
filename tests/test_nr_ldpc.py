import numpy as np

from dl_scl_polar.nr.ldpc import (
    load_base_graph,
    build_h_matrix,
    encode_ldpc,
    rate_match_ldpc,
    derate_match_ldpc,
    decode_ldpc_nms,
)


def test_ldpc_parity_check():
    bg = load_base_graph(2)
    Z = 4
    H = build_h_matrix(bg, Z)
    k = H.shape[1] - H.shape[0]
    payload = np.random.randint(0, 2, size=k, dtype=np.int8)
    codeword = encode_ldpc(payload, H)
    syndrome = (H @ codeword) % 2
    assert not syndrome.any()


def test_ldpc_rate_match_roundtrip():
    bg = load_base_graph(2)
    Z = 2
    H = build_h_matrix(bg, Z)
    k = H.shape[1] - H.shape[0]
    payload = np.zeros(k, dtype=np.int8)
    codeword = encode_ldpc(payload, H)
    rm = rate_match_ldpc(codeword, codeword.size + 5)
    der = derate_match_ldpc(rm.astype(np.float64), codeword.size)
    assert der.size == codeword.size


def test_ldpc_decode_high_snr():
    bg = load_base_graph(2)
    Z = 4
    H = build_h_matrix(bg, Z)
    k = H.shape[1] - H.shape[0]
    payload = np.random.randint(0, 2, size=k, dtype=np.int8)
    codeword = encode_ldpc(payload, H)
    symbols = 1.0 - 2.0 * codeword
    noise = np.random.normal(0.0, 0.05, size=symbols.shape)
    llr = 2.0 * (symbols + noise) / (0.05 ** 2)
    result = decode_ldpc_nms(llr, H, max_iter=10, alpha=0.9)
    assert result["parity_ok"]
    hard = result["hard"]
    np.testing.assert_array_equal(hard[:k], payload)
