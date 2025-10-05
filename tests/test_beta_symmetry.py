import torch

from dl_scl_polar.dlscl.beta import SymmetricBeta


def test_beta_matrix_is_symmetric_with_unit_diag():
    beta = SymmetricBeta(dim=4)
    beta.clamp_diagonal()
    mat = beta.beta_matrix()
    assert torch.allclose(mat, mat.T)
    assert torch.allclose(torch.diag(mat), torch.ones(4, dtype=mat.dtype, device=mat.device))


def test_forward_supports_1d_and_2d_inputs():
    dim = 3
    beta = SymmetricBeta(dim)
    beta.clamp_diagonal()

    vec = torch.arange(1, dim + 1, dtype=torch.float32)
    out_vec = beta(vec)
    assert out_vec.shape == (dim,)

    mat_input = torch.stack([vec, 2 * vec])
    out = beta(mat_input)
    assert out.shape == (2, dim)

    # Check gradients propagate
    mat_input_grad = torch.stack([vec, 2 * vec]).requires_grad_()
    loss = beta(mat_input_grad).sum()
    loss.backward()
    assert mat_input_grad.grad is not None
