from e3nn import o3

from mace.tools import cg


def test_U_matrix():
    irreps_in = o3.Irreps("1x0e + 1x1o + 1x2e")
    irreps_out = o3.Irreps("1x0e + 1x1o")
    u_matrix = cg.U_matrix_real(
        irreps_in=irreps_in, irreps_out=irreps_out, correlation=3, use_cueq_cg=False
    )[-1]
    assert u_matrix.shape == (3, 9, 9, 9, 21)
