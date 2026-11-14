import numpy as np
import pennylane as qml
import pytest
from numpy.polynomial import chebyshev as C
from scipy.linalg import expm

import qsvtlab as ql


# ---------------------------------------------------------------- phase solver
@pytest.mark.parametrize("kappa", [3, 10, 30])
def test_phase_solver_high_degree_inverse(kappa):
    c, _ = ql.rescale(ql.inverse_poly(kappa, 1e-3), 0.5)
    res = ql.solve_qsp_phases(c)
    x = np.linspace(-1, 1, 1001)
    assert np.abs(ql.qsp_response(res.phases, x) - C.chebval(x, c)).max() < 1e-9


def test_phase_solver_beats_pennylane_builtin():
    """PennyLane's monomial-basis solvers fail where ours succeeds."""
    c, _ = ql.rescale(ql.inverse_poly(10, 1e-2), 0.5)
    mono = C.cheb2poly(c)
    with pytest.raises(Exception):
        qml.poly_to_angles(mono, "QSVT")  # root-finding
    ql.solve_qsp_phases(c)  # must not raise


def test_phase_solver_newton_fallback():
    c = np.array([0.0, 0.95, 0.0, -0.0])  # ||c||_1 = 0.95 > FPI radius
    res = ql.solve_qsp_phases(c)
    assert res.residual < 1e-10


# ---------------------------------------------------------------- circuits
@pytest.fixture
def herm():
    rng = np.random.default_rng(7)
    M = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    return (M + M.conj().T) / 2


@pytest.mark.parametrize(
    "coeffs",
    [
        np.array([0, 0.3, 0, -0.2]),
        np.array([0.1, 0, 0.2, 0, -0.15]),
        ql.rescale(ql.sin_poly(3.0, 15))[0],
        ql.rescale(ql.cos_poly(3.0, 16))[0],
    ],
)
def test_real_qsvt_block_is_exact(herm, coeffs):
    be = ql.unitary_dilation(herm, [1, 2], ancilla_wire=0)
    ev, V = np.linalg.eigh(herm / be.alpha)
    target = V @ np.diag(C.chebval(ev, coeffs)) @ V.conj().T
    block = ql.qsvt_block(be, ql.qsvt_phases_for(coeffs), lcu_wire="r")
    assert np.allclose(block, target, atol=1e-9)


def test_lcu_block_encoding_normalization():
    H = ql.tfim(3)
    be = ql.lcu_block_encoding(H, ["a0", "a1", "a2"])
    U = qml.matrix(be.make_op, wire_order=be.wires)()
    N = be.dim
    assert np.allclose(U[:N, :N] * be.alpha, qml.matrix(H, wire_order=be.system_wires), atol=1e-10)


def test_blockencode_rescaling_pitfall():
    """Documents the qml.BlockEncode normalization pitfall that motivated unitary_dilation."""
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    A = Q @ np.diag([1, -0.5, 0.3, -0.2]) @ Q.T  # spectral norm exactly 1
    U = qml.matrix(qml.BlockEncode(A, wires=[0, 1, 2]))
    assert not np.allclose(U[:4, :4], A, atol=1e-3)


# ---------------------------------------------------------------- linear solver
def test_linear_solver_hermitian():
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    A = Q @ np.diag(np.r_[np.linspace(0.1, 1, 4), -np.linspace(0.15, 0.9, 4)]) @ Q.T
    b = rng.normal(size=8)
    r = ql.qsvt_solve(A, b, eps=1e-3)
    assert r.fidelity > 1 - 1e-6
    assert abs(r.norm_estimate / r.norm_exact - 1) < 1e-3


def test_linear_solver_non_hermitian():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(4, 4)) + 2 * np.eye(4)
    b = rng.normal(size=4)
    r = ql.qsvt_solve(A, b, eps=1e-3)
    x = np.linalg.solve(A, b)
    assert abs(np.vdot(x / np.linalg.norm(x), r.state)) ** 2 > 1 - 1e-6


def test_linear_solver_lcu():
    H = qml.dot([1.0, 0.5, 0.4, 0.3], [qml.Z(0) @ qml.Z(1), qml.X(0), qml.X(1), qml.Z(0)])
    r = ql.qsvt_solve(H, np.ones(4), eps=1e-3)
    assert r.fidelity > 1 - 1e-6


# ---------------------------------------------------------------- Hamiltonian simulation
def test_qsvt_time_evolution():
    H = ql.tfim(3)
    r = ql.qsvt_time_evolution(H, 1.0, eps=1e-6)
    assert r.error < 1e-5


def test_trotter_converges():
    H = ql.tfim(3)
    e1, e2 = ql.trotter_error(H, 1.0, 4, 2), ql.trotter_error(H, 1.0, 8, 2)
    assert e2 < e1 / 3  # second order: ~4x per doubling


# ---------------------------------------------------------------- chemistry
from numpy.polynomial import chebyshev as _C  # noqa: E402

from qsvtlab.chemistry import EigenFilter, filter_poly, molecule_hamiltonian  # noqa: E402


@pytest.fixture(scope="module")
def h2_stretched():
    H, n, hf = molecule_hamiltonian(["H", "H"], [[0, 0, 0], [0, 0, 2.0]])
    return EigenFilter(H, hf)


def test_filter_poly_properties():
    ell, delta = 60, 0.05
    c = filter_poly(ell, delta, s=1.0)
    x = np.linspace(-1, 1, 20001)
    v = _C.chebval(x, c)
    assert abs(_C.chebval(0.0, c) - 1) < 1e-12
    assert np.abs(v).max() <= 1 + 1e-9
    assert np.abs(v[np.abs(x) >= delta]).max() <= 2 * np.exp(-np.sqrt(2) * ell * delta)


def test_filter_circuit_matches_closed_form(h2_stretched):
    F = h2_stretched
    for mu in (-1.2, -0.95, -0.4):
        assert abs(F.success_probability(mu, 40, 0.05) - F.p_success_exact(mu, 40, 0.05)) < 1e-9


def test_ground_energy_and_state(h2_stretched):
    F = h2_stretched
    E0, sc, sf = F.estimate_ground_energy(coarse=(60, 0.05, 81), fine=(200, 0.015, 21))
    e_exact = np.linalg.eigvalsh(F.Hmat)[0]
    assert abs(E0 - e_exact) < 1e-5  # far below chemical accuracy (1.6e-3)
    g = F.prepare_ground_state(E0, ell=200)
    assert g.fidelity > 1 - 1e-8
    assert abs(sc.weights[0] - g.hf_overlap) < 0.02  # peak height = |<E0|HF>|^2


# ---------------------------------------------------------------- crossover analysis
def test_suzuki_matches_pennylane():
    from qsvtlab.hamiltonian_sim import suzuki_step

    H = ql.tfim(3)
    for order in (1, 2, 4):
        U = np.linalg.matrix_power(suzuki_step(H, -0.25, order), 4)  # PennyLane: exp(+iHt)
        V = qml.matrix(qml.TrotterProduct(H, 1.0, n=4, order=order), wire_order=[0, 1, 2])
        assert np.allclose(U, V, atol=1e-12)


def test_truncation_bound_is_tight():
    from qsvtlab.hamiltonian_sim import qsvt_truncation_error

    r = ql.qsvt_time_evolution(ql.tfim(3), 1.0, eps=1e-6)
    bound = qsvt_truncation_error(r.tau, r.deg_cos, r.deg_sin)
    assert r.error <= bound * (1 + 1e-6)
    assert r.error > 0.5 * bound
