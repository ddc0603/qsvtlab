r"""QSVT Hamiltonian simulation vs. Trotterization.

:math:`e^{-iHt} = \cos(Ht) - i\sin(Ht)`. With an LCU block encoding of :math:`H/\lambda`, both
parts are Jacobi-Anger polynomials in :math:`H/\lambda` with :math:`\tau = \lambda t`. Each one
is implemented with :func:`qsvt_phases_for`, and the two are combined through one more
ancilla (``sel``):

.. math::
    \langle +|_{\rm sel}\,\big(|0\rangle\langle0|\otimes U_{\cos} + |1\rangle\langle1|\otimes(-i)U_{\sin}\big)\,|+\rangle_{\rm sel}
    = \tfrac{s}{2}\, e^{-iHt}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
from scipy.linalg import expm

from .block_encodings import lcu_block_encoding
from .circuits import qsvt_phases_for, real_qsvt
from .polynomials import cos_poly, jacobi_anger_degree, sin_poly

__all__ = [
    "tfim",
    "heisenberg",
    "QSVTSimResult",
    "qsvt_time_evolution",
    "trotter_error",
    "count_gates",
    "qsvt_cnot_cost",
    "trotter_cnot_cost",
    "qsvt_truncation_error",
    "qsvt_min_cost",
    "suzuki_step",
    "trotter_min_steps",
    "trotter_min_cost",
]

GATE_SET = {
    "CNOT", "RX", "RY", "RZ", "Hadamard", "PauliX", "PauliY", "PauliZ", "S", "T",
    "Adjoint(S)", "Adjoint(T)", "GlobalPhase", "PhaseShift",
}


def tfim(n: int, J: float = 1.0, h: float = 0.7, periodic: bool = False):
    """Transverse-field Ising chain H = -J sum Z_i Z_{i+1} - h sum X_i."""
    pairs = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if periodic and n > 2 else [])
    return qml.dot(
        [-J] * len(pairs) + [-h] * n,
        [qml.Z(i) @ qml.Z(j) for i, j in pairs] + [qml.X(i) for i in range(n)],
    )


def heisenberg(n: int, J: float = 1.0, hz: float = 0.5):
    """XXX Heisenberg chain with a staggered Z field."""
    coeffs, ops = [], []
    for i in range(n - 1):
        for P in (qml.X, qml.Y, qml.Z):
            coeffs.append(J)
            ops.append(P(i) @ P(i + 1))
    for i in range(n):
        coeffs.append(hz * (-1) ** i)
        ops.append(qml.Z(i))
    return qml.dot(coeffs, ops)


@dataclass
class QSVTSimResult:
    unitary: np.ndarray  # rescaled post-selected block ≈ e^{-iHt}
    error: float  # spectral-norm error vs exact
    deg_cos: int
    deg_sin: int
    tau: float
    success_prob_bound: float  # (s/2)^2 for a worst-case input


def _jacobi_anger_pair(tau, eps, s):
    R = jacobi_anger_degree(tau, eps)
    return s * cos_poly(tau, R + (R % 2)), s * sin_poly(tau, R + 1 - (R % 2))


def qsvt_time_evolution(H, t: float, eps: float = 1e-6, s: float = 0.5) -> QSVTSimResult:
    """Build the full QSVT e^{-iHt} circuit, post-select, and compare with expm."""
    coeffs, _ = H.terms()
    sys_wires = list(H.wires)
    anc = [f"a{i}" for i in range(int(np.ceil(np.log2(len(coeffs)))))]
    be = lcu_block_encoding(H, anc)
    tau = be.alpha * t
    c_cos, c_sin = _jacobi_anger_pair(tau, eps, s)
    ph_cos, ph_sin = qsvt_phases_for(c_cos), qsvt_phases_for(c_sin)

    sel, lcu = "sel", "r"
    wires = [sel, lcu] + be.wires

    def circuit():
        qml.Hadamard(sel)
        qml.ctrl(real_qsvt, control=sel, control_values=0)(be, ph_cos, lcu)
        qml.ctrl(real_qsvt, control=sel, control_values=1)(be, ph_sin, lcu)
        qml.PhaseShift(-np.pi / 2, wires=sel)  # (-i) on the sin branch
        qml.Hadamard(sel)

    U = qml.matrix(circuit, wire_order=wires)()
    N = 2 ** len(sys_wires)
    block = U[:N, :N] * (2.0 / s)
    exact = expm(-1j * qml.matrix(H, wire_order=sys_wires) * t)
    err = float(np.linalg.norm(block - exact, 2))
    return QSVTSimResult(block, err, len(c_cos) - 1, len(c_sin) - 1, tau, (s / 2) ** 2)


def trotter_error(H, t: float, steps: int, order: int) -> float:
    sys_wires = list(H.wires)
    # NB: qml.TrotterProduct(H, t) approximates exp(+iHt), so pass -t for exp(-iHt).
    U = qml.matrix(qml.TrotterProduct(H, -t, n=steps, order=order), wire_order=sys_wires)
    exact = expm(-1j * qml.matrix(H, wire_order=sys_wires) * t)
    return float(np.linalg.norm(U - exact, 2))


def count_gates(fn) -> dict:
    """Decompose a quantum function into {CNOT, single-qubit} and count the gates."""
    tape = qml.tape.make_qscript(fn)()
    [tape2], _ = qml.transforms.decompose(tape, gate_set=GATE_SET)
    counts: dict = {}
    for op in tape2.operations:
        counts[op.name] = counts.get(op.name, 0) + 1
    return counts


def qsvt_cnot_cost(H, deg_cos: int, deg_sin: int) -> int:
    r"""CNOT cost model for QSVT simulation.

    Each block-encoding query costs one PrepSelPrep. The cos and sin sequences share their
    queries: the branch is selected by controlling only the projector phases on ``sel``, and
    the shorter sequence is padded with an identity phase. That gives
    :math:`\max(d_{\cos}, d_{\sin})` queries plus :math:`\max+1` projector phases, each
    controlled on (``sel``, ``r``).
    """
    coeffs, _ = H.terms()
    anc = [f"a{i}" for i in range(int(np.ceil(np.log2(len(coeffs)))))]
    be = lcu_block_encoding(H, anc)
    q = count_gates(be.make_op).get("CNOT", 0)
    proj = count_gates(
        lambda: qml.ctrl(
            qml.PCPhase(0.1, dim=be.dim, wires=be.wires), control=["sel", "r"], control_values=[0, 0]
        )
    ).get("CNOT", 0)
    d = max(deg_cos, deg_sin)
    return d * q + 2 * (d + 1) * proj


def trotter_cnot_cost(H, steps: int, order: int) -> int:
    return count_gates(lambda: qml.TrotterProduct(H, 1.0, n=steps, order=order)).get("CNOT", 0)


# =========================================================================== crossover analysis
def qsvt_truncation_error(tau: float, deg_cos: int, deg_sin: int, n_grid: int = 4001) -> float:
    r"""sup_{x∈[-1,1]} |e^{-iτx} - (cos_R(x) - i sin_R(x))|: a rigorous bound on the spectral-norm
    error of :func:`qsvt_time_evolution`, which implements these polynomials exactly."""
    from numpy.polynomial import chebyshev as C

    x = np.cos(np.linspace(0, np.pi, n_grid))
    approx = C.chebval(x, cos_poly(tau, deg_cos)) - 1j * C.chebval(x, sin_poly(tau, deg_sin))
    return float(np.abs(np.exp(-1j * tau * x) - approx).max())


def qsvt_min_cost(H, t: float, eps: float) -> tuple[int, int]:
    """Smallest Jacobi-Anger degree meeting ``eps``, and its CNOT cost. Returns (degree, cnots)."""
    lam = float(np.sum(np.abs(H.terms()[0])))
    tau = lam * t
    R = max(1, int(tau) - 5)
    while qsvt_truncation_error(tau, R + (R % 2), R + 1 - (R % 2)) > eps:
        R += 1
    dc, ds = R + (R % 2), R + 1 - (R % 2)
    return max(dc, ds), qsvt_cnot_cost(H, dc, ds)


def _pauli_exps(H, wires):
    """Return a function dt -> list of exp(-i c_j P_j dt) matrices, one per term."""
    coeffs, ops = H.terms()
    mats = [qml.matrix(o, wire_order=wires) for o in ops]
    I = np.eye(2 ** len(wires))

    def at(dt):
        return [np.cos(c * dt) * I - 1j * np.sin(c * dt) * P for c, P in zip(np.real(coeffs), mats)]

    return at


def suzuki_step(H, dt: float, order: int, wires=None) -> np.ndarray:
    """One Suzuki product-formula step for exp(-iH dt), order 1 or 2k (same recursion as qml.TrotterProduct)."""
    wires = list(H.wires) if wires is None else wires
    at = _pauli_exps(H, wires)

    def S(k, h):
        if k == 1:  # PennyLane applies the terms in reverse order for order 1
            U = np.eye(2 ** len(wires), dtype=complex)
            for E in at(h)[::-1]:
                U = E @ U
            return U
        if k == 2:
            half = at(h / 2)
            U = np.eye(2 ** len(wires), dtype=complex)
            for E in half:
                U = E @ U
            for E in half[::-1]:
                U = E @ U
            return U
        p = 1 / (4 - 4 ** (1 / (k - 1)))
        A = S(k - 2, p * h)
        return A @ A @ S(k - 2, (1 - 4 * p) * h) @ A @ A

    return S(order, dt)


def trotter_min_steps(H, t: float, eps: float, order: int, max_steps: int = 2**20) -> int:
    """Smallest number of Trotter steps with spectral-norm error <= eps (doubling + bisection)."""
    from numpy.linalg import matrix_power

    wires = list(H.wires)
    exact = expm(-1j * qml.matrix(H, wire_order=wires) * t)
    err = lambda r: np.linalg.norm(matrix_power(suzuki_step(H, t / r, order, wires), r) - exact, 2)
    hi = 1
    while err(hi) > eps:
        hi *= 2
        if hi > max_steps:
            return -1
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if err(mid) <= eps else (mid, hi)
    return hi


def trotter_min_cost(H, t: float, eps: float, orders=(2, 4)) -> tuple[int, int, int]:
    """Cheapest Trotter order for ``eps``. Returns (order, steps, cnots)."""
    best = None
    per_step = {o: trotter_cnot_cost(H, 1, o) for o in orders}
    for o in orders:
        r = trotter_min_steps(H, t, eps, o)
        if r > 0 and (best is None or r * per_step[o] < best[2]):
            best = (o, r, r * per_step[o])
    return best
