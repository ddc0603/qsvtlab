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
