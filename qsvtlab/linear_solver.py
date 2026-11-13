r"""End-to-end QSVT quantum linear-system solver.

Given :math:`A x = b`, prepare :math:`|x\rangle \propto A^{-1}|b\rangle` by

1. block-encoding :math:`A/\alpha` (Hermitian, or embedded as
   :math:`\begin{psmallmatrix}0&A\\A^\dagger&0\end{psmallmatrix}` otherwise),
2. applying an odd polynomial :math:`p(x)\approx \frac{s}{\kappa x}` with QSVT, and
3. post-selecting the block-encoding ancilla and the real-part ancilla on :math:`|0\rangle`.

The whole circuit runs in PennyLane. The post-selected state is compared against the exact
classical solution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
from numpy.polynomial import chebyshev as C

from .block_encodings import lcu_block_encoding, unitary_dilation
from .circuits import qsvt_phases_for, real_qsvt
from .polynomials import inverse_poly, inverse_poly_lstsq, rescale

__all__ = ["LinearSolveResult", "hermitian_embedding", "qsvt_solve"]


@dataclass
class LinearSolveResult:
    state: np.ndarray  # normalized post-selected solution state (system register)
    exact: np.ndarray  # normalized classical solution
    fidelity: float  # |<x_exact|x_qsvt>|^2
    success_prob: float  # probability of the post-selection succeeding
    degree: int  # QSVT degree = number of block-encoding queries
    kappa: float
    alpha: float
    n_qubits: int
    poly_coeffs: np.ndarray
    norm_estimate: float  # ||A^{-1} b|| recovered from the success probability
    norm_exact: float


def hermitian_embedding(A: np.ndarray, b: np.ndarray):
    """Embed general A into a Hermitian matrix of power-of-two size; returns (H, b', slice)."""
    A = np.asarray(A, dtype=complex)
    m = A.shape[0]
    if np.allclose(A, A.conj().T):
        H, bb, sl = A, np.asarray(b, dtype=complex), slice(0, m)
    else:
        H = np.block([[np.zeros_like(A), A], [A.conj().T, np.zeros_like(A)]])
        bb = np.concatenate([b, np.zeros(m)]).astype(complex)
        sl = slice(m, 2 * m)
    N = 1 << int(np.ceil(np.log2(len(H))))
    if N != len(H):  # pad with identity block, which keeps the condition number benign
        Hp = np.eye(N, dtype=complex) * np.linalg.norm(H, 2)
        Hp[: len(H), : len(H)] = H
        bp = np.zeros(N, dtype=complex)
        bp[: len(bb)] = bb
        H, bb = Hp, bp
    return H, bb, sl


def qsvt_solve(
    A,
    b: np.ndarray,
    eps: float = 1e-2,
    kappa: float | None = None,
    method: str = "ckc",
    degree: int | None = None,
    poly_max: float = 0.5,
    device: str = "default.qubit",
) -> LinearSolveResult:
    r"""Solve :math:`Ax=b` with QSVT and return the post-selected quantum state.

    Args:
        A: matrix (``np.ndarray``) or Pauli-sum Hamiltonian (``qml.operation.Operator``).
           Operators use the gate-level PREPARE-SELECT-PREPARE block encoding.
        b: right-hand side.
        eps: target polynomial accuracy (``method="ckc"``).
        kappa: condition-number bound for the *normalized* matrix. Computed exactly if None.
        method: ``"ckc"`` (Childs-Kothari-Somma, provable bound) or ``"lstsq"``
            (least-squares fit at the given ``degree``).
        poly_max: the polynomial is rescaled so that ``max|p| = poly_max`` (must be < 1).
    """
    b = np.asarray(b, dtype=complex)

    if isinstance(A, qml.operation.Operator):
        sys_wires = list(A.wires)
        n_terms = len(A.terms()[0])
        anc = [f"a{i}" for i in range(int(np.ceil(np.log2(n_terms))))]
        be = lcu_block_encoding(A, anc)
        Amat = qml.matrix(A, wire_order=sys_wires)
        sl = slice(0, len(Amat))
        bb = b
    else:
        Amat, bb, sl = hermitian_embedding(A, b)
        n = int(np.log2(len(Amat)))
        sys_wires = list(range(n))
        be = unitary_dilation(Amat, sys_wires, ancilla_wire="a0")

    evals = np.linalg.eigvalsh(Amat / be.alpha)
    if kappa is None:
        kappa = 1.0 / np.abs(evals).min()

    if method == "ckc":
        c = inverse_poly(kappa, eps)
    elif method == "lstsq":
        c = inverse_poly_lstsq(kappa, degree or int(4 * kappa))
    else:
        raise ValueError(method)
    c, scale = rescale(c, poly_max)
    phases = qsvt_phases_for(c)

    lcu_wire = "r"
    all_wires = [lcu_wire] + be.wires
    dev = qml.device(device, wires=all_wires)
    b_norm = bb / np.linalg.norm(bb)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(b_norm, wires=sys_wires)
        real_qsvt(be, phases, lcu_wire)
        return qml.state()

    psi = np.asarray(circuit())
    N = len(b_norm)
    post = psi[:N]  # lcu=0, ancillas=0 (they are the leading wires)
    p_succ = float(np.vdot(post, post).real)
    x_q = post / np.sqrt(p_succ)

    x_exact_full = np.linalg.solve(Amat, bb)
    x_exact = x_exact_full / np.linalg.norm(x_exact_full)
    fid = float(np.abs(np.vdot(x_exact, x_q)) ** 2)

    # p(A/alpha) ≈ scale * g * alpha * A^{-1}, with g = 1 (ckc approximates 1/x) or
    # g = 1/kappa (lstsq approximates 1/(kappa x)). Hence ||A^{-1} b|| ≈ sqrt(p_succ)·||b|| / (scale·g·alpha).
    g = 1.0 if method == "ckc" else 1.0 / kappa
    norm_est = float(np.sqrt(p_succ) * np.linalg.norm(bb) / (scale * g * be.alpha))

    return LinearSolveResult(
        state=x_q[sl] / np.linalg.norm(x_q[sl]) if sl.stop <= N else x_q,
        exact=x_exact[sl] / np.linalg.norm(x_exact[sl]),
        fidelity=fid,
        success_prob=p_succ,
        degree=len(c) - 1,
        kappa=float(kappa),
        alpha=be.alpha,
        n_qubits=len(all_wires),
        poly_coeffs=c,
        norm_estimate=norm_est,
        norm_exact=float(np.linalg.norm(x_exact_full)),
    )
