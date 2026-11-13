r"""Block encodings with explicit, controllable normalization.

.. note::
    ``qml.BlockEncode`` silently rescales :math:`A` by
    :math:`\max(\|AA^\dagger\|_\infty, \|A^\dagger A\|_\infty)` (an induced
    :math:`\infty`-norm, not the spectral norm) whenever that quantity exceeds 1.
    A matrix with :math:`\|A\|_2 = 1` can therefore end up encoded as roughly
    :math:`0.9\,A`, which silently shifts every eigenvalue that the QSVT polynomial acts on.
    The dilation below normalizes by the spectral norm and reports the factor, so there
    are no surprises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pennylane as qml

__all__ = ["BlockEncoding", "unitary_dilation", "lcu_block_encoding"]


@dataclass
class BlockEncoding:
    """A unitary U with (<0|_anc ⊗ I) U (|0>_anc ⊗ I) = A / alpha."""

    make_op: callable  # () -> Operator, so it can be queued inside a circuit
    alpha: float
    ancilla_wires: list
    system_wires: list
    hermitian: bool = True
    meta: dict = field(default_factory=dict)

    @property
    def wires(self):
        return list(self.ancilla_wires) + list(self.system_wires)

    @property
    def dim(self) -> int:
        """Dimension of the encoded block (the projector rank)."""
        return 2 ** len(self.system_wires)


def _psd_sqrt(M: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh((M + M.conj().T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.conj().T


def unitary_dilation(A: np.ndarray, system_wires, ancilla_wire, alpha: float | None = None):
    r"""Exact one-ancilla block encoding of a Hermitian matrix :math:`A`.

    .. math:: U = \begin{pmatrix} \tilde A & \sqrt{I-\tilde A^2} \\ \sqrt{I-\tilde A^2} & -\tilde A \end{pmatrix},
              \quad \tilde A = A/\alpha.

    For Hermitian :math:`A` the unitary :math:`U` is also Hermitian, so :math:`U^\dagger = U`.
    This is the textbook "oracle" model: ideal for studying the algorithm, but not
    compiled to gates. For a hardware-style encoding use :func:`lcu_block_encoding`.
    """
    A = np.asarray(A, dtype=complex)
    if not np.allclose(A, A.conj().T, atol=1e-10):
        raise ValueError("unitary_dilation expects a Hermitian matrix; embed non-Hermitian A first.")
    if alpha is None:
        alpha = float(np.linalg.norm(A, 2))
    At = A / alpha
    S = _psd_sqrt(np.eye(len(A)) - At @ At)
    U = np.block([[At, S], [S, -At]])
    U = np.real_if_close(U, tol=1e6)
    wires = [ancilla_wire] + list(system_wires)
    return BlockEncoding(
        make_op=lambda: qml.QubitUnitary(U, wires=wires),
        alpha=alpha,
        ancilla_wires=[ancilla_wire],
        system_wires=list(system_wires),
        meta={"type": "dilation", "matrix": U},
    )


def lcu_block_encoding(H: qml.operation.Operator, ancilla_wires):
    r"""PREPARE-SELECT-PREPARE block encoding of a Pauli-sum Hamiltonian.

    The normalization is :math:`\alpha = \lambda = \sum_j |c_j|`. This construction compiles
    to elementary gates, so it gives meaningful gate counts.
    """
    coeffs, ops = H.terms()
    n_anc = int(np.ceil(np.log2(len(coeffs))))
    if len(ancilla_wires) < n_anc:
        raise ValueError(f"Need {n_anc} ancilla wires for {len(coeffs)} terms.")
    anc = list(ancilla_wires)[:n_anc]
    lam = float(np.sum(np.abs(coeffs)))
    lcu = qml.dot(coeffs, ops)
    return BlockEncoding(
        make_op=lambda: qml.PrepSelPrep(lcu, control=anc),
        alpha=lam,
        ancilla_wires=anc,
        system_wires=list(H.wires),
        meta={"type": "lcu", "n_terms": len(coeffs)},
    )
