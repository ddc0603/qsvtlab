r"""QSVT circuits that implement a *real* polynomial exactly.

A QSVT sequence with projector phases :math:`\vec\varphi` puts :math:`P(A) + iQ(A)` into the
top-left block, where :math:`P` is the polynomial we want and :math:`Q` is junk. Negating every
phase conjugates the polynomial's coefficients. A linear combination of the two sequences
therefore isolates the real part:

.. math::
    \tfrac12\big(U_{\vec\varphi} + U_{-\vec\varphi}\big) \;\to\; \mathrm{Re}\,[P + iQ] = f(A).

This costs a single extra ancilla qubit (``lcu_wire``). It is put in :math:`|+\rangle`, each
projector phase is controlled on it with sign :math:`\pm`, and it is post-selected on
:math:`|+\rangle`.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from .angles import solve_qsp_phases, wx_to_reflection
from .block_encodings import BlockEncoding

__all__ = ["qsvt_phases_for", "real_qsvt", "qsvt_block"]


def qsvt_phases_for(cheb_coeffs: np.ndarray) -> np.ndarray:
    """Chebyshev coefficients -> projector phases for :func:`real_qsvt`."""
    return wx_to_reflection(solve_qsp_phases(cheb_coeffs).phases)


def _signed_projector(phi, be: BlockEncoding, lcu_wire):
    r"""exp(i phi Z_lcu (2Π - 1)): +phi when lcu=|0>, -phi when lcu=|1>.

    The operator is diagonal, so it is queued as one ``DiagonalQubitUnitary``. In gates it is
    a pair of oppositely-controlled ``PCPhase`` ops; see ``hamiltonian_sim.qsvt_cnot_cost``.
    Simulating it as a diagonal is about 2x faster than two dense controlled matrices.
    """
    n_total = 2 ** len(be.wires)
    proj = np.where(np.arange(n_total) < be.dim, 1.0, -1.0)  # 2Π - 1 on the encoding register
    diag = np.exp(1j * phi * np.concatenate([proj, -proj]))  # lcu wire is most significant
    qml.DiagonalQubitUnitary(diag, wires=[lcu_wire] + be.wires)


def real_qsvt(be: BlockEncoding, phases: np.ndarray, lcu_wire, real_part: bool = True):
    r"""Queue a QSVT circuit whose post-selected block is f(A/alpha).

    Post-select ``be.ancilla_wires`` on :math:`|0\rangle`, and ``lcu_wire`` on
    :math:`|0\rangle` after the final Hadamard.

    Matrix-product order is :math:`e^{i\varphi_0}\,U\,e^{i\varphi_1}\,U^\dagger\cdots`, so the
    gates are applied in reverse.
    """
    d = len(phases) - 1
    if real_part:
        qml.Hadamard(lcu_wire)
    for k in range(d, -1, -1):
        if real_part:
            _signed_projector(phases[k], be, lcu_wire)
        else:
            qml.PCPhase(phases[k], dim=be.dim, wires=be.wires)
        if k > 0:
            # alternate U, U^dagger (position k from the left: odd -> U, even -> U^dagger)
            if (k - 1) % 2 == 0:
                be.make_op()
            else:
                qml.adjoint(be.make_op)()
    if real_part:
        qml.Hadamard(lcu_wire)


def qsvt_block(be: BlockEncoding, phases: np.ndarray, lcu_wire, real_part: bool = True):
    """Return the post-selected system block of :func:`real_qsvt` (for verification)."""
    wires = ([lcu_wire] if real_part else []) + be.wires
    U = qml.matrix(lambda: real_qsvt(be, phases, lcu_wire, real_part), wire_order=wires)()
    n = be.dim
    return U[:n, :n]
