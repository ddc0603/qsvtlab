r"""Chebyshev-native QSP phase-factor solver.

Background
----------
In the "W_x" convention of quantum signal processing (QSP),

.. math::

    U_\Phi(x) = e^{i\phi_0 Z} \prod_{j=1}^{d} W(x) e^{i\phi_j Z},
    \qquad W(x) = \begin{pmatrix} x & i\sqrt{1-x^2} \\ i\sqrt{1-x^2} & x \end{pmatrix}.

For *symmetric* phase factors, :math:`g(x) = \mathrm{Im}\,\langle 0|U_\Phi(x)|0\rangle` can
represent any real polynomial :math:`f` of degree :math:`d` and definite parity with
:math:`\|f\|_{\infty,[-1,1]} < 1`.

PennyLane's built-in :func:`pennylane.poly_to_angles` takes the target polynomial in the
*monomial* basis. That basis is exponentially ill-conditioned, so in practice it fails for
degrees above roughly 50. Useful QSVT applications (for example matrix inversion with
:math:`\kappa = 50`) need degrees in the thousands.

This module works entirely in the Chebyshev basis and implements

* the fixed-point iteration (FPI) of Dong, Lin, Ni & Wang, "Infinite quantum signal
  processing", arXiv:2209.10162. It converges linearly when :math:`\|c\|_1 \lesssim 0.86`.
* a Newton fallback (finite-difference Jacobian) for targets outside that regime.

Each iteration costs :math:`O(d^2)` and is fully vectorized over the quadrature nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "QSPResult",
    "qsp_u00",
    "qsp_response",
    "solve_qsp_phases",
    "wx_to_reflection",
]


@dataclass
class QSPResult:
    phases: np.ndarray  # full symmetric W_x phases, length d+1
    degree: int
    iterations: int
    residual: float  # max |Chebyshev coefficient error|
    method: str  # "fpi" or "newton"


def qsp_u00(phases: np.ndarray, x: np.ndarray) -> np.ndarray:
    """<0|U_Phi(x)|0> in the W_x convention, vectorized over x."""
    x = np.asarray(x, dtype=float)
    s = np.sqrt(np.clip(1.0 - x**2, 0.0, None))
    # Apply U_Phi to |0> right-to-left, tracking the two amplitudes (a, b).
    a = np.exp(1j * phases[-1]) * np.ones_like(x, dtype=complex)
    b = np.zeros_like(a)
    for ph in phases[-2::-1]:
        a, b = x * a + 1j * s * b, 1j * s * a + x * b
        a, b = np.exp(1j * ph) * a, np.exp(-1j * ph) * b
    return a


def qsp_response(phases: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The real polynomial implemented by symmetric phases: Im <0|U_Phi(x)|0>."""
    return np.imag(qsp_u00(phases, x))


def _full_phases(reduced: np.ndarray, d: int) -> np.ndarray:
    rr = reduced[::-1]
    full = np.concatenate([rr, reduced]) if d % 2 == 1 else np.concatenate([rr, reduced[1:]])
    full = full.copy()
    full[0] += np.pi / 4
    full[-1] += np.pi / 4
    return full


def solve_qsp_phases(
    cheb_coeffs: np.ndarray,
    tol: float = 1e-12,
    max_fpi_iter: int = 200,
    max_newton_iter: int = 50,
) -> QSPResult:
    r"""Find symmetric W_x phases with :math:`\mathrm{Im}\,\langle0|U_\Phi(x)|0\rangle = f(x)`.

    Args:
        cheb_coeffs: Chebyshev coefficients ``c_k`` of ``f(x) = sum_k c_k T_k(x)``. ``f`` must
            have definite parity (coefficients of the wrong parity are ignored) and
            ``max |f| < 1`` on ``[-1, 1]``.
        tol: convergence tolerance on the Chebyshev coefficients.

    Returns:
        QSPResult with the full phase vector of length ``d + 1``.
    """
    c = np.asarray(cheb_coeffs, dtype=float)
    c = np.trim_zeros(c, "b") if np.any(c) else c[:1]
    d = len(c) - 1
    parity = d % 2
    c_red = c[parity::2]
    n_red = len(c_red)

    # Positive-half Chebyshev nodes; projection onto T_{2k+parity}.
    n_nodes = 2 * n_red
    theta = np.pi * (np.arange(n_nodes) + 0.5) / (2 * n_nodes)
    nodes = np.cos(theta)
    vander = np.cos(np.outer(theta, np.arange(parity, d + 1, 2)))
    proj = np.linalg.pinv(vander)

    def residual(r):
        return proj @ qsp_response(_full_phases(r, d), nodes) - c_red

    # 1) Fixed-point iteration.
    r = np.zeros(n_red)
    it = 0
    err = np.inf
    for it in range(max_fpi_iter):
        F = residual(r)
        err = np.abs(F).max()
        if err < tol:
            return QSPResult(_full_phases(r, d), d, it, err, "fpi")
        if not np.isfinite(err) or (it > 20 and err > 0.5):
            break
        r = r - 0.5 * F

    # 2) Newton fallback.
    if not np.isfinite(err) or err > 0.5:
        r = np.zeros(n_red)
    h = 1e-7
    for k in range(max_newton_iter):
        F = residual(r)
        err = np.abs(F).max()
        if err < tol:
            return QSPResult(_full_phases(r, d), d, it + k, err, "newton")
        J = np.empty((n_red, n_red))
        for j in range(n_red):
            e = np.zeros(n_red)
            e[j] = h
            J[:, j] = (residual(r + e) - F - c_red * 0) / h
        r = r - np.linalg.solve(J, F)

    raise RuntimeError(
        f"QSP phase solver did not converge (degree {d}, residual {err:.2e}). "
        "Try rescaling the target so that max|f| <= 0.5."
    )


def wx_to_reflection(phases: np.ndarray) -> np.ndarray:
    r"""Convert symmetric W_x phases into projector phases for a QSVT circuit.

    For a Hermitian block encoding :math:`U` (a reflection in each Jordan block),

    .. math:: W(x) = i\, e^{-i\pi Z/4} R(x)\, e^{-i\pi Z/4},

    so the W_x sequence equals :math:`i^d` times a reflection-convention sequence with phases
    :math:`(\phi_0 - \pi/4, \phi_1 - \pi/2, \dots, \phi_{d-1} - \pi/2, \phi_d - \pi/4)`.
    The global factor, plus an extra :math:`-i` that turns ``Im`` into ``Re``, is absorbed
    into the first phase. This works because :math:`\Pi e^{i\delta(2\Pi-1)} = e^{i\delta}\Pi`.
    The result is a phase list whose QSVT block has real part exactly :math:`f(A)`.
    """
    phases = np.asarray(phases, dtype=float)
    d = len(phases) - 1
    out = phases - np.pi / 2
    out[0] += np.pi / 4
    out[-1] += np.pi / 4
    out[0] += (d - 1) * np.pi / 2
    return out
