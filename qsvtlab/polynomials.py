"""Chebyshev-basis polynomial approximations used by the QSVT applications."""

from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.special import jv
from scipy.stats import binom

__all__ = [
    "inverse_poly",
    "inverse_poly_lstsq",
    "cos_poly",
    "sin_poly",
    "jacobi_anger_degree",
    "sup_norm",
    "rescale",
]

_GRID = np.cos(np.linspace(0, np.pi, 40001))


def sup_norm(c: np.ndarray) -> float:
    """max |f(x)| on [-1, 1] (dense Chebyshev-clustered grid)."""
    return float(np.abs(C.chebval(_GRID, c)).max())


def rescale(c: np.ndarray, target_max: float = 0.5) -> tuple[np.ndarray, float]:
    """Scale coefficients so that max|f| = target_max. Returns (scaled coeffs, scale)."""
    s = target_max / sup_norm(c)
    return c * s, s


def inverse_poly(kappa: float, eps: float) -> np.ndarray:
    r"""Odd polynomial approximating :math:`1/x` on :math:`[-1,-1/\kappa]\cup[1/\kappa,1]`.

    This is the construction of Childs, Kothari & Somma (SIAM J. Comput. 2017). The function
    :math:`(1-(1-x^2)^b)/x` with :math:`b = \lceil\kappa^2\log(\kappa/\epsilon)\rceil` is
    :math:`\epsilon`-close to :math:`1/x` on the domain. Its Chebyshev expansion is truncated at
    degree :math:`2D+1`, with :math:`D = \lceil\sqrt{b\log(4b/\epsilon)}\rceil`. The degree is
    :math:`O(\kappa\log(\kappa/\epsilon))`. The coefficients are evaluated stably as binomial
    tail probabilities.
    """
    b = int(np.ceil(kappa**2 * np.log(kappa / eps)))
    D = int(np.ceil(np.sqrt(b * np.log(4 * b / eps))))
    c = np.zeros(2 * D + 2)
    j = np.arange(D + 1)
    c[2 * j + 1] = 4 * (-1.0) ** j * binom.sf(b + j, 2 * b, 0.5)
    return c


def inverse_poly_lstsq(kappa: float, degree: int, n_pts: int = 8000) -> np.ndarray:
    r"""Odd least-squares Chebyshev fit of :math:`1/(\kappa x)` on :math:`|x| \ge 1/\kappa`.

    The fit is constrained to stay bounded inside the gap. It reaches a given accuracy at a
    lower degree than :func:`inverse_poly`, but comes with no a-priori error bound.
    """
    if degree % 2 == 0:
        degree += 1
    theta = np.pi * (np.arange(n_pts) + 0.5) / n_pts
    x = np.cos(theta)
    x = x[np.abs(x) >= 1.0 / kappa]
    ks = np.arange(1, degree + 1, 2)
    V = C.chebvander(x, degree)[:, ks]
    coef, *_ = np.linalg.lstsq(V, 1.0 / (kappa * x), rcond=None)
    c = np.zeros(degree + 1)
    c[ks] = coef
    return c


def jacobi_anger_degree(tau: float, eps: float) -> int:
    """Smallest truncation degree R with sum_{k>R} 2|J_k(tau)| < eps."""
    R = int(np.ceil(abs(tau))) + 1
    while 2 * np.sum(np.abs(jv(np.arange(R + 1, R + 200), tau))) > eps:
        R += 1
    return R


def cos_poly(tau: float, degree: int) -> np.ndarray:
    r"""Even Jacobi-Anger expansion :math:`\cos(\tau x) = J_0(\tau) + 2\sum_k (-1)^k J_{2k}(\tau) T_{2k}(x)`."""
    degree -= degree % 2
    c = np.zeros(degree + 1)
    c[0] = jv(0, tau)
    k = np.arange(1, degree // 2 + 1)
    c[2 * k] = 2 * (-1.0) ** k * jv(2 * k, tau)
    return c


def sin_poly(tau: float, degree: int) -> np.ndarray:
    r"""Odd Jacobi-Anger expansion :math:`\sin(\tau x) = 2\sum_k (-1)^k J_{2k+1}(\tau) T_{2k+1}(x)`."""
    if degree % 2 == 0:
        degree += 1
    c = np.zeros(degree + 1)
    k = np.arange(0, (degree - 1) // 2 + 1)
    c[2 * k + 1] = 2 * (-1.0) ** k * jv(2 * k + 1, tau)
    return c
