r"""Ground-state energy estimation and preparation with QSVT eigenstate filtering.

Algorithm
---------
Let :math:`|\phi_0\rangle` be a cheap reference state (Hartree-Fock) with overlaps
:math:`\gamma_k = \langle E_k|\phi_0\rangle`. The Lin-Tong filter

.. math::
    R_\ell(x;\delta) = \frac{T_\ell\!\left(-1 + 2\frac{x^2-\delta^2}{1-\delta^2}\right)}
                            {T_\ell\!\left(-1 - \frac{2\delta^2}{1-\delta^2}\right)}

is an even polynomial of degree :math:`2\ell`. It equals 1 at :math:`x = 0` and is at most
:math:`2e^{-\sqrt2\,\ell\delta}` for :math:`|x| \ge \delta` (Lin & Tong, Quantum 4, 361 (2020)).
QSVT applied to the block encoding of :math:`(H - \mu)/\alpha` succeeds with probability

.. math:: p(\mu) = s^2 \sum_k |\gamma_k|^2\, R_\ell\big((E_k-\mu)/\alpha\big)^2 .

1. **Spectroscopy.** Scanning :math:`\mu` turns :math:`p(\mu)` into a spectrum. It has a
   peak of height :math:`s^2|\gamma_k|^2` at every eigenvalue the reference state overlaps.
   No prior knowledge of :math:`E_0` is needed.
2. **Refinement.** Each isolated peak is symmetric about :math:`E_k`, so maximizing
   :math:`p(\mu)` locates :math:`E_0` far more precisely than the filter width.
3. **Preparation.** Filtering at :math:`\mu = \hat E_0` with a larger :math:`\ell` projects
   onto the ground state. Infidelity decays exponentially in :math:`\ell`.

Every :math:`p(\mu)` is computed by simulating the QSVT circuit in PennyLane and reading
the ancilla probabilities. There is no classical shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
from numpy.polynomial import chebyshev as C
from scipy.optimize import minimize_scalar

from .block_encodings import unitary_dilation
from .circuits import qsvt_phases_for, real_qsvt

__all__ = [
    "molecule_hamiltonian",
    "filter_poly",
    "EigenFilter",
    "SpectroscopyResult",
    "GroundStateResult",
]

ANGSTROM_TO_BOHR = 1 / 0.529177210903


def molecule_hamiltonian(symbols, coords_angstrom, charge=0):
    """Qubit Hamiltonian (Jordan-Wigner, STO-3G) and HF bitstring via ``qml.qchem``."""
    coords = np.asarray(coords_angstrom, dtype=float).reshape(-1) * ANGSTROM_TO_BOHR
    mol = qml.qchem.Molecule(list(symbols), coords, charge=charge)
    H, n_qubits = qml.qchem.molecular_hamiltonian(mol)
    n_elec = mol.n_electrons
    return H, n_qubits, qml.qchem.hf_state(n_elec, n_qubits)


def filter_poly(ell: int, delta: float, s: float = 0.9) -> np.ndarray:
    """Chebyshev coefficients of s * R_ell(x; delta), an even polynomial of degree 2*ell."""
    Tl = lambda y: C.chebval(y, np.r_[np.zeros(ell), 1.0])
    norm = Tl(-1 - 2 * delta**2 / (1 - delta**2))
    f = lambda x: s * Tl(-1 + 2 * (x**2 - delta**2) / (1 - delta**2)) / norm
    c = C.chebinterpolate(f, 2 * ell)
    c[1::2] = 0.0
    return c


@dataclass
class SpectroscopyResult:
    mu: np.ndarray
    p_success: np.ndarray
    peaks: np.ndarray  # refined peak energies (ascending)
    weights: np.ndarray  # estimated |gamma_k|^2 at each peak
    n_circuits: int


@dataclass
class GroundStateResult:
    energy_estimate: float
    energy_exact: float
    state: np.ndarray
    fidelity: float  # |<E0|psi>|^2
    energy_of_state: float  # <psi|H|psi>
    success_prob: float
    hf_overlap: float
    degree: int


class EigenFilter:
    """QSVT eigenstate filter for a Hamiltonian ``H`` with reference state ``ref_bits``."""

    def __init__(self, H, ref_bits, alpha: float | None = None):
        self.H = H
        self.wires = list(range(len(ref_bits)))
        self.Hmat = qml.matrix(H, wire_order=self.wires)
        self.ref_bits = np.asarray(ref_bits)
        coeffs, ops = H.terms()
        ident = [c for c, o in zip(coeffs, ops) if len(o.wires) == 0 or o.name == "Identity"]
        self.c_id = float(np.real(np.sum(ident))) if ident else 0.0
        self.lam = float(np.sum(np.abs(coeffs))) - abs(self.c_id)  # 1-norm without identity
        # |E - mu| <= 2*lam for every mu in [c_id - lam, c_id + lam]  ->  safe normalization.
        self.alpha = alpha if alpha is not None else 2 * self.lam
        self.mu_range = (self.c_id - self.lam, self.c_id + self.lam)
        self.n_circuits = 0
        self._phase_cache: dict = {}

    # ---------------------------------------------------------------- circuits
    def _phases(self, ell, delta, s):
        key = (ell, round(delta, 12), s)
        if key not in self._phase_cache:
            c = filter_poly(ell, delta, s)
            self._phase_cache[key] = (c, qsvt_phases_for(c))
        return self._phase_cache[key]

    def _run(self, mu, ell, delta, s=0.9):
        _, phases = self._phases(ell, delta, s)
        shifted = (self.Hmat - mu * np.eye(len(self.Hmat))) / self.alpha
        be = unitary_dilation(shifted, self.wires, ancilla_wire="a0", alpha=1.0)
        dev = qml.device("default.qubit", wires=["r"] + be.wires)

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(self.ref_bits, wires=self.wires)
            real_qsvt(be, phases, "r")
            return qml.state()

        self.n_circuits += 1
        psi = np.asarray(circuit())
        return psi[: len(self.Hmat)]  # r = 0, a0 = 0 branch

    def success_probability(self, mu, ell, delta, s=0.9) -> float:
        post = self._run(mu, ell, delta, s)
        return float(np.vdot(post, post).real)

    # ---------------------------------------------------------------- algorithm
    @staticmethod
    def sidelobe_bound(ell, delta) -> float:
        """Upper bound on R_ell(x)^2 for |x| >= delta (Lin & Tong, Lemma 2)."""
        return float(min(1.0, 4 * np.exp(-2 * np.sqrt(2) * ell * delta)))

    def spectroscopy(self, ell=100, delta=0.03, n_mu=241, threshold=None, s=0.9,
                     window=None) -> SpectroscopyResult:
        """Scan mu over the safe range and locate peaks of p(mu).

        ``threshold`` is a minimum peak weight |gamma_k|^2. By default it is 3x the sidelobe
        bound, so filter ripples can never be mistaken for eigenvalues.
        """
        if threshold is None:
            threshold = max(1e-3, 3 * self.sidelobe_bound(ell, delta))
        lo, hi = window if window is not None else self.mu_range
        mu = np.linspace(max(lo, self.mu_range[0]), min(hi, self.mu_range[1]), n_mu)
        p = np.array([self.success_probability(m, ell, delta, s) for m in mu])
        # local maxima above threshold, then refine each by bounded 1-D maximization
        idx = [i for i in range(1, n_mu - 1) if p[i] >= p[i - 1] and p[i] > p[i + 1] and p[i] > threshold * s**2]
        if p[0] > p[1] and p[0] > threshold * s**2:
            idx = [0] + idx
        peaks, weights = [], []
        step = mu[1] - mu[0]
        for i in idx:
            r = minimize_scalar(lambda m: -self.success_probability(m, ell, delta, s),
                                bounds=(mu[i] - step, mu[i] + step), method="bounded",
                                options={"xatol": 1e-9})
            peaks.append(r.x)
            weights.append(-r.fun / s**2)
        return SpectroscopyResult(mu, p, np.array(peaks), np.array(weights), self.n_circuits)

    def estimate_ground_energy(self, coarse=(60, 0.05, 121), fine=(300, 0.01, 41), half_width=None):
        """Coarse scan of the whole spectrum, then a high-resolution zoom on the lowest peak.

        Returns (E0_estimate, coarse SpectroscopyResult, fine SpectroscopyResult).
        """
        ell_c, d_c, n_c = coarse
        sc = self.spectroscopy(ell=ell_c, delta=d_c, n_mu=n_c)
        if len(sc.peaks) == 0:
            raise RuntimeError("no peak found: reference state has negligible overlap")
        hw = half_width if half_width is not None else 2 * d_c * self.alpha
        ell_f, d_f, n_f = fine
        sf = self.spectroscopy(ell=ell_f, delta=d_f, n_mu=n_f,
                               window=(sc.peaks[0] - hw, sc.peaks[0] + hw))
        return float(sf.peaks[0]), sc, sf

    def prepare_ground_state(self, mu, ell=150, delta=0.03, s=0.9) -> GroundStateResult:
        post = self._run(mu, ell, delta, s)
        p = float(np.vdot(post, post).real)
        psi = post / np.sqrt(p)
        ev, V = np.linalg.eigh(self.Hmat)
        ref_idx = int("".join(map(str, self.ref_bits)), 2)
        return GroundStateResult(
            energy_estimate=float(mu),
            energy_exact=float(ev[0]),
            state=psi,
            fidelity=float(np.abs(np.vdot(V[:, 0], psi)) ** 2),
            energy_of_state=float(np.vdot(psi, self.Hmat @ psi).real),
            success_prob=p,
            hf_overlap=float(np.abs(V[ref_idx, 0]) ** 2),
            degree=2 * ell,
        )

    def p_success_exact(self, mu, ell, delta, s=0.9) -> float:
        """Closed-form p(mu) from the spectrum. Used only in tests to validate the circuits."""
        ev, V = np.linalg.eigh(self.Hmat)
        ref = np.zeros(len(ev))
        ref[int("".join(map(str, self.ref_bits)), 2)] = 1
        g2 = np.abs(V.conj().T @ ref) ** 2
        c, _ = self._phases(ell, delta, s)
        return float(np.sum(g2 * C.chebval((ev - mu) / self.alpha, c) ** 2))
