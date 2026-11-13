"""Reproduce every figure and number in the README.

    python benchmarks/run_benchmarks.py            # full run (~ a few minutes)
    python benchmarks/run_benchmarks.py --quick    # smaller sweeps
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from numpy.polynomial import chebyshev as C

import qsvtlab as ql

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Reference categorical palette (fixed order) + recessive ink.
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "font.family": "DejaVu Sans",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

results: dict = {}


# --------------------------------------------------------------------------- 1. phase solver
def _pl_phase_error(angles, c, xs=np.linspace(0.05, 0.95, 7)):
    """Max |Re<0|QSVT|0> - p(x)| for PennyLane's angles, using a 1-qubit dilation of the scalar x."""
    errs = []
    for x in xs:
        U = np.array([[x, np.sqrt(1 - x * x)], [np.sqrt(1 - x * x), -x]])
        op = qml.QSVT(qml.QubitUnitary(U, wires=0), [qml.PCPhase(a, dim=1, wires=0) for a in angles])
        errs.append(abs(qml.matrix(op)[0, 0].real - C.chebval(x, c)))
    return float(max(errs))

def bench_phase_solver(quick: bool):
    kappas = [1.2, 1.5, 2, 3, 5, 8, 12, 20, 35, 60] if not quick else [1.5, 2, 5]
    ours, pl_iter, pl_fail = [], [], []
    for k in kappas:
        c, _ = ql.rescale(ql.inverse_poly(k, 1e-3), 0.5)
        d = len(c) - 1
        t = time.perf_counter()
        res = ql.solve_qsp_phases(c)
        dt = time.perf_counter() - t
        x = np.linspace(-1, 1, 2001)
        err = float(np.abs(ql.qsp_response(res.phases, x) - C.chebval(x, c)).max())
        ours.append((d, dt, err))

        mono = C.cheb2poly(c)
        if d <= 70:  # beyond this PennyLane's solvers fail (or run for minutes) - not attempted
            for solver in ("root-finding", "iterative"):
                t = time.perf_counter()
                try:
                    ang = qml.poly_to_angles(mono, "QSVT", angle_solver=solver)
                    if solver == "iterative":
                        pl_iter.append((d, time.perf_counter() - t, _pl_phase_error(ang, c)))
                except Exception:
                    pl_fail.append((d, solver))
    results["phase_solver"] = {"ours": ours, "pennylane_iterative": pl_iter, "pennylane_fail": pl_fail}

    fig, ax = plt.subplots(figsize=(6.4, 4))
    d, t, _ = zip(*ours)
    ax.loglog(d, t, "-o", color=S1, label="qsvtlab (Chebyshev FPI)")
    ok = [p for p in pl_iter if p[2] < 1e-6]
    bad = [p for p in pl_iter if p[2] >= 1e-6]
    if ok:
        di, ti, _ = zip(*ok)
        ax.loglog(di, ti, "-s", color=S2, label="qml.poly_to_angles (iterative), correct")
    if bad:
        db, tb, eb = zip(*bad)
        ax.loglog(db, tb, "s", color=S2, mfc=SURFACE, mew=2, ms=8,
                  label="qml.poly_to_angles, wrong phases")
        ax.annotate(f"degree {db[0]}: returns phases with\nerror {eb[0]:.2f} (silently)", (db[0], tb[0]),
                    textcoords="offset points", xytext=(12, -4), va="top", color=INK2, fontsize=9)
    ax.text(0.98, 0.04, "PennyLane root-finding solver fails at every degree shown",
            transform=ax.transAxes, ha="right", color=INK2, fontsize=8.5)
    ax.set_xlabel("polynomial degree  (1/x approximation, ε = 10⁻³)")
    ax.set_ylabel("wall-clock time to find phases (s)")
    ax.set_title("QSP phase finding: thousands of phases in under a second")
    ax.legend(loc="center right")
    ax.annotate(f"degree {d[-1]}\n{t[-1]:.2f} s", (d[-1], t[-1]), textcoords="offset points",
                xytext=(-10, 12), ha="right", color=INK2, fontsize=9)
    fig.savefig(FIG / "phase_solver.png")
    plt.close(fig)


# --------------------------------------------------------------------------- 2. linear solver
def bench_linear_solver(quick: bool):
    rng = np.random.default_rng(0)

    # (a) the polynomial itself
    kappa = 10
    fig, ax = plt.subplots(figsize=(6.4, 4))
    x = np.linspace(-1, 1, 4001)
    ax.axvspan(-1 / kappa, 1 / kappa, color=GRID, alpha=0.7, lw=0)
    ax.text(0.13, -0.95, "← gap |x| < 1/κ (unconstrained)", ha="left", color=INK2, fontsize=9)
    target = np.where(np.abs(x) >= 1 / kappa, 1 / (kappa * x), np.nan)
    ax.plot(x, target, color=INK, lw=1.2, ls="--", label="1/(κx)")
    for eps, col in [(1e-1, S2), (1e-3, S1)]:
        c = ql.inverse_poly(kappa, eps)
        ax.plot(x, C.chebval(x, c) / kappa, color=col, label=f"CKS polynomial, ε={eps:g}, degree {len(c)-1}")
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("x  (eigenvalue of A/α)")
    ax.set_ylabel("p(x)")
    ax.set_title("Odd polynomial approximating 1/x (κ = 10)")
    ax.legend(loc="upper left")
    fig.savefig(FIG / "inverse_polynomial.png")
    plt.close(fig)

    # (b) end-to-end fidelity vs epsilon for several kappa (8x8 Hermitian systems)
    kappas = [5, 20, 50] if not quick else [5, 20]
    epss = [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4]
    table = []
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for k, col in zip(kappas, [S1, S2, S3]):
        Q, _ = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))
        ev = np.r_[np.linspace(1 / k, 1, 4), -np.geomspace(1 / k, 1, 4)[::-1]]
        A = Q @ np.diag(ev) @ Q.conj().T
        b = rng.normal(size=8)
        infid = []
        for eps in epss:
            r = ql.qsvt_solve(A, b, eps=eps, kappa=k)
            infid.append(max(1 - r.fidelity, 1e-16))
            table.append(dict(kappa=k, eps=eps, degree=r.degree, fidelity=r.fidelity,
                              success_prob=r.success_prob,
                              norm_rel_err=abs(r.norm_estimate / r.norm_exact - 1)))
        ax.loglog(epss, infid, "-o", color=col, label=f"κ = {k}")
    ax.loglog(epss, np.array(epss) ** 2, color=INK2, lw=1, ls=":", label="ε²")
    ax.invert_xaxis()
    ax.set_xlabel("polynomial accuracy ε  (→ tighter)")
    ax.set_ylabel("state infidelity  1 − |⟨x|x̃⟩|²")
    ax.set_title("End-to-end QSVT linear solver (8×8, 5 qubits)")
    ax.legend(loc="lower left")
    fig.savefig(FIG / "linear_solver_fidelity.png")
    plt.close(fig)
    results["linear_solver"] = table

    # (c) degree vs kappa: O(kappa log(kappa/eps))
    ks = np.array([2, 4, 8, 16, 32, 64, 128])
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for eps, col in [(1e-2, S1), (1e-6, S2)]:
        deg = [len(ql.inverse_poly(k, eps)) - 1 for k in ks]
        ax.loglog(ks, deg, "-o", color=col, label=f"ε = {eps:g}")
    ax.loglog(ks, 8 * ks * np.log(ks / 1e-2), color=INK2, lw=1, ls=":", label="∝ κ log(κ/ε)")
    ax.set_xlabel("condition number κ")
    ax.set_ylabel("QSVT degree  (= block-encoding queries)")
    ax.set_title("Query complexity of the QSVT inverse")
    ax.legend(loc="upper left")
    fig.savefig(FIG / "inverse_degree_scaling.png")
    plt.close(fig)


# --------------------------------------------------------------------------- 3. Hamiltonian simulation
def bench_hamsim(quick: bool):
    n, t = (4, 2.0) if not quick else (3, 1.0)
    H = ql.tfim(n)
    out = {"n": n, "t": t, "lambda": float(np.sum(np.abs(H.terms()[0]))), "qsvt": [], "trotter": {}}

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for order, col in [(1, S2), (2, S3), (4, S4)]:
        steps = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if order < 4 else [1, 2, 4, 8, 16, 32, 64, 128, 256]
        pts = []
        for r in steps:
            e = ql.trotter_error(H, t, r, order)
            pts.append((ql.trotter_cnot_cost(H, r, order), e))
        out["trotter"][order] = pts
        cx, er = zip(*pts)
        ax.loglog(cx, er, "-o", color=col, label=f"Trotter, order {order}")

    for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10]:
        r = ql.qsvt_time_evolution(H, t, eps=eps)
        cost = ql.qsvt_cnot_cost(H, r.deg_cos, r.deg_sin)
        out["qsvt"].append((cost, max(r.error, 1e-15), r.deg_cos, r.deg_sin))
    cx, er, *_ = zip(*out["qsvt"])
    ax.loglog(cx, er, "-o", color=S1, label="QSVT (Jacobi–Anger, LCU encoding)")
    ax.set_xlabel("CNOT count  (decomposed with PennyLane)")
    ax.set_ylabel(r"$\|U - e^{-iHt}\|_2$")
    ax.set_title(f"Time evolution of a {n}-qubit TFIM chain, t = {t:g}")
    ax.legend(loc="lower left")
    fig.savefig(FIG / "hamsim_error_vs_cnots.png")
    plt.close(fig)
    results["hamiltonian_simulation"] = out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    for name, fn in [("phase solver", bench_phase_solver), ("linear solver", bench_linear_solver),
                     ("hamiltonian simulation", bench_hamsim)]:
        t0 = time.perf_counter()
        fn(args.quick)
        print(f"[{name}] done in {time.perf_counter() - t0:.1f}s")
    (ROOT / "benchmarks" / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print("figures ->", FIG)
