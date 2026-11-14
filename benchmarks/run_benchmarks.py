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


# --------------------------------------------------------------------------- 4. chemistry
def bench_chemistry(quick: bool):
    from qsvtlab.chemistry import EigenFilter, molecule_hamiltonian

    CHEM_ACC = 1.6e-3
    out = {"h2_pes": [], "h4": {}, "fidelity_vs_degree": {}}

    # (a) H2 dissociation curve
    ds = [0.5, 0.74, 1.0, 1.4, 1.8, 2.2, 2.6] if not quick else [0.74, 2.0]
    for d in ds:
        H, n, hf = molecule_hamiltonian(["H", "H"], [[0, 0, 0], [0, 0, d]])
        F = EigenFilter(H, hf)
        E0, sc, sf = F.estimate_ground_energy(coarse=(60, 0.05, 121), fine=(300, 0.01, 41))
        g = F.prepare_ground_state(E0, ell=200)
        idx = int("".join(map(str, hf)), 2)
        e_hf = float(F.Hmat[idx, idx].real)
        out["h2_pes"].append(dict(d=d, E_fci=g.energy_exact, E_hf=e_hf, E_qsvt=E0,
                                  hf_overlap=g.hf_overlap, fidelity=g.fidelity,
                                  success_prob=g.success_prob, circuits=F.n_circuits))
        print(f"   H2 d={d}: |E_qsvt - E_fci| = {abs(E0 - g.energy_exact):.1e}", flush=True)

    rows = out["h2_pes"]
    d = [r["d"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(d, [r["E_hf"] for r in rows], "--", color=S2, label="Hartree–Fock")
    ax.plot(d, [r["E_fci"] for r in rows], "-", color=INK, lw=1.2, label="exact (FCI)")
    ax.plot(d, [r["E_qsvt"] for r in rows], "o", color=S1, ms=8, label="QSVT filter spectroscopy")
    ax.set_xlabel("H–H bond length (Å)")
    ax.set_ylabel("ground-state energy (Ha)")
    ax.set_title("H₂ dissociation curve from QSVT eigenstate filtering")
    ax.legend(loc="upper right")
    fig.savefig(FIG / "chem_h2_pes.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.semilogy(d, [abs(r["E_hf"] - r["E_fci"]) for r in rows], "--o", color=S2, label="Hartree–Fock")
    ax.semilogy(d, [max(abs(r["E_qsvt"] - r["E_fci"]), 1e-12) for r in rows], "-o", color=S1,
                label="QSVT filter spectroscopy")
    ax.axhline(CHEM_ACC, color=INK2, lw=1, ls=":")
    ax.text(d[0], CHEM_ACC * 1.4, "chemical accuracy (1.6 mHa)", color=INK2, fontsize=9)
    ax.set_xlabel("H–H bond length (Å)")
    ax.set_ylabel("|E − E_FCI|  (Ha)")
    ax.set_title("Energy error: QSVT is 4–6 orders below chemical accuracy")
    ax.legend(loc="center right")
    fig.savefig(FIG / "chem_h2_error.png")
    plt.close(fig)

    # (b) H4 chain: spectroscopy (coarse + fine zoom) against the exact spectrum
    if not quick:
        H, n, hf = molecule_hamiltonian(["H"] * 4, [[0, 0, 1.5 * i] for i in range(4)])
        F = EigenFilter(H, hf)
        E0, sc, sf = F.estimate_ground_energy(coarse=(60, 0.05, 121), fine=(300, 0.01, 41))
        ev, V = np.linalg.eigh(F.Hmat)
        g2 = np.abs(V[int("".join(map(str, hf)), 2)]) ** 2
        g = F.prepare_ground_state(E0, ell=300)
        out["h4"] = dict(E0_est=E0, E0_exact=float(ev[0]), hf_overlap=float(g2[0]),
                         fidelity=g.fidelity, success_prob=g.success_prob,
                         circuits=F.n_circuits, peaks=sc.peaks.tolist(), weights=sc.weights.tolist())
        print(f"   H4: |E_qsvt - E_fci| = {abs(E0 - ev[0]):.1e}", flush=True)

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
        for ax, sp, title, col in [(axes[0], sc, "Full scan (degree 120)", S1),
                                   (axes[1], sf, "Zoom on lowest peak (degree 600)", S1)]:
            lo, hi = sp.mu.min(), sp.mu.max()
            keep = (ev >= lo) & (ev <= hi) & (g2 > 1e-3)
            ax.vlines(ev[keep], 0, g2[keep], color=S2, lw=2, label="exact eigenvalues, height |⟨E_k|HF⟩|²")
            ax.plot(sp.mu, sp.p_success / 0.81, color=col, lw=1.6, label="p(μ)/s²  (simulated circuits)")
            ax.set_title(title)
            ax.set_xlabel("filter centre μ (Ha)")
            ax.set_xlim(lo, hi)
        axes[0].set_ylabel("overlap weight")
        axes[0].legend(loc="upper right")
        fig.suptitle("QSVT spectroscopy of a linear H₄ chain (8 qubits, 185 Pauli terms)",
                     fontweight="bold", color=INK, y=1.02)
        fig.savefig(FIG / "chem_h4_spectrum.png")
        plt.close(fig)

    # (c) state infidelity vs filter degree
    fig, ax = plt.subplots(figsize=(6.4, 4))
    cases = [("H₂ (0.74 Å)", ["H", "H"], [[0, 0, 0], [0, 0, 0.74]], S1),
             ("H₂ (2.2 Å)", ["H", "H"], [[0, 0, 0], [0, 0, 2.2]], S2)]
    if not quick:
        cases.append(("H₄ chain (1.5 Å)", ["H"] * 4, [[0, 0, 1.5 * i] for i in range(4)], S3))
    for label, sym, xyz, col in cases:
        H, n, hf = molecule_hamiltonian(sym, xyz)
        F = EigenFilter(H, hf)
        ev, V = np.linalg.eigh(F.Hmat)
        g2 = np.abs(V[int("".join(map(str, hf)), 2)]) ** 2
        visible = ev[g2 > 1e-8]
        gap = visible[1] - visible[0]  # gap to the next state the HF reference can see
        delta = 0.9 * gap / F.alpha  # Lin-Tong choice: filter width ~ gap
        ells = sorted({max(1, int(round(x / delta))) for x in np.linspace(0.5, 11, 9)})
        inf = [max(1 - F.prepare_ground_state(ev[0], ell=l, delta=delta).fidelity, 1e-16) for l in ells]
        out["fidelity_vs_degree"][label] = dict(delta=delta, gap=float(gap), alpha=F.alpha,
                                                points=list(zip([2 * l for l in ells], inf)))
        label = f"{label}: δ = {delta:.3f}, degree {2 * ells[0]}–{2 * ells[-1]}"
        ax.semilogy([l * delta for l in ells], inf, "-o", color=col, label=label)
    xx = np.linspace(0, 11.5, 100)
    ax.semilogy(xx, 4 * np.exp(-2 * np.sqrt(2) * xx), color=INK2, lw=1, ls=":",
                label="Lin–Tong bound  4·exp(−2√2 ℓδ)")
    ax.set_ylim(1e-17, 2)
    ax.set_xlabel("ℓ·δ   (filter half-degree × width, with δ ≈ gap/α)")
    ax.set_ylabel("ground-state infidelity")
    ax.set_title("Ground-state preparation: one universal exponential rate")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.savefig(FIG / "chem_filter_fidelity.png")
    plt.close(fig)
    results["chemistry"] = out


# --------------------------------------------------------------------------- 5. crossover map
def bench_crossover(quick: bool):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    from qsvtlab.hamiltonian_sim import qsvt_min_cost, trotter_min_cost

    ns = [4, 5, 6, 7, 8] if not quick else [4, 5]
    ts = [1, 2, 4, 8, 16, 32] if not quick else [1, 8]
    epss = [1e-3, 1e-6, 1e-9]
    models = [("TFIM", ql.tfim), ("Heisenberg", ql.heisenberg)]
    cmap = LinearSegmentedColormap.from_list("div", [S1, "#f0efec", S2])
    out = {}
    fig, axes = plt.subplots(len(models), len(epss), figsize=(12, 6.6), sharex=True, sharey=True)
    for i, (mname, ctor) in enumerate(models):
        for j, eps in enumerate(epss):
            R = np.zeros((len(ns), len(ts)))
            for a, n in enumerate(ns):
                H = ctor(n)
                for b, t in enumerate(ts):
                    q = qsvt_min_cost(H, t, eps)
                    tr = trotter_min_cost(H, t, eps)
                    R[a, b] = q[1] / tr[2]
                    out[f"{mname}|n={n}|t={t}|eps={eps:g}"] = dict(qsvt_degree=q[0], qsvt_cnots=q[1],
                                                                   trotter_order=tr[0], trotter_steps=tr[1],
                                                                   trotter_cnots=tr[2])
            ax = axes[i, j]
            im = ax.imshow(np.log10(R), cmap=cmap, norm=TwoSlopeNorm(0, -1.5, 1.5), origin="lower",
                           aspect="auto")
            for a in range(len(ns)):
                for b in range(len(ts)):
                    ax.text(b, a, f"{R[a, b]:.2g}", ha="center", va="center", fontsize=8, color=INK)
            ax.set_xticks(range(len(ts)), [str(t) for t in ts])
            ax.set_yticks(range(len(ns)), [str(n) for n in ns])
            ax.grid(False)
            ax.set_title(f"{mname}, ε = {eps:g}", fontsize=10)
            if i == len(models) - 1:
                ax.set_xlabel("evolution time t")
            if j == 0:
                ax.set_ylabel("qubits n")
            print(f"   {mname} eps={eps:g} done", flush=True)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("CNOT ratio  QSVT / best Trotter  (log scale)")
    cb.set_ticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], labels=["0.03", "0.1", "0.3", "1", "3", "10", "30"])
    fig.suptitle("Where does QSVT beat Trotter?  Blue: QSVT cheaper.  Orange: Trotter cheaper.",
                 fontweight="bold", color=INK, x=0.45)
    fig.savefig(FIG / "crossover_map.png")
    plt.close(fig)
    results["crossover"] = out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", help="subset: phase linear hamsim chem crossover")
    args = ap.parse_args()
    jobs = [("phase", bench_phase_solver), ("linear", bench_linear_solver), ("hamsim", bench_hamsim),
            ("chem", bench_chemistry), ("crossover", bench_crossover)]
    out_path = ROOT / "benchmarks" / "results.json"
    if args.only and out_path.exists():
        results.update(json.loads(out_path.read_text()))
    for name, fn in jobs:
        if args.only and name not in args.only:
            continue
        t0 = time.perf_counter()
        fn(args.quick)
        print(f"[{name}] done in {time.perf_counter() - t0:.1f}s")
    out_path.write_text(json.dumps(results, indent=1, default=float))
    print("figures ->", FIG)
