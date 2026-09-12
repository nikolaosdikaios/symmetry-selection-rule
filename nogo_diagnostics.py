#!/usr/bin/env python3
"""
nogo_diagnostics.py  --  real data for SM Fig. S1 (fig4)
========================================================
replicate.py's task_nogo() computes the 24 restart optima of the
infinite-protection contest at delta/J = 0.3 but only PRINTS the best and
discards the rest, and it runs no differential-evolution search. This
standalone script imports replicate.py (no edits) and produces both curves
Fig. S1 needs:

  restart_optima_cache.npy   the Fisher optimum from each of 100 cold L-BFGS-B
                             restarts of the INDEPENDENT baseline at
                             delta/J = 0.3 (histogram, panel a)
  de_trace_cache.npy         best-so-far Fisher along a differential-evolution
                             search of the same baseline (panel b)

Using the independent baseline (not the infinite-protection arm) matches the
Letter/SM text for Fig. S1, whose point is that the CLASSICAL baseline optimum
is a broad, repeatedly found basin (44%% within 5%%, 7%% within 1%%) and is
independently confirmed by a global search converging to ~5.97e-2.

Run:
    python3 nogo_diagnostics.py     # ~1-3 min
    python3 export_csvs.py          # picks up both caches -> data/*.csv
    python3 make_figures.py         # fig4 becomes REAL

Requires replicate.py in the same folder.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize, differential_evolution

__version__ = "2026-07-24-with-S8"

import replicate as R

DHZ = 3.0                            # delta/J = 0.3 at J = 10 Hz
N_RESTART = 100


def baseline_objective(q):
    """Negative receiver-referred Fisher of the INDEPENDENT baseline at DHZ,
    collective readout, using replicate.py's exact forward model."""
    L0 = R.L0_of(DHZ, "base", collapse=False)
    return -R.fisher_collective(q, L0, R.MEAS_COLL)


def restart_histogram():
    rng = np.random.default_rng(2024)
    optima = []
    for s in range(N_RESTART):
        p = np.array([rng.uniform(lo, hi) for lo, hi in R.BND_COLL])
        res = minimize(baseline_objective, p, method="L-BFGS-B",
                       bounds=R.BND_COLL, options={"maxiter": 150, "ftol": 1e-14})
        optima.append(-res.fun)
    optima = np.array(optima)
    best = optima.max()
    w5 = 100 * np.mean(optima > 0.95 * best)
    w1 = 100 * np.mean(optima > 0.99 * best)
    print(f"    {N_RESTART} restarts: best {best:.4e}, "
          f"{w5:.0f}%% within 5%%, {w1:.0f}%% within 1%%")
    np.save("restart_optima_cache.npy", optima)
    return best


def de_trace():
    trace = []

    def cb(xk, convergence=None):
        trace.append(-baseline_objective(xk))

    differential_evolution(baseline_objective, R.BND_COLL, seed=1,
                           maxiter=100, tol=1e-14, init="sobol",
                           updating="deferred", polish=False, callback=cb)
    trace = np.array(trace)
    out = np.column_stack([np.arange(1, len(trace) + 1), trace])
    print(f"    differential evolution converged to {trace[-1]:.4e} "
          f"in {len(trace)} generations")
    np.save("de_trace_cache.npy", out)


# ---------------------------------------------------------------------------
# Consequences of an unconverged reference (SM Sec. S8)
# ---------------------------------------------------------------------------
# The Supplement states that a piecewise-constant reference optimized on a
# coarse time grid stalls far below the optimum reached by the physically
# motivated prepare-store-readout ansatz. That claim previously had no script
# behind it. The two functions below supply one, so the quoted factor is a
# reproducible output rather than a recorded observation.

N_SLICE = 6          # deliberately coarse
T_CTRL = 0.10        # s of piecewise-constant control before acquisition


def fisher_pwc(u, L0, meas, n_slice=N_SLICE, T_ctrl=T_CTRL):
    """Receiver-referred Fisher for a piecewise-constant COLLECTIVE control.

    The control is n_slice slices of constant (ux, uy) amplitude applied to
    Fx and Fy, followed by the same multisample free induction decay used
    everywhere else. This is the coarse-grid ansatz of SM Sec. S8, not the
    physically motivated ansatz of Sec. S6."""
    dt = T_ctrl / n_slice
    D = R.D
    Y = np.concatenate([R.rho0v, np.zeros(D, dtype=complex)])
    for k in range(n_slice):
        Lk = L0 + R.comm_super(u[2 * k] * R.Fx + u[2 * k + 1] * R.Fy)
        M = np.zeros((2 * D, 2 * D), dtype=complex)
        M[:D, :D] = Lk
        M[D:, D:] = Lk
        M[D:, :D] = R.dLd
        Y = expm(M * dt) @ Y
    Ff = R._aug_free(L0, R.DTF)
    tot = 0.0
    for _ in range(R.NF):
        for m in meas:
            tot += 2 * abs(m @ Y[D:]) ** 2
        Y = Ff @ Y
    return tot


def unconverged_reference(n_restart=40, seed=11):
    """Cold-started optimization of the coarse piecewise-constant reference.

    IMPORTANT, PLEASE READ BEFORE QUOTING SM Sec. S8.
    The Supplement states that a piecewise-constant reference on a coarse time
    grid stalls at 1.75e-3, a factor of thirty four below the optimum of the
    physically motivated ansatz. This function implements a straightforward
    reading of that setup, namely N_SLICE constant (ux, uy) slices on Fx and
    Fy over T_CTRL seconds followed by the standard acquisition, and it does
    NOT reproduce the stall. Every cold start converges to within a few percent
    of the full optimum. The reason is visible in the physics. The independent
    reference carries no protected sector, so its optimal strategy is a simple
    excite-and-acquire that a six-slice grid represents easily, and there is no
    long-storage basin for a coarse grid to miss.

    The function therefore returns the DISTRIBUTION of cold-start optima and
    the factor by which the median understates the physical-ansatz optimum,
    rather than asserting a number. If the 1.75e-3 figure is to remain in the
    Supplement, the original coarse-grid configuration should be specified here
    so that it becomes reproducible. Otherwise the claim should be reported as
    a recorded observation of this work rather than a scripted result."""
    L0 = R.L0_of(DHZ, "base", collapse=False)
    bnds = [(-60.0, 60.0)] * (2 * N_SLICE)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_restart):
        p = np.array([rng.uniform(lo, hi) for lo, hi in bnds])
        res = minimize(lambda q: -fisher_pwc(q, L0, R.MEAS_COLL), p,
                       method="L-BFGS-B", bounds=bnds,
                       options={"maxiter": 150, "ftol": 1e-14})
        vals.append(-res.fun)
    return np.array(vals)


def main():
    print("=" * 70)
    print(f"nogo_diagnostics.py  version {__version__}")
    print("No-go diagnostics for Fig. S1 and Sec. S8  (importing replicate.py)")
    print("=" * 70)
    print("  (a) cold-restart histogram of the independent baseline ...")
    restart_histogram()
    print("  (b) differential-evolution best-so-far trace ...")
    de_trace()
    print("  (c) coarse-grid reference, SM Sec. S8 ...")
    vals = unconverged_reference()
    L0 = R.L0_of(DHZ, "base", collapse=False)
    rng = np.random.default_rng(3)
    f_phys = max(-minimize(baseline_objective,
                           np.array([rng.uniform(lo, hi) for lo, hi in R.BND_COLL]),
                           method="L-BFGS-B", bounds=R.BND_COLL,
                           options={"maxiter": 300, "ftol": 1e-14}).fun
                 for _ in range(20))
    med = float(np.median(vals))
    print(f"      coarse-grid cold starts: best {vals.max():.3e}, "
          f"median {med:.3e}, worst {vals.min():.3e}")
    print(f"      physical-ansatz optimum: {f_phys:.3e}")
    print(f"      median understates it by {f_phys/med:.1f}x, "
          f"worst by {f_phys/vals.min():.1f}x")
    if f_phys / med < 2:
        print("      NOTE the coarse grid does NOT stall here. SM Sec. S8 quotes")
        print("      1.75e-3 and a factor of 34, which this setup does not")
        print("      reproduce. Specify the original coarse-grid configuration")
        print("      or report that figure as a recorded observation.")
    np.save("unconverged_reference_cache.npy", vals)

    print("\nNext:  python export_csvs.py  &&  python make_figures.py")


if __name__ == "__main__":
    main()
