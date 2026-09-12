#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replicate.py
============
Full reproduction of every numerical result and figure in

    "The symmetry that protects a long-lived spin state also hides it:
     a symmetry selection rule for coherence-enhanced spin metrology."

Dependencies: numpy, scipy, matplotlib  (nothing else).

Command-line usage
------------------
    python replicate.py verify      # brightness/darkness algebra            (seconds)
    python replicate.py nogo        # fixed-protection same-ansatz contest   (~1 min)
    python replicate.py tension     # protection / accessibility / sensitivity curves (~1 min)
    python replicate.py phase       # symmetric phase-diagram sweep          (~6-9 min)
    python replicate.py loophole    # broken-symmetry sweep                  (~6-9 min)
    python replicate.py figures     # regenerate the 3 figures               (seconds)
    python replicate.py all         # sweeps + figures                       (~20 min)

`phase`, `loophole` and `tension` write *.npy caches; `figures` plots from those
caches if present, otherwise from the embedded converged values reported in the paper.

Model summary
-------------
Two spin-1/2 nuclei, scalar coupling J and chemical-shift difference delta (Hz):
    H = 2*pi*J (I1.I2) + 2*pi*(delta/2)(I1z - I2z).
Estimate delta from the inductively detected free-induction decay, maximizing the
classical Fisher information over control waveforms.  The "protected" arm has a
relaxation-protected singlet sector; the "independent" baseline is uncoupled spins
with matched T1, T2 and no protected sector.
"""

import sys, os, time
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

# ======================================================================
# 1.  OPERATORS  (two spin-1/2, product basis, column-stacking vec)
# ======================================================================
TWOPI = 2 * np.pi
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
kr = np.kron

Iz0, Iz1 = kr(sz, I2), kr(I2, sz)
Ix0, Ix1 = kr(sx, I2), kr(I2, sx)
Iy0, Iy1 = kr(sy, I2), kr(I2, sy)
Ip0, Ip1 = Ix0 + 1j * Iy0, Ix1 + 1j * Iy1
Im0, Im1 = Ix0 - 1j * Iy0, Ix1 - 1j * Iy1
Fz, Fx, Fy, Fp = Iz0 + Iz1, Ix0 + Ix1, Iy0 + Iy1, Ip0 + Ip1
Idot = Ix0 @ Ix1 + Iy0 @ Iy1 + Iz0 @ Iz1                 # I1 . I2
d = 4
D = d * d

def vecf(M):                                            # column-stacking vectorization
    return M.flatten(order="F")

def comm_super(H):                                      # super-operator for -i[H, .]
    return -1j * (np.kron(np.eye(d), H) - np.kron(H.T, np.eye(d)))

# singlet-triplet states as vectors in the product basis {uu, ud, du, dd}
up = np.array([1, 0], dtype=complex)
dn = np.array([0, 1], dtype=complex)
Svec = (kr(up, dn) - kr(dn, up)) / np.sqrt(2)
Tpv = kr(up, up)
T0v = (kr(up, dn) + kr(dn, up)) / np.sqrt(2)
Tmv = kr(dn, dn)
def ket(a, b):                                          # |a><b|
    return np.outer(a, b.conj())

# ======================================================================
# 2.  RELAXATION
# ======================================================================
def R_pheno(R1, R2, Rlls):
    """Phenomenological relaxation diagonal in the singlet-triplet operator basis.
    Singlet population imbalance and the singlet-T0 zero-quantum coherence (which
    carries delta) relax at the long-lived rate Rlls; triplet orders at R1; all
    observable coherences at R2."""
    R = np.zeros((D, D), dtype=complex)
    def add(O, rate):
        v = vecf(O); v = v / np.sqrt(np.vdot(v, v))
        R[:] += -rate * np.outer(v, v.conj())
    Ss = ket(Svec, Svec)
    add(Ss - 0.25 * np.eye(d), Rlls)                                  # protected: singlet order
    add(ket(Tpv, Tpv) - ket(Tmv, Tmv), R1)                           # triplet Zeeman order
    add(ket(Tpv, Tpv) + ket(Tmv, Tmv) - 2 * ket(T0v, T0v), R1)       # triplet dipolar order
    add(ket(Svec, T0v), Rlls); add(ket(T0v, Svec), Rlls)             # protected: S-T0 ZQ (carries delta)
    pairs = [(Tpv, T0v), (T0v, Tmv), (Tpv, Tmv), (Svec, Tpv), (Svec, Tmv),
             (Tpv, Svec), (Tmv, Svec), (T0v, Tpv), (Tmv, T0v), (Tmv, Tpv)]
    for a, b in pairs:
        add(ket(a, b), R2)                                           # observable coherences: fast
    return R

def R_indep(R1, R2):
    """Matched single-spin amplitude damping (R1) + pure dephasing -> same T1,T2,
    no protected sector."""
    R = np.zeros((D, D), dtype=complex)
    Rphi = R2 - R1 / 2
    for (Ip, Im, Iz) in [(Ip0, Im0, Iz0), (Ip1, Im1, Iz1)]:
        for c in [np.sqrt(R1 / 2) * Ip, np.sqrt(R1 / 2) * Im, np.sqrt(2 * max(Rphi, 0)) * Iz]:
            cdc = c.conj().T @ c
            R += np.kron(c.conj(), c) - 0.5 * np.kron(np.eye(d), cdc) - 0.5 * np.kron(cdc.T, np.eye(d))
    return R

# physical collapse of the singlet lifetime with delta/J  (Methods, Eq. 4)
def Rlls_eff(delta_over_J, R2, Rlls_bare):
    x = delta_over_J ** 2 / (1 + delta_over_J ** 2)      # sin^2(2 theta), tan(2 theta) = delta/J
    return Rlls_bare + (R2 - Rlls_bare) * x

# ----- default lifetimes used throughout -----
J = 10.0                          # coupling (Hz)
T1, T2 = 1.2, 0.2                 # s
R1, R2 = 1 / T1, 1 / T2
TLLS_BARE = 5.0                   # bare singlet lifetime (s) = 25 * T2
RLLS_BARE = 1 / TLLS_BARE
Hc = TWOPI * J * Idot
dLd = comm_super(0.5 * (Iz0 - Iz1))          # d/d(delta) of the Liouvillian (delta in Hz)
rho0v = vecf(Fz.astype(complex))             # thermal order

def L0_of(dHz, arm, collapse=True):
    """Liouvillian at shift delta = dHz Hz for the given arm."""
    Hs = (TWOPI * dHz / 2) * (Iz0 - Iz1)
    if arm == "base":
        return comm_super(Hs) + R_indep(R1, R2)
    Rl = Rlls_eff(dHz / J, R2, RLLS_BARE) if collapse else RLLS_BARE
    return comm_super(Hs + Hc) + R_pheno(R1, R2, Rl)

# ======================================================================
# 3.  CONTROL ANSATZ + FISHER (exact d/d(delta) via Van Loan augmentation)
# ======================================================================
NF, DTF = 16, 0.0375             # 16-sample, 0.6 s free-induction decay

def _aug_free(L0, t):
    """exp of the 2D x 2D block [[L0,0],[dLd,L0]] t  ->  propagates [rho ; d_delta rho]."""
    M = np.zeros((2 * D, 2 * D), dtype=complex)
    M[:D, :D] = L0; M[D:, D:] = L0; M[D:, :D] = dLd      # dLd in LOWER-left (correct sign)
    return expm(M * t)

def _pulse_collective(th, ph):
    U = expm(-1j * th * (np.cos(ph) * Fx + np.sin(ph) * Fy))
    return np.kron(U.conj(), U)

def _pulse_selective(th0, ph0, th1, ph1):
    H = th0 * (np.cos(ph0) * Ix0 + np.sin(ph0) * Iy0) + th1 * (np.cos(ph1) * Ix1 + np.sin(ph1) * Iy1)
    U = expm(-1j * H)
    return np.kron(U.conj(), U)

def _aug_pulse(P):
    Z = np.zeros((D, D), dtype=complex)
    return np.block([[P, Z], [Z, P]])

def fisher_collective(p, L0, meas):
    """11-param ansatz: 4 collective pulses + (d1, tau, d2) + FID.  meas = list of vec(M^T)."""
    th1, ph1, th2, ph2, d1, tau, th3, ph3, d2, th4, ph4 = p
    free = lambda t: _aug_free(L0, t)
    Y = np.concatenate([rho0v, np.zeros(D, dtype=complex)])
    Y = _aug_pulse(_pulse_collective(th1, ph1)) @ Y; Y = free(d1) @ Y
    Y = _aug_pulse(_pulse_collective(th2, ph2)) @ Y; Y = free(tau) @ Y
    Y = _aug_pulse(_pulse_collective(th3, ph3)) @ Y; Y = free(d2) @ Y
    Y = _aug_pulse(_pulse_collective(th4, ph4)) @ Y
    Ff = free(DTF); tot = 0.0
    for _ in range(NF):
        for m in meas:
            tot += 2 * abs(m @ Y[D:]) ** 2
        Y = Ff @ Y
    return tot

def fisher_selective(p, L0, meas):
    """15-param ansatz: 3 selective pulses (4 params each) + (d1, tau, d2) + FID."""
    free = lambda t: _aug_free(L0, t)
    Y = np.concatenate([rho0v, np.zeros(D, dtype=complex)])
    a = p
    Y = _aug_pulse(_pulse_selective(a[0], a[1], a[2], a[3])) @ Y; Y = free(a[4]) @ Y
    Y = _aug_pulse(_pulse_selective(a[5], a[6], a[7], a[8])) @ Y; Y = free(a[9]) @ Y
    Y = _aug_pulse(_pulse_selective(a[10], a[11], a[12], a[13])) @ Y; Y = free(a[14]) @ Y
    Ff = free(DTF); tot = 0.0
    for _ in range(NF):
        for m in meas:
            tot += 2 * abs(m @ Y[D:]) ** 2
        Y = Ff @ Y
    return tot

# measurement operators
MEAS_COLL = [vecf(Fp.T)]                       # inductive (collective) detection
MEAS_INDIV = [vecf(Ip0.T), vecf(Ip1.T)]        # individual (two-receiver) detection

BND_COLL = [(-4, 4)] * 4 + [(0, 0.2), (0, 2.0)] + [(-4, 4)] * 2 + [(0, 0.2)] + [(-4, 4)] * 2
BND_INDIV = [(-4, 4)] * 4 + [(0, 0.2)] + [(-4, 4)] * 4 + [(0, 3.0)] + [(-4, 4)] * 4 + [(0, 0.2)]

def optimize(dHz, arm, kind, meas, n_restart, seed, collapse=True, warm=None):
    """Maximize Fisher over controls. kind in {'collective','selective'}."""
    L0 = L0_of(dHz, arm, collapse)
    fish = fisher_collective if kind == "collective" else fisher_selective
    bnds = BND_COLL if kind == "collective" else BND_INDIV
    storage_idx = 5 if kind == "collective" else 9        # the long-storage delay
    rng = np.random.default_rng(seed)
    starts = [warm] if warm is not None else []
    for s in range(n_restart):
        p = np.array([rng.uniform(lo, hi) for lo, hi in bnds])
        if arm == "full" and s % 2 == 1:                  # storage-biased warm start
            p[storage_idx] = rng.uniform(1.0, 2.0 if kind == "collective" else 2.5)
        starts.append(p)
    best, bp = -1.0, None
    for p0 in starts:
        r = minimize(lambda q: -fish(q, L0, meas), p0, method="L-BFGS-B",
                     bounds=bnds, options={"maxiter": 150, "ftol": 1e-14})
        if -r.fun > best:
            best, bp = -r.fun, r.x
    return best, bp

# ======================================================================
# 4.  EMBEDDED CONVERGED RESULTS  (as reported in the paper)
# ======================================================================
PHASE_EMB = dict(   # symmetric (collective control + collective detection), collapse model
    dj=[0.01, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.50, 0.80, 1.20, 2.00, 3.00, 5.00],
    Ff=[3.81e-7, 3.52e-6, 2.32e-5, 8.31e-5, 2.25e-4, 2.10e-3, 5.06e-3, 1.44e-2, 1.91e-2, 2.31e-2, 2.67e-2, 2.79e-2, 2.79e-2],
    Fb=[1.14e-3, 9.64e-3, 2.39e-2, 4.71e-2, 6.64e-2, 6.48e-2, 5.96e-2, 5.31e-2, 5.20e-2, 5.33e-2, 5.39e-2, 5.56e-2, 5.65e-2])
LOOP_EMB = dict(    # broken symmetry (individual control + individual detection), collapse model
    dj=[0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.50, 0.80, 1.20, 2.00, 3.00, 5.00],
    Ff=[2.94e-3, 3.06e-2, 4.99e-2, 1.053e-1, 2.087e-1, 3.351e-1, 4.039e-1, 3.688e-1, 1.675e-1, 6.662e-2, 4.057e-2, 2.045e-2, 2.297e-2, 2.452e-2],
    Fb=[5.06e-2, 5.34e-2, 5.34e-2, 5.34e-2, 5.34e-2, 5.34e-2, 5.34e-2, 5.34e-2, 5.34e-2, 5.34e-2, 5.37e-2, 5.38e-2, 5.23e-2, 5.20e-2])
ACCESS_EMB = dict(  # max reachable singlet-T0 coherence (pure coherent reachability)
    dj=[0.03, 0.10, 0.20, 0.30, 0.50, 0.80, 1.20, 2.0, 3.0, 5.0],
    v=[0.060, 0.198, 0.385, 0.550, 0.800, 0.976, 1.0, 1.0, 1.0, 1.0])

# ======================================================================
# 5.  TASKS
# ======================================================================
def task_verify():
    print("=== brightness / darkness algebra (Eqs. 2-3) ===")
    print("  F+|S>            (collective, must be 0) :", round(np.linalg.norm(Fp @ Svec), 12))
    print("  I+0|S> = -|T+>/sqrt2                     :", np.allclose(Ip0 @ Svec, -Tpv / np.sqrt(2)))
    print("  |(I+0 - I+1)|S>| (antisymmetric, bright) :", round(np.linalg.norm((Ip0 - Ip1) @ Svec), 6), " (= sqrt2)")
    print("  (Iz0 - Iz1)|S> = |T0>                    :", np.allclose((Iz0 - Iz1) @ Svec, T0v))
    # leakage of collective generators across the S/T split (must be ~0)
    P_s = ket(Svec, Svec)
    P_t = np.eye(d) - P_s
    leak = lambda O: np.linalg.norm(P_t @ O @ P_s)
    print("  leakage S->T of Fx, Fy, Fz, I1.I2        :",
          [float(np.round(leak(O), 12)) for O in (Fx, Fy, Fz, Idot)])
    print("  leakage S->T of H_delta = (Iz0-Iz1)      :", round(leak(Iz0 - Iz1), 4), " (the only door)")

def task_nogo():
    """Fixed-protection, same-ansatz contest at delta/J = 0.3 (delta = 3 Hz)."""
    print("=== selection rule: fixed-protection same-ansatz contest (delta = 3 Hz, delta/J = 0.3) ===")
    dHz = 3.0
    t = time.time()
    fb, _ = optimize(dHz, "base", "collective", MEAS_COLL, 16, 21, collapse=False)
    print(f"  independent baseline           : {fb:.4e}")
    # infinite protection
    L_inf = comm_super((TWOPI * dHz / 2) * (Iz0 - Iz1) + Hc) + R_pheno(R1, R2, 1e-9)
    rng = np.random.default_rng(61); best = -1
    for s in range(24):
        p = np.array([rng.uniform(lo, hi) for lo, hi in BND_COLL])
        if s % 2 == 1: p[5] = rng.uniform(0.8, 1.8)
        r = minimize(lambda q: -fisher_collective(q, L_inf, MEAS_COLL), p,
                     method="L-BFGS-B", bounds=BND_COLL, options={"maxiter": 150, "ftol": 1e-14})
        best = max(best, -r.fun)
    print(f"  protected, T_LLS = infinity    : {best:.4e}   (ratio {best/fb:.3f})")
    # realistic protection 25*T2
    ff5, _ = optimize(dHz, "full", "collective", MEAS_COLL, 24, 51, collapse=False)
    print(f"  protected, T_LLS = 25*T2 (5 s) : {ff5:.4e}   (ratio {ff5/fb:.3f})")
    print(f"  [elapsed {time.time()-t:.0f}s]  -> even infinite protection loses to the baseline.")

def task_tension():
    """Protection factor, accessibility, sensitivity vs delta/J (Fig. 2a inputs)."""
    print("=== protection / accessibility / sensitivity vs delta/J ===")
    STc = ket(Svec, T0v); STc /= np.sqrt(np.vdot(vecf(STc), vecf(STc)))   # delta-carrier
    def access(dHz):  # max reachable |S-T0 coherence| from Fz by pure coherent control
        Hcoh = comm_super((TWOPI * dHz / 2) * (Iz0 - Iz1) + Hc)
        m = vecf(STc.conj()); r0 = vecf(Fz.astype(complex)); rng = np.random.default_rng(1); best = 0.0
        def neg(p):
            P = lambda th, ph: np.kron(expm(-1j*th*(np.cos(ph)*Fx+np.sin(ph)*Fy)).conj(),
                                       expm(-1j*th*(np.cos(ph)*Fx+np.sin(ph)*Fy)))
            y = r0
            y = P(p[0],p[1]) @ y; y = expm(Hcoh*abs(p[2])) @ y; y = P(p[3],p[4]) @ y
            y = expm(Hcoh*abs(p[5])) @ y; y = P(p[6],p[7]) @ y
            return -abs(m @ y)
        for _ in range(16):
            p0 = [rng.uniform(0, np.pi), rng.uniform(0, TWOPI), rng.uniform(0, 0.2),
                  rng.uniform(0, np.pi), rng.uniform(0, TWOPI), rng.uniform(0, 0.2),
                  rng.uniform(0, np.pi), rng.uniform(0, TWOPI)]
            best = max(best, -minimize(neg, p0, method="Nelder-Mead",
                                       options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-9}).fun)
        return best
    print("  d/J   T_LLS_eff/T2   sensitivity   accessibility")
    out = {"dj": [], "prot": [], "sens": [], "acc": []}
    for r in [0.03, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0]:
        prot = (1 / Rlls_eff(r, R2, RLLS_BARE)) / T2
        sens = r / np.sqrt(r * r + 1)
        acc = access(r * J)
        out["dj"].append(r); out["prot"].append(prot); out["sens"].append(sens); out["acc"].append(acc)
        print(f"  {r:4.2f}   {prot:10.2f}   {sens:9.3f}    {acc:.3f}")
    np.save("tension_data.npy", out)
    print("  saved tension_data.npy")

def _sweep(kind, meas, fname, grid, collapse=True):
    """Generic delta/J sweep with continuation warm starts; saves {r:(Ff,Fb)} to fname."""
    data = np.load(fname, allow_pickle=True).item() if os.path.exists(fname) else {}
    wF, wB = data.get("_wF"), data.get("_wB"); t = time.time()
    nF, nB = (3, 2) if kind == "collective" else (5, 4)
    for r in grid:
        dHz = r * J
        fF, wF = optimize(dHz, "full", kind, meas, nF, int(r * 1000) + 7, collapse, wF)
        fB, wB = optimize(dHz, "base", kind, meas, nB, int(r * 1000) + 13, collapse, wB)
        data[r] = (fF, fB); data["_wF"], data["_wB"] = wF, wB; np.save(fname, data)
        print(f"  d/J={r:5.2f}  full={fF:.4e}  base={fB:.4e}  ratio={fF/fB:.3f}  [{time.time()-t:.0f}s]")
    return data

def task_phase():
    print("=== symmetric phase-diagram sweep (collective control + detection) ===")
    grid = [0.01, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.50, 0.80, 1.20, 2.00, 3.00, 5.00]
    _sweep("collective", MEAS_COLL, "phase_data.npy", grid)

def task_loophole():
    print("=== broken-symmetry sweep (individual control + detection) ===")
    grid = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.50, 0.80, 1.20, 2.00, 3.00, 5.00]
    _sweep("selective", MEAS_INDIV, "loophole_data.npy", grid)

# ======================================================================
# 6.  FIGURES
# ======================================================================
def _load(fname, emb):
    if os.path.exists(fname):
        d0 = np.load(fname, allow_pickle=True).item()
        ks = sorted(k for k in d0 if not str(k).startswith("_"))
        return (np.array(ks), np.array([d0[k][0] for k in ks]), np.array([d0[k][1] for k in ks]))
    return np.array(emb["dj"]), np.array(emb["Ff"]), np.array(emb["Fb"])

def fig_mechanism():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.7))
    Cd, Cb, Cx, Ca, Cg = "#2166ac", "#1b7837", "#b2182b", "#e6550d", "#777777"
    def panel(ax, mode):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.4, 3.4), 3.0, 4.0, boxstyle="round,pad=0.05", lw=1.4, ec=Cd, fc="#eaf1f8"))
        ax.add_patch(FancyBboxPatch((6.0, 2.3), 3.5, 5.9, boxstyle="round,pad=0.05", lw=1.4, ec=Cb, fc="#eaf4ec"))
        ax.text(1.9, 7.05, "SINGLET", ha="center", fontsize=8.5, color=Cd, weight="bold")
        ax.text(1.9, 6.65, "dark · protected", ha="center", fontsize=7, color=Cd, style="italic")
        ax.text(7.75, 7.85, "TRIPLET (bright)", ha="center", fontsize=8.5, color=Cb, weight="bold")
        ax.plot([1.0, 2.8], [5.3, 5.3], color=Cd, lw=2.6); ax.text(0.78, 5.3, r"$|S\rangle$", ha="right", va="center", fontsize=9)
        for y, lab in [(6.7, r"$|T_+\rangle$"), (5.1, r"$|T_0\rangle$"), (3.5, r"$|T_-\rangle$")]:
            ax.plot([6.6, 9.0], [y, y], color=Cb, lw=2.6); ax.text(9.2, y, lab, va="center", fontsize=9)
        for y0, y1 in [(6.55, 5.25), (4.95, 3.65)]:
            ax.add_patch(FancyArrowPatch((7.8, y0), (7.8, y1), arrowstyle="<->", mutation_scale=10, lw=1.6, color=Cb))
        ax.text(8.15, 4.95, r"$F_x,F_y$" + "\n" + r"$H_c$", fontsize=7, color=Cb, va="center")
        ax.annotate("", xy=(6.5, 6.0), xytext=(5.0, 8.7), arrowprops=dict(arrowstyle="->", lw=1.3, color=Cb))
        ax.text(4.6, 9.0, r"coil  $F_+$", fontsize=7.5, color=Cb, ha="center")
        ax.text(4.6, 8.55, "(bright only)", fontsize=6.5, color=Cb, ha="center")
        ax.add_patch(FancyArrowPatch((2.8, 5.35), (6.6, 5.1), arrowstyle="<->", mutation_scale=11, lw=1.4, color=Cx,
                                     connectionstyle="arc3,rad=-0.15"))
        ax.text(4.7, 6.05, r"$H_\delta=\frac{\delta}{2}(I_{z0}\!-\!I_{z1})$", ha="center", fontsize=7.8, color=Cx)
        ax.text(4.7, 5.55, r"rate $\propto\delta$  (the measured parameter)", ha="center", fontsize=6.8, color=Cx)
        ax.text(4.7, 4.55, r"gap $2\pi\sqrt{J^2+\delta^2}$", ha="center", fontsize=6.6, color="0.4")
        ax.add_patch(Rectangle((1.0, 0.4), 8.5, 1.15, lw=1.0, ec=Cg, fc="#efefef", hatch="////"))
        ax.text(5.25, 0.97, "fluctuating dipolar bath (relaxation)", ha="center", fontsize=7, color="0.3")
        ax.annotate("", xy=(7.75, 2.25), xytext=(7.75, 1.55), arrowprops=dict(arrowstyle="->", lw=1.6, color=Cg))
        ax.text(8.0, 1.9, r"fast: $T_2,T_1$", fontsize=6.6, color="0.3", va="center")
        ax.annotate("", xy=(1.9, 3.35), xytext=(1.9, 1.55), arrowprops=dict(arrowstyle="->", lw=1.4, color=Cg, ls=(0, (3, 2))))
        ax.plot(1.9, 2.45, "x", ms=9, mew=2.4, color=Cx)
        ax.text(2.15, 2.0, r"blocked: $T_{\rm LLS}\!\gg\!T_2$", fontsize=6.5, color="0.3", va="center")
        if mode == "broken":
            ax.add_patch(FancyArrowPatch((2.8, 5.45), (6.6, 6.6), arrowstyle="<->", mutation_scale=13, lw=2.8, color=Ca,
                                         connectionstyle="arc3,rad=0.18"))
            ax.text(4.7, 7.55, r"$I_{+0}\!-\!I_{+1}$ : selective,  $O(1)$", ha="center", fontsize=7.8, color=Ca, weight="bold")
    axs[0].set_title("a   Collective control + coil  (homonuclear)", fontsize=9, loc="left"); panel(axs[0], "collective")
    axs[1].set_title("b   Individual control + detection  (heteronuclear / inequivalent)", fontsize=9, loc="left"); panel(axs[1], "broken")
    fig.tight_layout(); fig.savefig("mechanism.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print("  saved mechanism.png")

def fig_phase():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    dj, Ff, Fb = _load("phase_data.npy", PHASE_EMB)
    x = np.logspace(-2, 0.8, 300)
    prot = (1 / (RLLS_BARE + (R2 - RLLS_BARE) * x ** 2 / (1 + x ** 2))) / T2
    sens = x / np.sqrt(x ** 2 + 1)
    axd, axv = np.array(ACCESS_EMB["dj"]), np.array(ACCESS_EMB["v"])
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
    fig, A = plt.subplots(3, 1, figsize=(7.0, 9.2), sharex=True, gridspec_kw={"height_ratios": [1, 1, 0.95], "hspace": 0.12})
    Cp, Cc, Cs, Cf, Cbz = "#2166ac", "#b2182b", "#d6912e", "#762a83", "#1b7837"
    a = A[0]
    a.semilogx(x, prot, color=Cp, lw=2.4); a.set_ylabel(r"protection factor $T_{\mathrm{LLS}}^{\mathrm{eff}}/T_2$", color=Cp)
    a.tick_params(axis="y", labelcolor=Cp); a.set_ylim(0, 27)
    a2 = a.twinx(); a2.semilogx(x, sens, color=Cs, lw=2.0, label=r"sensitivity $\delta/\sqrt{\delta^2+J^2}$")
    a2.plot(axd, axv, "o", color=Cc, ms=5, label="accessibility (reachable S–T0)")
    a2.plot(x[x <= 5], np.interp(x[x <= 5], axd, axv), color=Cc, lw=2.0)
    a2.set_ylabel("accessibility / sensitivity  (0–1)"); a2.set_ylim(0, 1.05)
    a.axvspan(0.3, 2.0, color="0.85", alpha=0.5, zorder=0)
    a.text(0.72, 13.5, "no overlap of\nhigh protection\n& high access", ha="center", va="center", fontsize=8.5, style="italic")
    a.plot([], [], color=Cp, lw=2.4, label=r"protection $T_{\mathrm{LLS}}^{\mathrm{eff}}/T_2$")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8, framealpha=0.9)
    a.set_title("(A)  Protection needs $\\delta\\ll J$, access/sensitivity need $\\delta\\gtrsim J$", fontsize=9.5, loc="left")
    b = A[1]
    b.loglog(dj, Fb, "s-", color=Cbz, lw=1.8, ms=6, label=r"independent baseline  $\max_u F_{\delta\delta}$")
    b.loglog(dj, Ff, "o-", color=Cf, lw=1.8, ms=6, label=r"coupled + protected (LLS)  $\max_u F_{\delta\delta}$")
    b.fill_between(dj, Ff, Fb, where=(Fb > Ff), color=Cbz, alpha=0.10)
    b.set_ylabel(r"detected-signal Fisher  $\max_u F_{\delta\delta}$"); b.set_ylim(2e-7, 2e-1)
    b.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    b.text(0.02, 8e-5, r"$F_{\mathrm{full}}\!\sim\!\delta^{2}$" + "\n(dark: small prefactor)", fontsize=8, color=Cf)
    b.text(0.02, 2.2e-3, r"$F_{\mathrm{base}}\!\sim\!\delta^2$" + "\n(sub-linewidth)", fontsize=8, color=Cbz)
    b.set_title("(B)  Maximized Fisher for the discriminator $\\delta$  (both arms optimized)", fontsize=9.5, loc="left")
    c = A[2]; ratio = Ff / Fb
    c.semilogx(dj, ratio, "o-", color="k", lw=2.0, ms=6); c.axhline(1.0, color=Cbz, lw=1.5, ls="--")
    c.fill_between([1e-2, 1e1], [1, 1], [1.2, 1.2], color=Cbz, alpha=0.12)
    c.text(1.6, 1.07, "advantage region (never reached)", color=Cbz, fontsize=8.5, va="bottom")
    c.annotate(f"peak ratio = {ratio.max():.2f}", xy=(dj[np.argmax(ratio)], ratio.max()), xytext=(1.05, 0.80),
               fontsize=9, arrowprops=dict(arrowstyle="->", lw=1, color="k"))
    c.set_ylabel(r"advantage  $F_{\mathrm{full}}/F_{\mathrm{base}}$"); c.set_xlabel(r"$\delta/J$   (chemical-shift splitting / coupling)")
    c.set_ylim(0, 1.2); c.set_xlim(1e-2, 1e1)
    c.set_title("(C)  The selection rule: the protected sector never beats the baseline", fontsize=9.5, loc="left")
    for ax in A: ax.grid(True, which="both", alpha=0.18)
    fig.suptitle(r"Protection–accessibility tension  ($J=10$ Hz, $T_2=200$ ms, $T_{\mathrm{LLS}}^{\mathrm{bare}}=5$ s)", fontsize=11, y=0.997)
    fig.savefig("phase_diagram.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"  saved phase_diagram.png  (peak ratio {ratio.max():.3f} at d/J={dj[np.argmax(ratio)]})")

def fig_loophole():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    djs, Ffs, Fbs = _load("phase_data.npy", PHASE_EMB)
    djb, Ffb, Fbb = _load("loophole_data.npy", LOOP_EMB)
    rs, rb = Ffs / Fbs, Ffb / Fbb; lo, hi = 0.0355, 0.95
    Csym, Cbrk, Cbase = "#762a83", "#e6550d", "#1b7837"
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 8.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.05], "hspace": 0.13})
    a.loglog(djb, Fbb, "-", color=Cbase, lw=1.6, label=r"baseline $\max_u F_{\delta\delta}$ (either readout)")
    a.loglog(djs, Ffs, "o-", color=Csym, lw=1.8, ms=5, label="protected arm, COLLECTIVE readout (dark)")
    a.loglog(djb, Ffb, "s-", color=Cbrk, lw=1.8, ms=5, label="protected arm, INDIVIDUAL readout (bright)")
    a.axvspan(lo, hi, color="#fdd0a2", alpha=0.45, zorder=0)
    a.annotate("", xy=(0.2, 4.04e-1), xytext=(0.2, 5.06e-3), arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2))
    a.text(0.225, 4.5e-2, "×70\nsymmetry\nbroken", fontsize=8, color="0.25", va="center")
    a.set_ylabel(r"detected-signal Fisher  $\max_u F_{\delta\delta}$"); a.set_ylim(2e-7, 1)
    a.legend(loc="lower right", fontsize=8, framealpha=0.95)
    a.set_title("(A)  Breaking permutation symmetry lifts the dark sector — only the protected arm moves", fontsize=9.5, loc="left")
    b.semilogx(djs, rs, "o-", color=Csym, lw=2.0, ms=5, label="symmetric (collective control + coil)  —  selection rule")
    b.semilogx(djb, rb, "s-", color=Cbrk, lw=2.2, ms=6, label="broken symmetry (selective pulses + individual detection)")
    b.set_yscale("log"); b.axhline(1.0, color="k", lw=1.3, ls="--")
    b.axvspan(lo, hi, color="#fdd0a2", alpha=0.45, zorder=0)
    b.text(0.2, 12, "quantum-advantage window", ha="center", fontsize=8.5, color="#a63603")
    b.plot([lo, hi], [1, 1], "v", color="#a63603", ms=7, clip_on=False)
    b.text(lo, 0.62, f"crosses 1\n$\\delta/J\\!\\approx\\!{lo:.2f}$", fontsize=7.5, ha="center", color="#a63603")
    b.text(hi, 0.62, f"crosses 1\n$\\delta/J\\!\\approx\\!{hi:.2f}$", fontsize=7.5, ha="center", color="#a63603")
    b.annotate("peak  7.6×", xy=(0.2, 7.565), xytext=(0.045, 3.0), fontsize=9, arrowprops=dict(arrowstyle="->", lw=1, color="k"))
    b.text(0.011, 0.018, "singlet protected\nbut dark  (selection rule)", fontsize=7.5, color=Csym)
    b.text(3.1, 0.115, "protection gone\n($T_{\\mathrm{LLS}}\\!\\to\\!T_2$):\ncurves re-converge", fontsize=7.5, ha="center", color="0.3")
    b.set_ylim(2e-4, 20); b.set_xlim(1e-2, 1e1)
    b.set_ylabel(r"advantage  $F_{\mathrm{protected}}/F_{\mathrm{baseline}}$"); b.set_xlabel(r"$\delta/J$   (chemical-shift splitting / coupling)")
    b.legend(loc="lower center", fontsize=8, framealpha=0.95)
    b.set_title("(B)  The theorem and its loophole: where the ratio finally crosses unity", fontsize=9.5, loc="left")
    for ax in (a, b): ax.grid(True, which="both", alpha=0.18)
    fig.suptitle("Collective darkness (selection rule) vs broken-symmetry access (loophole)\n"
                 r"$J=10$ Hz, $T_2=200$ ms, $T_{\mathrm{LLS}}^{\mathrm{bare}}=5$ s, matched $T_1$", fontsize=10.5, y=1.0)
    fig.savefig("loophole.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"  saved loophole.png  (broken peak {rb.max():.2f} at d/J={djb[np.argmax(rb)]}, window ~[{lo:.2f}, {hi:.2f}])")

def task_figures():
    print("=== regenerating figures ===")
    fig_mechanism(); fig_phase(); fig_loophole()

# ======================================================================
# 7.  CLI
# ======================================================================
TASKS = {"verify": task_verify, "nogo": task_nogo, "tension": task_tension,
         "phase": task_phase, "loophole": task_loophole, "figures": task_figures}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "all":
        task_verify(); task_nogo(); task_tension(); task_phase(); task_loophole(); task_figures()
    elif cmd in TASKS:
        TASKS[cmd]()
    else:
        print(__doc__)
        print("Unknown command:", cmd, "\nChoose one of:", ", ".join(list(TASKS) + ["all"]))
