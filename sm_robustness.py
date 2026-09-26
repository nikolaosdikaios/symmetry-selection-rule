"""Supplement S6: robustness of the symmetry-breaking gain in limit B.
(a) uniform miscalibration of every flip angle by a factor 1 + e, protocols not re-optimized;
(b) RF inhomogeneity: signals averaged over a Gaussian spread of flip-angle factors across the sample;
(c) dependence on T2*: protected pair and reference re-optimized at T2* = 0.1 and 0.4 s.
Writes results/sm_robustness.json and figs/figS4_robustness.pdf.
With --plot, only redraws the figure from an existing results/sm_robustness.json."""
import json, numpy as np, matplotlib
import matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import model as M, protocols as Pr, static_runs as SR
J = 10.0; B = json.load(open("results/static_sel_B.json"))["points"]; GamB = SR.GAMMA["B"]
FLIP = [0, 2, 5, 7, 10, 12]

def dsignals(p, L0, rows, Gam, scale=1.0):
    q = np.array(p, float).copy(); q[FLIP] *= scale
    U1, U2, U3 = M.pulse_selective(*q[0:4]), M.pulse_selective(*q[5:9]), M.pulse_selective(*q[10:14])
    d1, ts, d2 = q[4], q[9], q[14]
    Y = M.aug(L0, d1) @ (U1 @ M.RHO0); Fs = M.aug(L0, ts); cols, phi = [], []
    for p1 in range(-2, 3):
        v = SR.MASK[p1] * Y
        if not v.any(): continue
        v = Fs @ (U2 @ v)
        for p2 in range(-2, 3):
            w = SR.MASK[p2] * v
            if not w.any(): continue
            cols.append(SR.MASK[SR.QDET] * (U3 @ w)); phi.append(p1 * d1 + p2 * ts + SR.QDET * d2)
    V = M.aug(L0, d2) @ np.array(cols).T; phi = np.array(phi)
    W = np.exp(-Gam * np.abs(phi[None, :] + SR.QDET * SR.TK[:, None]))
    return np.array([(W * (S @ V[:M.D] + R @ V[M.D:])).sum(axis=1) for R, S in rows])
fisher = lambda ds: 2 * np.sum(np.abs(ds) ** 2)
def arm(delta, name, Gam=GamB):
    Jj, g = SR.ARMS[name]; L0 = M.liouvillian(delta, Jj, g); return L0, SR.fid_rows(L0, M.MEAS_INDIV), Gam

import sys
if "--plot" in sys.argv:
    out = json.load(open("results/sm_robustness.json"))
else:
    pR = B["5.0"]["R"]["params"]; out = {"miscal": {}, "rf": {}, "t2star": {}}
    Lr = arm(5.0, "R"); assert abs(fisher(dsignals(pR, *Lr)) - B["5.0"]["R"]["F"]) / B["5.0"]["R"]["F"] < 1e-9
    eps = np.linspace(-0.15, 0.15, 13); xg, wg = np.polynomial.hermite_e.hermegauss(9); wg = wg / wg.sum()
    for d in (2.0, 5.0, 10.0):
        pP = B[str(d)]["P"]["params"]; Lp = arm(d, "P"); Lr = arm(d, "R")
        FP = [fisher(dsignals(pP, *Lp, scale=1 + e)) for e in eps]; FR = [fisher(dsignals(pR, *Lr, scale=1 + e)) for e in eps]
        out["miscal"][str(d)] = {"eps": list(eps), "FP": FP, "FR": FR}
        rf = {}
        for sg in (0.0, 0.05, 0.1, 0.2):
            dsP = sum(w * dsignals(pP, *Lp, scale=1 + sg * x) for x, w in zip(xg, wg))
            dsR = sum(w * dsignals(pR, *Lr, scale=1 + sg * x) for x, w in zip(xg, wg))
            rf[str(sg)] = fisher(dsP) / fisher(dsR)
        out["rf"][str(d)] = rf
        g0 = FP[6] / FR[6]
        print(f"delta/J = {d/J}: gain {g0:.2f}; at e = -15%, +15%: x{FP[0]/FR[0]/g0:.3f}, x{FP[-1]/FR[-1]/g0:.3f} "
              f"(protected alone x{FP[0]/FP[6]:.3f}, x{FP[-1]/FP[6]:.3f}); RF spread 5/10/20%: " + ", ".join(f"x{rf[k]/g0:.3f}" for k in ("0.05", "0.1", "0.2")))
    for T2s in (0.1, 0.4):
        Gam = 1 / T2s - 1 / SR.T2H; res = {}
        Lr = arm(5.0, "R", Gam); fR = lambda q: SR.fisher_selective_static(q, *Lr[:2], Gam)
        res["R"] = Pr.optimise(fR, Pr.BND_S, Pr.STORE_S, 4, 31, warm=pR)[0]
        for d in (2.0, 10.0):
            Lp = arm(d, "P", Gam); fP = lambda q: SR.fisher_selective_static(q, *Lp[:2], Gam)
            res[str(d)] = Pr.optimise(fP, Pr.BND_S, Pr.STORE_S, 4, 37, warm=B[str(d)]["P"]["params"])[0]
        out["t2star"][str(T2s)] = res
        print(f"T2* = {T2s} s: gain at delta/J = 0.2: {res['2.0']/res['R']:.2f}, at delta/J = 1: {res['10.0']/res['R']:.2f}")
    for d in ("2.0", "10.0"):
        print(f"T2* = 0.2 s: gain at delta/J = {float(d)/J}: {B[d]['P']['F'] / B['0.5']['R']['F']:.2f}")
    json.dump(out, open("results/sm_robustness.json", "w"), indent=1)
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
for d, c in (("2.0", "C0"), ("5.0", "C1"), ("10.0", "C3")):
    m = out["miscal"][d]; gain = np.array(m["FP"]) / np.array(m["FR"])
    ax[0].plot(100 * np.array(m["eps"]), gain / gain[6], "-o", c=c, ms=3, label=f"$\\delta/J={float(d)/J:g}$")
ax[0].set_xlabel("flip-angle error (%)"); ax[0].set_ylabel("gain / gain without error"); ax[0].legend(fontsize=7, frameon=False)
ax[0].set_title("(a) uniform miscalibration", fontsize=9)
T2 = [0.1, 0.2, 0.4]
for d, c in (("2.0", "C0"), ("10.0", "C3")):
    gains = [out["t2star"]["0.1"][d] / out["t2star"]["0.1"]["R"], B[d]["P"]["F"] / B["0.5"]["R"]["F"], out["t2star"]["0.4"][d] / out["t2star"]["0.4"]["R"]]
    ax[1].loglog(T2, gains, "-o", c=c, ms=4, label=f"$\\delta/J={float(d)/J:g}$")
ax[1].loglog(T2, [70 * (0.2 / t) ** 2 for t in T2], ":", c="0.5", label=r"$\propto (T_2^\ast)^{-2}$")
ax[1].set_xticks(T2); ax[1].set_xticklabels(["0.1", "0.2", "0.4"]); ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[1].set_xlabel(r"$T_2^\ast$ (s)"); ax[1].set_ylabel("gain over reference"); ax[1].legend(fontsize=7, frameon=False)
ax[1].set_title(r"(b) dependence on $T_2^\ast$ (re-optimized)", fontsize=9)
fig.tight_layout(); fig.savefig("figs/figS4_robustness.pdf"); fig.savefig("figs/figS4_robustness.png", dpi=110)
