"""Supplement S3: numerical checks of Theorems 1 and 2 in the relaxing pair.
(a) evenness of symmetric signals under delta -> -delta; (b) delta^2 scaling of symmetric protocols versus a finite
limit for a symmetry-breaking protocol (limit B, static offsets); (c) best symmetric Fisher information,
F_Q of the exchange-twirled state, for thermal, pure-singlet and mixed-singlet probes under relaxation.
Writes figs/figS1_checks.pdf."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import model as M, protocols as Pr, static_runs as SR
rng = np.random.default_rng(11); J = 10.0
def samples(Y, L0, meas):
    F = M.aug(L0, M.DT); out = []
    for _ in range(M.NF):
        out.append([m @ Y[:M.D] for m in meas]); Y = F @ Y
    return np.array(out)
def run_coll(p, L0, n1, n2):
    th1, ph1, th2, ph2, d1, te, ts, th3, ph3, d2, th4, ph4 = p
    E = np.linalg.matrix_power(M.aug(L0, te) @ Pr.PI_X @ M.aug(L0, te), 1)
    Y = M.aug(L0, d1) @ (M.pulse_collective(th1, ph1) @ M.RHO0)
    Y = np.linalg.matrix_power(E, n1) @ Y
    Y = M.pulse_collective(th3, ph3) @ (M.aug(L0, ts) @ (M.pulse_collective(th2, ph2) @ Y))
    Y = M.pulse_collective(th4, ph4) @ (M.aug(L0, d2) @ (np.linalg.matrix_power(E, n2) @ Y))
    return samples(Y, L0, M.MEAS_COLL)
def run_sel(p, L0):
    Y = M.aug(L0, p[4]) @ (M.pulse_selective(*p[0:4]) @ M.RHO0)
    Y = M.aug(L0, p[9]) @ (M.pulse_selective(*p[5:9]) @ Y)
    Y = M.aug(L0, p[14]) @ (M.pulse_selective(*p[10:14]) @ Y)
    return samples(Y, L0, M.MEAS_INDIV)
g = SR.G_HOM
pc = np.array([rng.uniform(lo, hi) for lo, hi in Pr.BND_C]); ps = np.array([rng.uniform(lo, hi) for lo, hi in Pr.BND_S])
for name, f in [("symmetric", lambda d: run_coll(pc, M.liouvillian(d, J, g), 2, 2)), ("symmetry-breaking", lambda d: run_sel(ps, M.liouvillian(d, J, g)))]:
    a, b = f(0.7), f(-0.7)
    print(f"(a) {name:18s} max|s(delta) - s(-delta)| / max|s| = {np.abs(a - b).max() / np.abs(a).max():.1e}")
# (b) scaling in limit B
Gam = SR.GAMMA["B"]; deltas = np.logspace(-3, 1, 21)
selB = json.load(open("results/static_sel_B.json"))["points"]
curves = {}
for k in range(4):
    p = np.array([rng.uniform(lo, hi) for lo, hi in Pr.BND_C])
    curves[f"symmetric {k+1}"] = [SR.fisher_collective_static(p, *(lambda L: (L, 2, 2, SR.fid_rows(L, M.MEAS_COLL)))(M.liouvillian(d, J, g)), Gam) for d in deltas]
pbest = np.array(selB["2.0"]["P"]["params"])
curves["symmetry-breaking"] = [SR.fisher_selective_static(pbest, (L := M.liouvillian(d, J, g)), SR.fid_rows(L, M.MEAS_INDIV), Gam) for d in deltas]
for k, v in curves.items():
    v = np.array(v); s = np.polyfit(np.log(deltas[:6]), np.log(v[:6]), 1)[0]
    print(f"(b) {k:18s} log-log slope for delta = 0.001-0.01 Hz: {s:.3f}")
# (c) best symmetric Fisher information of the free-evolving pair after time t
P = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], float); t = 0.05
def Fsym(rho0, d, relax=True):
    Y = M.aug(M.liouvillian(d, J, g if relax else {}), t) @ np.concatenate([M.vec(rho0), np.zeros(M.D)])
    r = Y[:M.D].reshape(4, 4, order="F"); dr = Y[M.D:].reshape(4, 4, order="F")
    s = 0.5 * (r + P @ r @ P); ds = 0.5 * (dr + P @ dr @ P); s = 0.5 * (s + s.conj().T); ds = 0.5 * (ds + ds.conj().T)
    lam, V = np.linalg.eigh(s); X = V.conj().T @ ds @ V
    return sum(2 * abs(X[i, j])**2 / (lam[i] + lam[j]) for i in range(4) for j in range(4) if lam[i] + lam[j] > 1e-15)
eps = 1e-5; dc = np.logspace(-4, 0.5, 40)
probes = {"pure singlet, no relaxation": (M.ket(M.S, M.S), 1.0, False), "pure singlet, relaxing": (M.ket(M.S, M.S), 1.0, True),
          "singlet p = 0.9, relaxing": (0.9*M.ket(M.S, M.S) + 0.1*np.eye(4)/4, 1.0, True),
          "thermal after 90° pulse (×1/ε²)": (np.eye(4)/4 + eps*M.Fx, eps**2, True)}
cvals = {k: [Fsym(r, d, rl) / sc for d in dc] for k, (r, sc, rl) in probes.items()}
for k, v in cvals.items():
    print(f"(c) {k:32s} F_sym at delta = 1e-4, 1e-2, 1 Hz: {v[0]:.2e}, {v[np.argmin(abs(dc-1e-2))]:.2e}, {v[np.argmin(abs(dc-1))]:.2e}")
print(f"    J-limited value 4 sin^2(pi J t)/J^2 at t = {t} s: {4*np.sin(np.pi*J*t)**2/J**2:.3f} Hz^-2")
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
for k, v in curves.items():
    ax[0].loglog(deltas, np.array(v) / v[-1], "-" if "sym" in k and "breaking" not in k else "k-", lw=1.8 if "breaking" in k else 1, label={"symmetric 1": "symmetric (4 random protocols)", "symmetry-breaking": "symmetry-breaking"}.get(k))
ax[0].loglog(deltas, (deltas / deltas[-1])**2, ":", c="0.5", label=r"$\propto\delta^2$")
ax[0].set_xlabel(r"$\delta$ (Hz)"); ax[0].set_ylabel(r"$F(\delta)/F(10\,{\rm Hz})$"); ax[0].legend(fontsize=7, frameon=False)
ax[0].set_title("(a) fixed protocols, limit B", fontsize=9)
for k, v in cvals.items():
    ax[1].loglog(dc, v, label=k)
ax[1].axhline(4*np.sin(np.pi*J*t)**2/J**2, c="0.5", ls=":", lw=0.8)
ax[1].set_xlabel(r"$\delta$ (Hz)"); ax[1].set_ylabel(r"$F_{\rm sym}$ (Hz$^{-2}$)"); ax[1].legend(fontsize=6.5, frameon=False)
ax[1].set_title(r"(b) best symmetric measurement, $t=0.05$ s", fontsize=9)
fig.tight_layout(); fig.savefig("figs/figS1_checks.pdf"); fig.savefig("figs/figS1_checks.png", dpi=110)
