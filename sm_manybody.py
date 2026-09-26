"""Supplement S7: many-body singlet states. (a) quantum Fisher information of N/2 singlet pairs under a linear
gradient, for adjacent and spanning pairings; (b) Fisher information of two symmetric projective measurements
(onto the probe, and onto the total-spin-zero sector) for N = 4, with pure and partially mixed probes;
(c) collective first moments vanish identically. Writes figs/figS5_manybody.pdf."""
import numpy as np, matplotlib
from scipy.linalg import expm
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.ticker
sx = np.array([[0, 1], [1, 0]]) / 2; sy = np.array([[0, -1j], [1j, 0]]) / 2; sz = np.diag([0.5, -0.5])
def op(a, k, N): return np.kron(np.kron(np.eye(2**k), a), np.eye(2**(N - k - 1)))
def probe(N, pairs):
    s = np.array([[0, 1], [-1, 0]]) / np.sqrt(2); T = np.ones(())
    for _ in pairs: T = np.multiply.outer(T, s)
    order = [q for pr in pairs for q in pr]
    return np.transpose(T, np.argsort(order)).reshape(-1)
def pairings(N): return {"adjacent": [(2*i, 2*i + 1) for i in range(N // 2)], "spanning": [(i, N - 1 - i) for i in range(N // 2)]}
for N in (4, 6):
    Gz = sum((k + 1) * op(sz, k, N) for k in range(N))
    for name, pr in pairings(N).items():
        psi = probe(N, pr); FQ = 4 * (np.vdot(psi, Gz @ Gz @ psi) - np.vdot(psi, Gz @ psi) ** 2).real
        print(f"N = {N}, {name:8s}: F_Q/t^2 = {FQ:.3f}; sum over pairs (g_a - g_b)^2 = {sum((a - b) ** 2 for a, b in pr)}")
N = 4; d = 2**N; Gz = sum((k + 1) * op(sz, k, N) for k in range(N)); psi = probe(N, pairings(N)["spanning"])
Jx, Jy, Jz = (sum(op(a, k, N) for k in range(N)) for a in (sx, sy, sz))
w, V = np.linalg.eigh(Jx @ Jx + Jy @ Jy + Jz @ Jz); P0 = V[:, w < 1e-9] @ V[:, w < 1e-9].conj().T
FQ = 4 * (np.vdot(psi, Gz @ Gz @ psi) - np.vdot(psi, Gz @ psi) ** 2).real
mom = max(abs(np.vdot(u, A @ u)) for x in np.linspace(0, 3, 31) for u in [expm(-1j * x * Gz) @ psi] for A in (Jx, Jy, Jz))
print(f"N = 4 spanning: total-spin-zero sector dimension {int(round(np.trace(P0).real))}; max |<J_alpha>| along the evolution = {mom:.1e}")
def fi_binary(Pi, p, x, h=1e-7):
    def prob(xx):
        u = expm(-1j * xx * Gz) @ psi
        return (p * np.vdot(u, Pi @ u) + (1 - p) * np.trace(Pi) / d).real
    q = prob(x); dq = (prob(x + h) - prob(x - h)) / (2 * h)
    return dq ** 2 / (q * (1 - q))
xs = np.logspace(-4, -0.3, 30); curves = {}
for lab, Pi, p in [("projection onto the probe, pure", np.outer(psi, psi.conj()), 1.0), ("total-spin-zero projection, pure", P0, 1.0),
                   ("projection onto the probe, p = 0.99", np.outer(psi, psi.conj()), 0.99), ("projection onto the probe, p = 0.9", np.outer(psi, psi.conj()), 0.9)]:
    curves[lab] = [fi_binary(Pi, p, x) / FQ for x in xs]
    print(f"  {lab:38s} F/F_Q at delta t = 1e-4, 1e-2: {curves[lab][0]:.3f}, {curves[lab][np.argmin(abs(xs - 1e-2))]:.3f}")
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
Ns = np.arange(2, 42, 2)
ax[0].loglog(Ns, Ns / 2, "o-", ms=3, label="adjacent pairs"); ax[0].loglog(Ns, Ns * (Ns**2 - 1) / 6, "s-", ms=3, label="spanning pairs")
ax[0].set_xticks([2, 5, 10, 20, 40]); ax[0].set_xticklabels(["2", "5", "10", "20", "40"]); ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].set_xlabel("$N$"); ax[0].set_ylabel(r"$F_Q/t^2$ (linear gradient)"); ax[0].legend(fontsize=7, frameon=False)
ax[0].set_title("(a) singlet-pair probes", fontsize=9)
for lab, v in curves.items():
    ax[1].semilogx(xs, v, label=lab)
ax[1].set_xlabel(r"$\delta t$"); ax[1].set_ylabel(r"$F/F_Q$"); ax[1].set_ylim(0, 1.1); ax[1].legend(fontsize=6.3, frameon=False, loc="center left")
ax[1].set_title("(b) symmetric projective readout, $N=4$", fontsize=9)
fig.tight_layout(); fig.savefig("figs/figS5_manybody.pdf")
