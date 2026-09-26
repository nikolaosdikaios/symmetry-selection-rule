"""Fig. 3 of the manuscript: lifetimes of the slowest mode at the zero-quantum frequency (T_c^eff) and of the
singlet-order mode (T_S^eff) versus delta/J, from the eigenvalues of the Liouvillian with the homogeneous
relaxation of limits A and B (T1 = T2,hom = 1.2 s, T_S = 5 s). Writes results/lifetimes_hom.json."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import model as M

J = 10.0
G = M.solve_environment(1.2, 1.2, 5.0)

SO = M.vec(M.ket(M.S, M.S) - 0.25 * np.eye(4)); SO /= np.linalg.norm(SO)

def zq_lifetime(delta):
    """Lifetime of the slowest mode oscillating at the zero-quantum frequency."""
    w = np.linalg.eigvals(M.liouvillian(delta, J, G))
    Om = M.TWOPI * np.sqrt(J**2 + delta**2)
    return -1 / max(l.real for l in w if abs(abs(l.imag) - Om) < 0.05 * Om)

def real_modes(delta):
    w, V = np.linalg.eig(M.liouvillian(delta, J, G))
    keep = [i for i in range(M.D) if abs(w[i].imag) < 1e-6 and w[i].real < -1e-9]
    return w[keep].real, V[:, keep] / np.linalg.norm(V[:, keep], axis=0)

# The singlet-order mode is followed continuously from delta/J = 0.01, by overlap with the eigenvector at
# the previous grid point. Picking the mode of largest overlap with singlet order at each delta instead
# jumps to a different mode near delta/J = 3, where singlet order is shared between two modes.
x = np.logspace(-2, np.log10(5), 400)
w, V = real_modes(x[0] * J); v_prev = V[:, np.argmax(abs(SO.conj() @ V))]
TS = []
for v in x:
    w, V = real_modes(v * J); k = np.argmax(abs(v_prev.conj() @ V)); v_prev = V[:, k]; TS.append(-1 / w[k])
life = np.column_stack([[zq_lifetime(v * J) for v in x], TS])
json.dump({"dj": list(x), "Tc": list(life[:, 0]), "TS": list(life[:, 1])}, open("results/lifetimes_hom.json", "w"), indent=1)
for lab, col in (("T_c^eff", 0), ("T_S^eff", 1)):
    print(lab, "at delta/J = 0.01, 0.2, 1, 5:", [round(float(np.interp(v, x, life[:, col])), 2) for v in (0.01, 0.2, 1, 5)])
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, a = plt.subplots(figsize=(3.4, 2.5))
a.loglog(x, life[:, 0], c="C3", label=r"$T_c^{\rm eff}$ (zero-quantum coherence)")
a.loglog(x, life[:, 1], c="C0", ls="-.", label=r"$T_S^{\rm eff}$ (singlet order)")
a.axhline(0.2, c="k", lw=0.6); a.text(0.011, 0.215, r"$T_2^\ast$", fontsize=8)
a.axhline(1.2, c="0.5", lw=0.6, ls=":"); a.text(0.011, 1.3, r"$T_1=T_{2,\rm hom}$", fontsize=8)
a.set_xlabel(r"$\delta/J$"); a.set_ylabel("lifetime (s)"); a.set_ylim(0.1, 30)
a.legend(fontsize=7, frameon=False, loc="upper right"); fig.tight_layout(); fig.savefig("figs/fig_lifetimes_ms.pdf")
