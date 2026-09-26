"""Fig. 2: best symmetric Fisher information for a singlet probe of purity p (J = 0), relative to the pure-state QFI (Proposition 1)."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
x = np.logspace(-3.5, np.log10(np.pi / 2 * 0.98), 500)          # x = pi * delta * t
fig, a = plt.subplots(figsize=(3.4, 2.6))
for p, c in [(1.0, "k"), (0.99, "C0"), (0.9, "C1"), (0.5, "C2"), (0.1, "C3")]:
    q = (1 - p) / 4
    exact = (p * np.sin(2 * x)) ** 2 * (1 / (p * np.cos(x) ** 2 + q) + 1 / (p * np.sin(x) ** 2 + q)) / 4
    a.loglog(x / np.pi, exact, c=c, lw=1.2, label=f"$p={p:g}$")
    if p < 1:
        xs2 = (1 - p) / (4 * p)
        a.loglog(x / np.pi, p * x**2 / (x**2 + xs2), "--", c=c, lw=0.7)
        a.axhline(2 * p**2 / (1 + p), c=c, ls=":", lw=0.7)
a.set_xlabel(r"$\delta t$"); a.set_ylabel(r"$F_{\rm sym}/4\pi^2t^2$"); a.set_ylim(1e-6, 2)
a.legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig("figs/fig_crossover.pdf")
