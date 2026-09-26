"""Supplement S4: distribution of multistart optima relative to the best value found, pooled over delta.
Reads results/static_sel_A.json and results/static_sel_B.json; writes figs/figS2_convergence.pdf."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
for a, lim in zip(ax, "BA"):
    d = json.load(open(f"results/static_sel_{lim}.json"))["points"]
    for arm, c in (("P", "C3"), ("R", "0.4")):
        rel = []
        for rec in d.values():
            if arm in rec:
                vals = rec[arm].get("restarts", []) + rec[arm].get("refined_restarts", [])
                rel += [v / rec[arm]["F"] for v in vals]
        rel = np.clip(np.array(rel), 0, 1)
        print(f"limit {lim}, {'protected' if arm == 'P' else 'reference'}: {len(rel)} starts; within 1%: {np.mean(rel > 0.99):.0%}, "
              f"within 5%: {np.mean(rel > 0.95):.0%}, within 20%: {np.mean(rel > 0.8):.0%}, median {np.median(rel):.2f}")
        a.hist(rel, bins=np.linspace(0, 1, 21), color=c, alpha=0.6, label="protected" if arm == "P" else "reference")
    a.set_xlabel(r"$F_{\rm start}/F_{\rm best}$"); a.set_title(f"limit {lim}", fontsize=9)
ax[0].set_ylabel("starts"); ax[0].legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("figs/figS2_convergence.pdf")
