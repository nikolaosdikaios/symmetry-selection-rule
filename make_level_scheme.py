"""Fig. 1: level scheme of the scalar-coupled pair, arranged by magnetic quantum number M."""
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "mathtext.fontset": "cm"})
fig, ax = plt.subplots(figsize=(3.4, 2.9))
ax.set_xlim(0, 4.0); ax.set_ylim(-0.62, 2.35); ax.axis("off")
ax.add_patch(Rectangle((0.08, 0.60), 3.84, 0.56, fc="#dfeaf6", ec="none", zorder=0))
ax.text(3.12, 0.84, "$M=0$ plane:\nimmune to\ncommon-mode $\\Delta F_z$", ha="left", va="center", fontsize=6.3, color="#1f4e79")
L = dict(lw=2.2, color="k", solid_capstyle="butt", zorder=3)
for y, lab in [(2.0, "T_{+}"), (1.0, "T_{0}"), (0.0, "T_{-}")]:
    ax.plot([1.85, 2.65], [y, y], **L); ax.text(2.72, y, f"$|{lab}\\rangle$", va="center")
ax.plot([0.35, 1.05], [0.80, 0.80], **L); ax.text(0.28, 0.80, "$|S\\rangle$", va="center", ha="right")
arrow = lambda p, q, **k: ax.add_patch(FancyArrowPatch(p, q, mutation_scale=7, lw=1.0, zorder=4, **k))
arrow((2.05, 1.07), (2.05, 1.93), arrowstyle="<->", color="C0")
arrow((2.05, 0.07), (2.05, 0.93), arrowstyle="<->", color="C0")
arrow((2.42, 1.07), (2.42, 1.93), arrowstyle="<->", color="0.45", ls=(0, (2, 1.5)), connectionstyle="arc3,rad=-0.35")
arrow((2.42, 0.07), (2.42, 0.93), arrowstyle="<->", color="0.45", ls=(0, (2, 1.5)), connectionstyle="arc3,rad=0.35")
ax.text(3.05, 1.72, "collective rotations,\n$F_+$ readout", color="C0", va="center", fontsize=6.8)
ax.text(3.05, 1.33, "dipolar $T_{2,q}$\n(no action on $|S\\rangle$)", color="0.35", va="center", fontsize=6.8)
arrow((1.10, 0.82), (1.80, 0.98), arrowstyle="<->", color="C3")
ax.text(1.42, 1.02, "$\\pi\\delta$", color="C3", ha="center", va="bottom")
ax.text(0.70, 0.66, "$J$ below $|T_0\\rangle$", fontsize=6.3, ha="center", va="bottom")
arrow((0.80, 0.87), (1.82, 1.95), arrowstyle="->", color="C2", ls=(0, (4, 2)))
ax.text(0.20, 1.62, "$I_{1+}-I_{2+}$:\nselective access", color="C2", va="center", fontsize=6.8)
ax.text(0.70, -0.40, "exchange-odd", ha="center", fontsize=6.8)
ax.text(2.25, -0.40, "exchange-even", ha="center", fontsize=6.8)
for y, m in [(2.0, "+1"), (1.0, "0"), (0.0, "-1")]:
    ax.text(3.97, y if m != "0" else 0.97, f"$M={m}$", ha="right", va="center", fontsize=6.3, color="0.4") if m != "0" else None
fig.tight_layout(pad=0.1); fig.savefig("figs/fig_levels.pdf"); fig.savefig("figs/fig_levels.png", dpi=220)
