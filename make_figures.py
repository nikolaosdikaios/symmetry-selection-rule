#!/usr/bin/env python3
"""
make_figures.py
===============
Figure production for the Letter and the Supplemental Material.

WHAT THIS SCRIPT GENERATES FULLY, FROM FIRST PRINCIPLES (no external data)
  fig1.pdf       Letter Fig. 1. The level scheme schematic, panels (a),(b).
  fig6.pdf       SM Fig. S3. Many body dark state QFI scalings. The two
                 curves are the exact closed forms F = N/2 (adjacent pairing)
                 and F = N(N^2-1)/6 (spanning pairing). Markers are brute
                 force checks that build the actual 2^N statevector, compute
                 4 Var(G), and verify darkness (F+, F-, Fz all annihilate
                 the state) to machine precision.

WHAT THIS SCRIPT DRAWS ANALYTICALLY PLUS DATA HOOKS
  fig2combo.pdf  Letter Fig. 2. Panel (b) solid curve is the EXACT window
                 law A(u) = u Pc^2 (1+u)/(1+Pc u)^2 with Pc = 25, which is
                 legitimate published content (Letter Eqs. (5)-(6)). The
                 simulation curves load from
                     data/collective_sweep.csv   columns: dJ, ratio
                     data/broken_sweep.csv       columns: dJ, ratio
                 produced by replicate.py. Panels missing their data carry a
                 visible AWAITING DATA watermark so nothing synthetic can be
                 submitted by accident.

SCAFFOLDS THAT REQUIRE replicate.py OUTPUT (watermarked until CSVs exist)
  fig4.pdf       SM Fig. S1.  data/restart_optima.csv   column:  F
                              data/de_trace.csv         columns: eval, F
  fig5.pdf       SM Fig. S2.  data/rf_error.csv         columns: err_pct, ratio
                              data/b0_inhomogeneity.csv columns: sigma, Fprot, Fbase

Only NumPy and Matplotlib are required.  Run:  python3 make_figures.py
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.8})
__version__ = "2026-07-24-derived-window"   # load_csv single-column fix, formal labels, fig6 ticks
DATA = pathlib.Path("data")


def load_csv(name):
    """Load a CSV written by export_csvs.py.

    NOTE. np.atleast_2d on a SINGLE-column file returns shape (1, N), so
    arr[:, 0] silently yields only the first value. restart_optima.csv has one
    column, which previously reduced the Fig. S1(a) histogram to a single
    count. Column count is therefore taken from the header."""
    p = DATA / name
    if not p.exists():
        return None
    with open(p) as fh:
        ncol = len(fh.readline().strip().split(","))
    arr = np.loadtxt(p, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1) if ncol == 1 else arr.reshape(1, -1)
    return arr


def watermark(ax, lines):
    ax.text(0.5, 0.5, "\n".join(lines), transform=ax.transAxes,
            ha="center", va="center", fontsize=9, alpha=0.30,
            rotation=15, zorder=0, color="crimson")


# ============================================================================
# fig1  --  the schematic (fully generated)
# ============================================================================
def draw_panel(ax, broken):
    """Level scheme. Every label carries a white background box so that no text
    ever sits on top of an arrow or a level line."""
    BB = dict(fc="white", ec="none", pad=1.2)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    # ---- singlet (left) ----
    ax.plot([0.7, 2.9], [4.6, 4.6], color="tab:blue", lw=2.8)
    ax.text(0.65, 4.6, r"$|S\rangle$", ha="right", va="center", fontsize=10)
    ax.text(1.8, 5.25, "singlet\ninvisible, protected", ha="center", fontsize=7,
            color="tab:blue", bbox=BB)

    # ---- triplet manifold (right) ----
    ax.add_patch(Rectangle((5.3, 2.9), 4.2, 5.0, fill=True,
                 facecolor="#eaf6ea", edgecolor="seagreen", lw=1.0, zorder=0))
    for y, lab in [(7.1, r"$|T_+\rangle$"), (5.4, r"$|T_0\rangle$"),
                   (3.5, r"$|T_-\rangle$")]:
        ax.plot([5.7, 8.9], [y, y], color="seagreen", lw=2.2, zorder=2)
        ax.text(9.0, y, lab, ha="left", va="center", fontsize=9)
    ax.text(7.3, 8.15, "triplet (bright)", ha="center", fontsize=7.5,
            color="seagreen")

    # ---- collective controls, drawn on the right edge to clear the labels ----
    for y0, y1 in [(5.6, 6.9), (3.7, 5.2)]:
        ax.add_patch(FancyArrowPatch((8.35, y0), (8.35, y1), arrowstyle="<->",
                     mutation_scale=9, color="seagreen", zorder=3))
    ax.text(8.45, 6.25, r"$F_x,F_y,H_c$", fontsize=7.5, color="seagreen",
            va="center", bbox=BB, zorder=4)

    # ---- the symmetry-breaking coupling (red), label placed BELOW the arrow ----
    ax.add_patch(FancyArrowPatch((2.95, 4.66), (5.65, 5.34), arrowstyle="<->",
                 mutation_scale=11, color="crimson", lw=1.8, zorder=3))
    ax.text(4.3, 3.95, r"$H_\delta=\frac{\Delta}{2}(I_{1z}-I_{2z})$",
            ha="center", fontsize=8, color="crimson", bbox=BB, zorder=4)
    ax.text(4.3, 3.25, r"coupling $\propto\delta$, transfer $\propto\delta^2$",
            ha="center", fontsize=7,
            color="crimson", bbox=BB, zorder=4)
    ax.text(4.3, 2.55, r"gap $2\pi\sqrt{J^2+\delta^2}$", ha="center",
            fontsize=7, color="0.35", bbox=BB, zorder=4)

    # ---- coil ----
    ax.add_patch(FancyArrowPatch((7.3, 9.05), (7.3, 8.35), arrowstyle="->",
                 mutation_scale=9, color="0.2"))
    ax.text(7.3, 9.4, r"coil $F_+$ (bright only)", ha="center", fontsize=8)

    # ---- bath ----
    ax.add_patch(Rectangle((0.7, 0.3), 8.8, 0.9, fill=False, hatch="////",
                 edgecolor="0.45"))
    ax.text(5.1, 0.75, "fluctuating dipolar bath", ha="center", va="center",
            fontsize=7.5, color="0.3", bbox=dict(fc="white", ec="none", pad=1))

    # ---- blocked relaxation channel to the singlet ----
    ax.plot([1.55, 1.55], [1.2, 4.45], ls="--", color="0.5", lw=1.2)
    ax.text(1.55, 2.8, r"$\times$", color="crimson", fontsize=15,
            ha="center", va="center", bbox=BB)
    ax.text(0.95, 2.8, r"blocked, $T_S\gg T_2$", rotation=90, ha="center",
            va="center", fontsize=7)

    # ---- fast relaxation channel to the triplet ----
    ax.add_patch(FancyArrowPatch((9.15, 1.2), (9.15, 2.8), arrowstyle="->",
                 mutation_scale=9, color="0.5"))
    ax.text(9.55, 2.0, r"fast, $T_2,T_1$", rotation=90, ha="center",
            va="center", fontsize=7)

    if broken:
        ax.add_patch(FancyArrowPatch((2.95, 4.85), (5.65, 6.95),
                     arrowstyle="<->", mutation_scale=15,
                     color="darkorange", lw=3.4, zorder=3))
        ax.text(2.55, 7.45, r"$I_{1+}-I_{2+}$" + "\nselective, rate $O(1)$",
                fontsize=8.5, color="darkorange", weight="bold", ha="left",
                bbox=dict(fc="white", ec="darkorange", lw=0.9, pad=2.5),
                zorder=5)
    else:
        ax.text(2.6, 7.6, "no symmetry-breaking\noperator available",
                fontsize=7.5, color="0.45", ha="left", style="italic",
                bbox=BB, zorder=4)


def make_fig1():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
    draw_panel(axes[0], broken=False)
    draw_panel(axes[1], broken=True)
    axes[0].set_title("(a)  collective control and readout\n"
                      "symmetry intact, singlet invisible", fontsize=9)
    axes[1].set_title("(b)  individual control and readout\n"
                      "symmetry broken, access restored",
                      fontsize=9, color="darkorange")
    for sp in axes[1].spines.values():
        sp.set_visible(True); sp.set_color("darkorange"); sp.set_linewidth(2.0)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.subplots_adjust(top=0.86, bottom=0.04, left=0.02, right=0.98, wspace=0.10)
    fig.savefig("fig1.pdf"); plt.close(fig)
    return "REAL"


# ============================================================================
# fig6  --  many body dark state QFI (fully generated, brute force checked)
# ============================================================================
def singlet_product_state(N, pairs):
    """Statevector of a product of singlets on the given qubit pairs.
    Bit convention, bit 0 means spin up."""
    psi = np.zeros(2 ** N)
    for b in range(2 ** N):
        amp = 1.0
        for (a, c) in pairs:
            ba, bc = (b >> a) & 1, (b >> c) & 1
            if (ba, bc) == (0, 1):
                amp *= +1 / np.sqrt(2)
            elif (ba, bc) == (1, 0):
                amp *= -1 / np.sqrt(2)
            else:
                amp = 0.0
                break
        psi[b] = amp
    return psi


def brute_force_check(N, pairing):
    if pairing == "adjacent":
        pairs = [(2 * j, 2 * j + 1) for j in range(N // 2)]
    else:
        pairs = [(k, N - 1 - k) for k in range(N // 2)]
    psi = singlet_product_state(N, pairs)
    g = np.arange(1, N + 1)                      # linear gradient g_k = k
    Gdiag = np.array([sum(g[k] * (0.5 if ((b >> k) & 1) == 0 else -0.5)
                          for k in range(N)) for b in range(2 ** N)])
    p = psi ** 2
    var = float(p @ Gdiag ** 2 - (p @ Gdiag) ** 2)
    # darkness, apply F-, F+, Fz
    Fm = np.zeros_like(psi); Fp = np.zeros_like(psi)
    for b in range(2 ** N):
        if psi[b] == 0:
            continue
        for k in range(N):
            if ((b >> k) & 1) == 0:              # up, can lower
                Fm[b | (1 << k)] += psi[b]
            else:                                # down, can raise
                Fp[b & ~(1 << k)] += psi[b]
    Fz = np.array([sum(0.5 if ((b >> k) & 1) == 0 else -0.5
                       for k in range(N)) for b in range(2 ** N)]) * psi
    dark = max(np.linalg.norm(Fm), np.linalg.norm(Fp), np.linalg.norm(Fz))
    return 4 * var, dark


def make_fig6():
    Ns = np.arange(4, 42, 2)
    F_adj = Ns / 2
    F_span = Ns * (Ns ** 2 - 1) / 6
    print("  fig6 brute force checks (F_QFI computed from the 2^N state):")
    bf = {"adjacent": [], "spanning": []}
    for N in (4, 6, 8, 10):
        for pairing in ("adjacent", "spanning"):
            F, dark = brute_force_check(N, pairing)
            ref = N / 2 if pairing == "adjacent" else N * (N ** 2 - 1) / 6
            bf[pairing].append((N, F))
            print(f"    N={N:2d} {pairing:9s}  F={F:9.4f}  formula={ref:9.4f}"
                  f"  |F±,Fz on state|={dark:.2e}")
            assert abs(F - ref) < 1e-9 and dark < 1e-12
    fig, ax = plt.subplots(figsize=(3.7, 3.1))
    ax.loglog(Ns, F_span, color="purple", lw=1.6,
              label=r"spanning pairs $(k,N{+}1{-}k)$,  $F\propto N^3$")
    ax.loglog(Ns, F_adj, color="darkorange", lw=1.6,
              label=r"adjacent pairs $(k,k{+}1)$,  $F\propto N$")
    for pairing, col in [("spanning", "purple"), ("adjacent", "darkorange")]:
        pts = np.array(bf[pairing])
        ax.loglog(pts[:, 0], pts[:, 1], "o", ms=4, mfc="white", mec=col,
                  label=None)
    ax.axhline(2e-3, ls="--", color="0.3", lw=1)
    ax.text(0.5, 0.05, r"collective readout, $F^{C}\!\to\!0$",
            transform=ax.transAxes, ha="center", fontsize=7, color="0.3",
            bbox=dict(fc="white", ec="none", pad=1))
    # log-scale minor ticks collide on this narrow range; show decades only and
    # suppress the minor ticks entirely so no labels can overlap
    from matplotlib.ticker import FixedLocator, NullFormatter, FixedFormatter
    ax.xaxis.set_major_locator(FixedLocator([4, 10, 20, 40]))
    ax.xaxis.set_major_formatter(FixedFormatter(["4", "10", "20", "40"]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_xlim(3.6, 46)
    ax.set_xlabel(r"number of spins  $N$")
    ax.set_ylabel(r"available QFI  $F^{Q}_{\delta\delta}$  (dark sector)")
    ax.set_ylim(1e-3, 3e5)
    ax.legend(fontsize=6.5, loc="upper left", frameon=True,
              framealpha=0.95, edgecolor="none")
    fig.tight_layout(); fig.savefig("fig6.pdf"); plt.close(fig)
    return "REAL (open markers are brute force checks)"


# ============================================================================
# fig2combo  --  window law (real) + simulation data hooks
# ============================================================================
def window_law(x, Pc=25.0):
    u = x ** 2
    return u * Pc ** 2 * (1 + u) / (1 + Pc * u) ** 2


def make_fig2combo():
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(3.5, 5.4))
    x = np.logspace(-2, 1, 400)

    # (a) collective readout ratio, simulation data only
    axa.set_xscale("log"); axa.set_ylim(0, 1.2); axa.set_xlim(1e-2, 10)
    axa.axhline(1.0, ls="--", color="0.3", lw=1)
    axa.axhline(0.50, ls=":", color="seagreen", lw=1.2)
    axa.text(1.3e-2, 0.53, "asymptotic plateau", fontsize=8,
             color="seagreen")
    d = load_csv("collective_sweep.csv")
    status_a = "DATA"
    if d is not None:
        axa.plot(d[:, 0], d[:, 1], "o-", ms=4, color="purple",
                 label="Simulation")
        axa.legend(fontsize=8.5, frameon=False)
    else:
        status_a = "AWAITING DATA"
        watermark(axa, ["AWAITING DATA", "load data/collective_sweep.csv",
                        "from replicate.py"])
    axa.set_ylabel(r"$F^{C}_{\rm prot}/F^{C}_{\rm ref}$")
    axa.set_title("(a)  collective readout", fontsize=9)

    # (b) broken symmetry, exact law plus data hook
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.set_xlim(1e-2, 10); axb.set_ylim(1e-2, 20)
    axb.plot(x, window_law(x), color="k", lw=1.6,
             label="Analytical prediction")
    axb.axhline(1.0, ls="--", color="0.3", lw=1)
    # Window edges and peak are DERIVED from the loaded sweep, never hard-coded,
    # so the markers cannot drift away from the curve they annotate. The values
    # are printed so they can be checked against the manuscript text.
    d_win = load_csv("broken_sweep.csv")
    if d_win is not None:
        xw, rw = d_win[:, 0], d_win[:, 1]
        lxw, lrw = np.log(xw), np.log(rw)
        edges = []
        for j in range(len(rw) - 1):
            if lrw[j] * lrw[j + 1] < 0:
                t = -lrw[j] / (lrw[j + 1] - lrw[j])
                edges.append(float(np.exp(lxw[j] + t * (lxw[j + 1] - lxw[j]))))
        ipk = int(np.argmax(rw)); xpk, rpk = float(xw[ipk]), float(rw[ipk])
        if len(edges) >= 2:
            axb.axvspan(edges[0], edges[-1], color="orange", alpha=0.12)
            axb.plot([edges[0], xpk, edges[-1]], [1.0, rpk, 1.0], "s", ms=7,
                     mfc="white", mec="darkorange", mew=1.6,
                     label="Window edges and peak")
            print(f"    window derived from data: onset {edges[0]:.3f}, "
                  f"peak {rpk:.2f} at {xpk:.2f}, upper edge {edges[-1]:.2f}")
            print("    ^ these must match the numbers quoted in the Letter")
    d = load_csv("broken_sweep.csv")
    status_b = "LAW REAL, DATA HOOK"
    if d is not None:
        axb.plot(d[:, 0], d[:, 1], "s-", ms=4, color="darkorange",
                 label="Simulation")
        status_b = "LAW REAL + DATA"
    else:
        axb.text(0.02, 0.05, "simulation curve loads from\n"
                 "data/broken_sweep.csv", transform=axb.transAxes,
                 fontsize=6.5, color="crimson", alpha=0.7)
    axb.set_xlabel(r"$\delta/J$")
    axb.set_ylabel(r"$F^{C}_{\rm prot}/F^{C}_{\rm ref}$")
    axb.set_title("(b)  individual addressing", fontsize=9)
    axb.legend(fontsize=8.5, frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig("fig2combo.pdf"); plt.close(fig)
    return f"(a) {status_a}   (b) {status_b}"


# ============================================================================
# fig4, fig5  --  scaffolds for replicate.py output
# ============================================================================
def make_fig4():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    axa.set_xlabel(r"restart optimum  $F_{\delta\delta}\times 10^{2}$")
    axa.ticklabel_format(style="plain", axis="x")
    axa.tick_params(axis="x", labelsize=8)
    axa.set_ylabel("count of 100 restarts")
    axa.axvline(5.96, ls="--", color="crimson", lw=1.2)
    axa.text(5.9, 0.95, "global optimum 5.96", transform=axa.get_xaxis_transform(),
             ha="right", fontsize=7, color="crimson")
    d = load_csv("restart_optima.csv"); s = "DATA"
    if d is not None:
        axa.hist(d[:, 0] * 100, bins=np.linspace(1, 6.5, 12), color="steelblue",
                 edgecolor="white")
    else:
        s = "AWAITING DATA"
        watermark(axa, ["AWAITING DATA", "data/restart_optima.csv"])
    axb.set_xlabel("differential evolution generation")
    axb.set_ylabel(r"best so far  $F_{\delta\delta}$")
    axb.axhline(5.97e-2, ls="--", color="crimson", lw=1.2)
    d = load_csv("de_trace.csv")
    if d is not None:
        axb.plot(d[:, 0], d[:, 1], color="steelblue")
    else:
        watermark(axb, ["AWAITING DATA", "data/de_trace.csv"])
    axa.set_title("(a)", fontsize=9); axb.set_title("(b)", fontsize=9)
    fig.tight_layout(); fig.savefig("fig4.pdf"); plt.close(fig)
    return s


def make_fig5():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for xg in (-15, 15):
        axa.axvline(xg, ls=":", color="0.6", lw=0.9)
    axa.text(0.5, 0.04, r"$\pm15\%$ band quoted in the text",
             transform=axa.transAxes, ha="center", fontsize=6.5, color="0.45")
    axa.set_xlabel("rf amplitude error (%)")
    axa.set_ylabel(r"$F^{C}_{\rm prot}/F^{C}_{\rm ref}$")
    axa.axhline(1.0, ls="--", color="0.3", lw=1)
    axa.set_ylim(0, 10)
    d = load_csv("rf_error.csv"); s = "DATA"
    if d is not None:
        axa.plot(d[:, 0], d[:, 1], "o-", ms=3.5, color="darkorange")
    else:
        s = "AWAITING DATA"
        watermark(axa, ["AWAITING DATA", "data/rf_error.csv"])
    axb.set_xlabel(r"$B_0$ inhomogeneity  $\sigma$  (Hz)")
    axb.set_ylabel(r"$F/F(0)$"); axb.set_yscale("log")
    axb.axvline(1.6, ls=":", color="0.4", lw=1.2)
    axb.text(1.65, 0.9, "linewidth 1.6 Hz", transform=axb.get_xaxis_transform(),
             fontsize=7, rotation=90, va="top", color="0.4")
    d = load_csv("b0_inhomogeneity.csv")
    if d is not None:
        axb.plot(d[:, 0], d[:, 1], "o-", ms=3.5, color="darkorange",
                 label="protected, zero quantum")
        axb.plot(d[:, 0], d[:, 2], "^-", ms=3.5, color="seagreen",
                 label="reference, single quantum")
        axb.legend(fontsize=6.5, frameon=False)
    else:
        watermark(axb, ["AWAITING DATA", "data/b0_inhomogeneity.csv"])
    axa.set_title("(a)", fontsize=9); axb.set_title("(b)", fontsize=9)
    fig.tight_layout(); fig.savefig("fig5.pdf"); plt.close(fig)
    return s


REQUIRED_CSVS = ["collective_sweep.csv", "broken_sweep.csv", "restart_optima.csv",
                 "de_trace.csv", "rf_error.csv", "b0_inhomogeneity.csv"]


def data_report():
    """Print exactly which inputs were found, so an AWAITING DATA status can be
    diagnosed without guesswork."""
    print(f"  input folder: {DATA.resolve()}")
    if not DATA.is_dir():
        print("  !! that folder does not exist. Run export_csvs.py from the SAME")
        print("     directory in which this script is run.")
        return
    missing = [n for n in REQUIRED_CSVS if not (DATA / n).exists()]
    for name in REQUIRED_CSVS:
        print(f"     {'found  ' if (DATA / name).exists() else 'MISSING'} {name}")
    if missing:
        print("  !! run:  python export_csvs.py --from-npy")


if __name__ == "__main__":
    print(f"make_figures.py  version {__version__}")
    data_report()
    print("Generating figures")
    report = {
        "fig1.pdf     ": make_fig1(),
        "fig6.pdf     ": make_fig6(),
        "fig2combo.pdf": make_fig2combo(),
        "fig4.pdf     ": make_fig4(),
        "fig5.pdf     ": make_fig5(),
    }
    print("\nStatus report")
    for k, v in report.items():
        print(f"  {k} {v}")
    print("\nDrop replicate.py CSV exports into ./data/ and rerun to replace"
          " every scaffold with real curves.")
