#!/usr/bin/env python3
"""
export_csvs.py  --  bridge from replicate.py results to make_figures.py inputs
==============================================================================
replicate.py saves .npy dictionaries (and embeds converged paper values);
make_figures.py reads six CSVs from ./data/.  This script closes that gap
WITHOUT editing replicate.py.  Drop it in the same folder and run:

    python3 export_csvs.py                 # use embedded paper values (instant)
    python3 export_csvs.py --from-npy      # prefer phase_data.npy/loophole_data.npy
                                           #   if present, else fall back to embedded

It writes whatever it legitimately can and prints an honest status line for
each of the six targets.  It never fabricates data: the two robustness curves
and the two convergence-diagnostic curves are produced ONLY if the underlying
numbers exist (a saved .npy, or a real run), and are otherwise reported as
MISSING with the exact reason.

Target CSVs and their columns (fixed by make_figures.py):
    data/collective_sweep.csv    dJ, ratio
    data/broken_sweep.csv        dJ, ratio
    data/restart_optima.csv      F
    data/de_trace.csv            eval, F
    data/rf_error.csv            err_pct, ratio
    data/b0_inhomogeneity.csv    sigma, Fprot, Fbase
"""
import os
import sys
import csv
import argparse
import numpy as np

__version__ = "2026-07-23-fixed"
DATA = "data"
os.makedirs(DATA, exist_ok=True)


def write_csv(name, header, rows):
    path = os.path.join(DATA, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def load_npy_sweep(fname):
    """Return sorted (dj, Ff, Fb) from a replicate.py _sweep .npy, or None."""
    if not os.path.exists(fname):
        return None
    d = np.load(fname, allow_pickle=True).item()
    ks = sorted(k for k in d.keys() if not (isinstance(k, str) and k.startswith("_")))
    if not ks:
        return None
    dj = np.array(ks, float)
    Ff = np.array([d[k][0] for k in ks], float)
    Fb = np.array([d[k][1] for k in ks], float)
    return dj, Ff, Fb


# ---- embedded converged values, copied verbatim from replicate.py ----------
PHASE_EMB = dict(
    dj=[0.01, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.50, 0.80, 1.20, 2.00, 3.00, 5.00],
    Ff=[3.81e-7, 3.52e-6, 2.32e-5, 8.31e-5, 2.25e-4, 2.10e-3, 5.06e-3, 1.44e-2,
        1.91e-2, 2.31e-2, 2.67e-2, 2.79e-2, 2.79e-2],
    Fb=[1.14e-3, 9.64e-3, 2.39e-2, 4.71e-2, 6.64e-2, 6.48e-2, 5.96e-2, 5.31e-2,
        5.20e-2, 5.33e-2, 5.39e-2, 5.56e-2, 5.65e-2])
LOOP_EMB = dict(
    dj=[0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.50, 0.80, 1.20, 2.00, 3.00, 5.00],
    Ff=[2.94e-3, 3.06e-2, 4.99e-2, 1.053e-1, 2.087e-1, 3.351e-1, 4.039e-1, 3.688e-1,
        1.675e-1, 6.662e-2, 4.057e-2, 2.045e-2, 2.297e-2, 2.452e-2],
    Fb=[5.06e-2, 5.34e-2, 5.34e-2, 5.34e-2, 5.34e-2, 5.34e-2, 5.34e-2, 5.34e-2,
        5.34e-2, 5.34e-2, 5.37e-2, 5.38e-2, 5.23e-2, 5.20e-2])


def emit_sweep(target, npy_name, emb, prefer_npy):
    src = None
    if prefer_npy:
        got = load_npy_sweep(npy_name)
        if got is not None:
            dj, Ff, Fb = got
            src = f"{npy_name} (live run)"
    if src is None:
        dj = np.array(emb["dj"], float)
        Ff = np.array(emb["Ff"], float)
        Fb = np.array(emb["Fb"], float)
        src = "embedded paper values"
    ratio = Ff / Fb
    rows = list(zip(dj, ratio))
    write_csv(target, ["dJ", "ratio"], rows)
    return f"WROTE  {target:28s} <- {src}  ({len(dj)} points, peak {ratio.max():.3f})"


def try_nogo_diagnostics():
    """restart_optima.csv and de_trace.csv require the per-restart optima and
    the differential-evolution best-so-far trace. replicate.py's task_nogo
    prints but does not save these, so they exist only if a patched run wrote
    them. Look for optional caches; otherwise report MISSING with the fix."""
    msgs = []
    ro = "restart_optima_cache.npy"
    if os.path.exists(ro):
        vals = np.load(ro).ravel()
        write_csv("restart_optima.csv", ["F"], [(v,) for v in vals])
        msgs.append(f"WROTE  {'restart_optima.csv':28s} <- {ro}  ({vals.size} restarts)")
    else:
        msgs.append("MISSING restart_optima.csv        "
                    "-> task_nogo prints optima but saves nothing; see NOTE 1")
    dt = "de_trace_cache.npy"
    if os.path.exists(dt):
        tr = np.load(dt)
        write_csv("de_trace.csv", ["eval", "F"],
                  [(int(e), f) for e, f in tr])
        msgs.append(f"WROTE  {'de_trace.csv':28s} <- {dt}  ({len(tr)} gens)")
    else:
        msgs.append("MISSING de_trace.csv              "
                    "-> differential-evolution trace not persisted; see NOTE 1")
    return msgs


def report_robustness():
    """rf_error.csv and b0_inhomogeneity.csv have NO source in replicate.py:
    there is no robustness task in the file at all."""
    out = []
    for tgt, note in [("rf_error.csv", "rf-amplitude sweep"),
                      ("b0_inhomogeneity.csv", "B0-inhomogeneity sweep")]:
        cache = tgt.replace(".csv", "_cache.npy")
        if os.path.exists(cache):
            arr = np.atleast_2d(np.load(cache))
            if tgt == "rf_error.csv":
                write_csv(tgt, ["err_pct", "ratio"], arr.tolist())
            else:
                write_csv(tgt, ["sigma", "Fprot", "Fbase"], arr.tolist())
            out.append(f"WROTE  {tgt:28s} <- {cache}")
        else:
            out.append(f"MISSING {tgt:27s} -> no {note} exists in replicate.py; see NOTE 2")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-npy", action="store_true",
                    help="prefer live phase_data.npy / loophole_data.npy over embedded values")
    args = ap.parse_args()

    print("=" * 74)
    print(f"export_csvs.py  version {__version__}")
    print("Exporting CSVs for make_figures.py  ->  ./data/")
    print("=" * 74)
    lines = []
    lines.append(emit_sweep("collective_sweep.csv", "phase_data.npy", PHASE_EMB, args.from_npy))
    lines.append(emit_sweep("broken_sweep.csv", "loophole_data.npy", LOOP_EMB, args.from_npy))
    lines += try_nogo_diagnostics()
    lines += report_robustness()
    for ln in lines:
        print("  " + ln)

    n_ok = sum(ln.startswith("WROTE") for ln in lines)
    print("-" * 74)
    print(f"  {n_ok}/6 CSVs written.")
    print("""
NOTE 1  restart_optima.csv and de_trace.csv (SM Fig. S1)
  task_nogo() in replicate.py computes both but only prints them. To persist,
  add these two lines inside task_nogo (optima list already exists as the loop
  over 24 restarts; collect -r.fun into a list called 'optima'):

      np.save("restart_optima_cache.npy", np.array(optima))
      # and, if you switch the baseline optimum to differential_evolution,
      # capture its .population_energies history into de_trace_cache.npy

  Then rerun:  python replicate.py nogo   &&   python export_csvs.py

NOTE 2  rf_error.csv and b0_inhomogeneity.csv (SM Fig. S2)
  There is NO robustness task in replicate.py. The SM robustness figure
  describes an rf-miscalibration sweep and a B0-inhomogeneity average that are
  not implemented in this file. Either implement them (a new task that scales
  all flip angles by (1+err) and re-evaluates the ratio at delta/J=0.2, and
  one that averages the detected signal over a Gaussian offset distribution),
  or mark SM Fig. S2 as illustrative. Until then make_figures.py will keep its
  AWAITING DATA watermark on fig5.pdf, which is the correct, honest behavior.

Next:  python make_figures.py     (fig2combo panels now populate; fig4/fig5
       stay watermarked until Notes 1-2 are addressed)
""")


if __name__ == "__main__":
    main()
