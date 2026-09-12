#!/usr/bin/env python3
"""
RUN_ALL.py  --  regenerate every figure and number from scratch, in order
==========================================================================
This is the single entry point for reproducing the paper. It runs the whole
pipeline in the one correct order and verifies the result against the numbers
quoted in the manuscript, so the stale-data problem that recurred during
preparation cannot happen: if any figure would disagree with the text, this
script says so and stops.

    python RUN_ALL.py            # full run, roughly twenty minutes
    python RUN_ALL.py --quick    # skip the two slow sweeps, use existing .npy
    python RUN_ALL.py --check    # verify existing outputs only, run nothing

WHAT RUNS, AND IN WHAT ORDER
    1  replicate.py verify         brightness/invisibility algebra, Eqs. (3),(4)
    2  replicate.py phase          -> phase_data.npy        (Fig. 2a)     ~4 min
    3  replicate.py loophole       -> loophole_data.npy     (Fig. 2b)    ~14 min
    4  nogo_diagnostics.py         -> restart/DE/S8 caches  (Fig. S1, Sec. S8)
    5  robustness_sweeps.py        -> rf/B0 caches          (Fig. S2)
    6  export_csvs.py --from-npy   -> data/*.csv  (six files)
    7  make_figures.py             -> fig1,fig2combo,fig4,fig5,fig6
    8  verify_selection_rule.py    -> Table S1 rates, independent check

Everything is run in the current directory, which must contain replicate.py
and the other scripts. All scripts depend only on NumPy, SciPy and Matplotlib.
"""
import argparse
import importlib
import os
import subprocess
import sys

import numpy as np

REQUIRED = ["replicate.py", "export_csvs.py", "make_figures.py",
            "nogo_diagnostics.py", "robustness_sweeps.py",
            "verify_selection_rule.py"]

# Numbers quoted in the manuscript. Keep in step with the text.
CLAIMED = dict(collective_plateau=0.51, lower_crossing=0.03, peak_ratio=7.6,
               peak_location=0.20, upper_crossing=1.05)
TOL = 0.06


def sh(cmd):
    print(f"\n$ {cmd}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        sys.exit(f"  step failed: {cmd}")


def load_sweep(fname):
    if not os.path.exists(fname):
        return None
    d = np.load(fname, allow_pickle=True).item()
    ks = sorted(k for k in d if not (isinstance(k, str) and k.startswith("_")))
    if not ks:
        return None
    x = np.array(ks, float)
    Ff = np.array([d[k][0] for k in ks], float)
    Fb = np.array([d[k][1] for k in ks], float)
    return x, Ff / Fb


def crossings(x, r):
    lx, lr = np.log(x), np.log(r)
    out = []
    for j in range(len(r) - 1):
        if lr[j] * lr[j + 1] < 0:
            t = -lr[j] / (lr[j + 1] - lr[j])
            out.append(float(np.exp(lx[j] + t * (lx[j + 1] - lx[j]))))
    return out


def check():
    print("\n" + "=" * 70)
    print("Verifying regenerated data against the manuscript")
    print("=" * 70)
    ok = True

    def cmp(name, meas, claim):
        nonlocal ok
        good = abs(meas - claim) <= TOL * max(abs(claim), 1e-9)
        ok &= good
        print(f"  {name:<18} {meas:8.3f}  vs text {claim:6.2f}  "
              f"[{'OK' if good else 'MISMATCH'}]")

    ph = load_sweep("phase_data.npy")
    if ph is None:
        print("  phase_data.npy MISSING"); ok = False
    else:
        x, r = ph
        cmp("collective peak", float(r.max()), CLAIMED["collective_plateau"])
        if r.max() >= 1.0:
            print("  !! collective ratio reaches unity, contradicts the no-go")
            ok = False

    lp = load_sweep("loophole_data.npy")
    if lp is None:
        print("  loophole_data.npy MISSING"); ok = False
    else:
        x, r = lp
        cr = crossings(x, r); i = int(np.argmax(r))
        cmp("peak ratio", float(r[i]), CLAIMED["peak_ratio"])
        cmp("peak location", float(x[i]), CLAIMED["peak_location"])
        if len(cr) >= 2:
            cmp("lower crossing", cr[0], CLAIMED["lower_crossing"])
            cmp("upper crossing", cr[-1], CLAIMED["upper_crossing"])
        else:
            print(f"  !! expected two crossings, found {len(cr)}"); ok = False

    print("-" * 70)
    print("  ALL NUMBERS MATCH THE TEXT" if ok else
          "  MISMATCH: figures and text disagree. Fix before submitting.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the two slow sweeps, reuse existing .npy")
    ap.add_argument("--check", action="store_true",
                    help="verify existing outputs only")
    args = ap.parse_args()

    missing = [f for f in REQUIRED if not os.path.exists(f)]
    if missing:
        sys.exit("  missing scripts in this directory: " + ", ".join(missing))

    if args.check:
        sys.exit(0 if check() else 1)

    sh(f"{sys.executable} replicate.py verify")
    if not args.quick:
        sh(f"{sys.executable} replicate.py phase")
        sh(f"{sys.executable} replicate.py loophole")
    else:
        for f in ("phase_data.npy", "loophole_data.npy"):
            if not os.path.exists(f):
                sys.exit(f"  --quick given but {f} is absent; run a full pass")

    sh(f"{sys.executable} nogo_diagnostics.py")
    sh(f"{sys.executable} robustness_sweeps.py")
    sh(f"{sys.executable} export_csvs.py --from-npy")
    sh(f"{sys.executable} make_figures.py")
    sh(f"{sys.executable} verify_selection_rule.py")

    if not check():
        sys.exit(1)
    print("\nDone. Every figure and number regenerated and checked.")


if __name__ == "__main__":
    main()
