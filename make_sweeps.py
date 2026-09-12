#!/usr/bin/env python3
"""
make_sweeps.py  --  generate phase_data.npy and loophole_data.npy, then check
                    them against the numbers quoted in the Letter
=============================================================================
These two files are produced by replicate.py itself:

    python replicate.py phase       ->  phase_data.npy      (Fig. 2a)
    python replicate.py loophole    ->  loophole_data.npy   (Fig. 2b)

This wrapper runs both in the CURRENT directory and then verifies the result,
because the numbers the manuscript quotes (collective plateau, window edges,
peak) are exactly the numbers that have repeatedly drifted between the text
and the figures. Running the sweeps without checking them is what allowed an
embedded dataset to reach the figures while the text quoted a live one.

Usage
    python make_sweeps.py              # run both sweeps, then verify
    python make_sweeps.py --check-only # skip the sweeps, verify existing files

Timing: roughly four minutes for the collective sweep and fourteen for the
broken-symmetry sweep, since every grid point runs a full multistart
optimization. replicate.py saves after each grid point, so an interrupted run
leaves usable partial data.

Requires replicate.py in the same directory.
"""
import argparse
import os
import sys

import numpy as np

__version__ = "2026-07-24"

# Values quoted in the Letter. Update these if the text changes.
CLAIMED = {
    "collective plateau": 0.51,
    "lower crossing": 0.03,
    "peak ratio": 7.6,
    "peak location": 0.20,
    "upper crossing": 1.05,
}
TOL = 0.06          # 6 percent, loose enough for optimizer stochasticity


def load_sweep(fname):
    """Return (dj, ratio) from a replicate.py _sweep file, or None."""
    if not os.path.exists(fname):
        return None
    d = np.load(fname, allow_pickle=True).item()
    ks = sorted(k for k in d.keys()
                if not (isinstance(k, str) and k.startswith("_")))
    if not ks:
        return None
    dj = np.array(ks, float)
    Ff = np.array([d[k][0] for k in ks], float)
    Fb = np.array([d[k][1] for k in ks], float)
    return dj, Ff / Fb


def unity_crossings(x, r):
    """Log-log interpolated points where the ratio passes through one."""
    lx, lr = np.log(x), np.log(r)
    out = []
    for j in range(len(r) - 1):
        if lr[j] * lr[j + 1] < 0:
            t = -lr[j] / (lr[j + 1] - lr[j])
            out.append(float(np.exp(lx[j] + t * (lx[j + 1] - lx[j]))))
    return out


def report(name, measured, claimed):
    ok = abs(measured - claimed) <= TOL * max(abs(claimed), 1e-9)
    flag = "OK " if ok else "MISMATCH"
    print(f"    {name:<20} measured {measured:8.3f}   Letter says "
          f"{claimed:6.2f}   [{flag}]")
    return ok


def verify():
    print("\n" + "=" * 74)
    print("Verifying the sweeps against the numbers quoted in the Letter")
    print("=" * 74)
    all_ok = True

    ph = load_sweep("phase_data.npy")
    if ph is None:
        print("  phase_data.npy      NOT FOUND")
        all_ok = False
    else:
        dj, r = ph
        print(f"  phase_data.npy      {len(dj)} grid points")
        all_ok &= report("collective plateau", float(r.max()),
                         CLAIMED["collective plateau"])
        if r.max() >= 1.0:
            print("    !! the collective ratio reaches unity, which would "
                  "contradict the no-go claim")
            all_ok = False

    lp = load_sweep("loophole_data.npy")
    if lp is None:
        print("  loophole_data.npy   NOT FOUND")
        all_ok = False
    else:
        dj, r = lp
        cr = unity_crossings(dj, r)
        ipk = int(np.argmax(r))
        print(f"  loophole_data.npy   {len(dj)} grid points")
        all_ok &= report("peak ratio", float(r[ipk]), CLAIMED["peak ratio"])
        all_ok &= report("peak location", float(dj[ipk]),
                         CLAIMED["peak location"])
        if len(cr) >= 2:
            all_ok &= report("lower crossing", cr[0], CLAIMED["lower crossing"])
            all_ok &= report("upper crossing", cr[-1], CLAIMED["upper crossing"])
        else:
            print(f"    !! expected two unity crossings, found {len(cr)}")
            all_ok = False

    print("-" * 74)
    if all_ok:
        print("  All quoted numbers reproduced. Now run:")
        print("      python export_csvs.py --from-npy")
        print("      python make_figures.py")
    else:
        print("  MISMATCH. Either the manuscript text or CLAIMED above is out")
        print("  of date. Do not regenerate figures until they agree, or the")
        print("  figures and the text will disagree in the submitted PDF.")
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="verify existing .npy files without rerunning")
    args = ap.parse_args()

    print(f"make_sweeps.py  version {__version__}")
    print(f"  working directory: {os.path.abspath('.')}")

    if not args.check_only:
        if not os.path.exists("replicate.py"):
            sys.exit("  replicate.py not found in this directory. The sweeps "
                     "are produced by replicate.py, so run this beside it.")
        import replicate as R
        print("\n  [1/2] collective sweep, roughly four minutes ...")
        R.task_phase()
        print("\n  [2/2] broken-symmetry sweep, roughly fourteen minutes ...")
        R.task_loophole()

    verify()


if __name__ == "__main__":
    main()
