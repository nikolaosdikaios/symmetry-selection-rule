#!/usr/bin/env python3
"""
run_all.py -- reproduce every number and figure of the paper and of its Supplemental Material.

Requirements: Python 3.10 or later with NumPy, SciPy and Matplotlib (tested with Python 3.11 and 3.12,
NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10). Put all the .py files in one folder and run there:

    python run_all.py                 everything, in the order used for the paper (2 to 4 h on one core)
    python run_all.py --parallel      the three independent optimization chains at once (2 h or more)
    python run_all.py --main          main text only (optimizations, checks, tables, Figs. 1-4)
    python run_all.py --supplement    Supplement only (needs the results of the main computation)
    python run_all.py --figures       figures, tables and checks from existing results (minutes)
    python run_all.py --check         compare results/*.json with the values quoted in the paper
    python run_all.py --dry-run       list the steps without running them
    python run_all.py --copy-figures DIR   also copy the figure PDFs to DIR (e.g. the LaTeX folder)

Outputs: results/*.json (every optimum, with its control parameters, restart values and settings),
figs/*.pdf (all figures), and a log of each step on screen.

Why the order matters. The controls are optimized by multistart L-BFGS-B. Each refinement pass
starts from fresh random controls and from the optima already stored for the other values of
delta, so the passes below must run in this order, with these seeds (SEED) and numbers of cold
starts (NCOLD), to reproduce the stored optima exactly. Other library versions or processors can
change individual optima within the stated convergence (about 2% for the symmetry-breaking
values, up to about 10% for the symmetric-class values, which are lower bounds).

Where each result comes from
  main text
    Fig. 1                      make_level_scheme.py
    Fig. 2, Proposition 1       make_crossover_figure.py, verify_crossover.py
    Fig. 3, T_c = 2.54 s        make_lifetimes_figure.py
    Tables I, II, Fig. 4        static_runs.py (optimization), make_static_figures.py (tables, figure)
    Table III, fitted rates     print_model_tables.py
    Table IV (Redfield)         redfield.py
    Table V (Markovian)         run_sweeps.py (optimization), make_static_figures.py (printout)
    Appendix C checks           validate_static.py
  Supplemental Material
    S1  sm_asymmetry.py         S2  sm_rates.py             S3  sm_selection_checks.py (Fig. S1)
    S4  sm_convergence.py (Fig. S2)                       S5  sm_gain_estimate.py (Table S2, Fig. S3)
    S6  sm_robustness.py (Fig. S4)                        S7  sm_manybody.py (Fig. S5)
  shared modules: model.py (spin model, relaxation, Fisher information), protocols.py (control
  ansatze and optimizer).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

PY = sys.executable

# ---------------------------------------------------------------------------------------------
# The optimization runs, in the order used for the paper. Each entry: (script, arguments, env).
# delta values are in Hz, with J = 10 Hz (delta/J = delta/10).
# ---------------------------------------------------------------------------------------------
MARKOVIAN = [  # Appendix D comparison: inhomogeneity as Markovian common-mode dephasing
    ("run_sweeps.py", "selective_cm 0.2 0.5 1 2 3", {}),
    ("run_sweeps.py", "selective_cm 5 10 20 50", {}),
    ("run_sweeps.py", "refine selective_cm 10 20 50", {}),
    ("run_sweeps.py", "refine selective_cm 0.2 0.5 1 2 3 5", {}),
    ("run_sweeps.py", "refine selective_cm 0.2 0.5 1 2 3 5", {"SEED": "1000"}),
    ("run_sweeps.py", "refine selective_cm 10 20 50", {"SEED": "1000"}),
    ("run_sweeps.py", "collective_P 0.2 0.5 1 2 3 5 10 20", {}),
    ("run_sweeps.py", "collective_R 0.2 0.5 1 2 3 5 10 20", {}),
]
STATIC = [  # limits A and B at matched T1, static inhomogeneity: symmetry-breaking class and Redfield (Tables I and IV)
    ("static_runs.py", "sel A P 0.2 0.5 1 2 5 20", {}),
    ("static_runs.py", "sel A R 0.5 5 20", {}),
    ("static_runs.py", "sel B R 0.5 5 20", {}),
    ("static_runs.py", "sel B Z 0.5 5", {}),
    ("static_runs.py", "sel B P 0.2 0.5 1 2 3 5 10 20", {}),
    ("static_runs.py", "refine A 0.2 0.5 1 2 5", {"NCOLD": "0"}),
    ("static_runs.py", "refine B 0.2 0.5 1 2 3 5 10 20", {"NCOLD": "2"}),
    ("static_runs.py", "refine B 0.2 0.5 1 3 10 20", {"SEED": "1000", "NCOLD": "2"}),
    ("static_runs.py", "refine A 20", {"SEED": "1000", "NCOLD": "2"}),
    ("static_runs.py", "sel A P 3 10", {}),
    ("static_runs.py", "refine A 0.2 0.5 1 2", {"SEED": "2000", "NCOLD": "3"}),
    ("static_runs.py", "refine A 3 5 10 20", {"SEED": "2000", "NCOLD": "3"}),
    ("static_runs.py", "refine A 0.2 0.5 1 2 3 5", {"SEED": "3000", "NCOLD": "2"}),
    ("static_runs.py", "refine A 10", {"SEED": "4000", "NCOLD": "4"}),
    ("static_runs.py", "refine A 2 3 5", {"SEED": "4000", "NCOLD": "1"}),
    ("redfield.py", "rates", {}),
    ("redfield.py", "run 0.3 1.0", {}),
]
STATIC_SYM = [  # symmetric class in limits A and B (Table II, Fig. 4a); shares no files with the chains above
    ("static_runs.py", "sym B P,R 0.5 2 5 10", {}),
    ("static_runs.py", "sym A P,R 0.5 2 5 10", {}),
    ("static_runs.py", "sym A P,R 0.5 2 5 10", {"SEED": "500"}),
    ("static_runs.py", "sym A P,R 0.5 2 5 10", {"SEED": "900"}),
]
MAIN_OUTPUTS = [  # checks, tables and figures of the main text (seconds to minutes)
    ("validate_static.py", "", {}),
    ("verify_crossover.py", "", {}),
    ("print_model_tables.py", "", {}),
    ("make_level_scheme.py", "", {}),
    ("make_crossover_figure.py", "", {}),
    ("make_lifetimes_figure.py", "", {}),
    ("make_static_figures.py", "", {}),
]
SUPPLEMENT = [  # Supplemental Material (about 10 min, most of it in sm_robustness.py)
    ("sm_asymmetry.py", "", {}),
    ("sm_rates.py", "", {}),
    ("sm_selection_checks.py", "", {}),
    ("sm_convergence.py", "", {}),
    ("sm_gain_estimate.py", "", {}),
    ("sm_robustness.py", "", {}),
    ("sm_manybody.py", "", {}),
]

# The matrices are 16 x 16 or 32 x 32, so multithreaded linear algebra only adds overhead and, with
# --parallel, oversubscribes the processor. Each step therefore runs on one thread.
THREADS = {k: os.environ.get(k, "1") for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}

_lock = threading.Lock()


def run_chain(name, steps, dry):
    for k, (script, args, env) in enumerate(steps, 1):
        cmd = [PY, "-W", "ignore", script] + args.split()
        label = " ".join(f"{a}={b}" for a, b in env.items()) + (" " if env else "") + " ".join(cmd[3:])
        with _lock:
            print(f"[{name} {k}/{len(steps)}] {label}", flush=True)
        if dry:
            continue
        t0 = time.time()
        r = subprocess.run(cmd, env={**os.environ, **THREADS, **env}, capture_output=True, text=True)
        with _lock:
            for line in r.stdout.splitlines():
                print(f"    {line}")
            print(f"    ({time.time() - t0:.0f} s)", flush=True)
        if r.returncode != 0:
            with _lock:
                print(r.stderr)
            raise SystemExit(f"step failed: {label}")


# ---------------------------------------------------------------------------------------------
# Values quoted in the paper, for --check. Keys are delta in Hz. Ratios are protected / reference.
# ---------------------------------------------------------------------------------------------
QUOTED = {
    "Table I, limit A": ("static_sel_A", "sel", {0.2: 0.016, 0.5: 0.033, 1: 0.093, 2: 0.32, 3: 0.65, 5: 1.4, 10: 2.8, 20: 3.4}),
    "Table I, limit B": ("static_sel_B", "sel", {0.2: 0.31, 0.5: 0.79, 1: 2.35, 2: 7.4, 3: 14.4, 5: 27.8, 10: 51, 20: 69}),
    "Table I, uncoupled pair": ("static_sel_B", "selZ", {0.5: 2.8, 5: 1.2}),
    "Table II, limit A": ("static_sym_A", "sym", {0.5: 0.006, 2: 0.079, 5: 0.28, 10: 0.85}),
    "Table II, limit B four-pulse": ("static_sym_B", "sym4", {0.5: 0.0029, 2: 1.2, 5: 5.9, 10: 21}),
    "Table II, limit B echo trains": ("static_sym_B", "sym", {0.5: 0.31, 2: 1.6, 5: 6.8, 10: 33}),
    "Table V, Markovian symmetry-breaking": ("selective_cm", "cm", {0.2: 0.34, 0.5: 0.80, 1: 2.36, 2: 7.6, 3: 14.5, 5: 28.7, 10: 50, 20: 71}),
    "Table V, Markovian symmetric": ("collective_P", "coll", {0.2: 0.043, 0.5: 0.052, 1: 0.074, 2: 0.087, 3: 0.53, 5: 1.05, 10: 13, 20: 27}),
}


def check(tol=0.05):
    def load(name):
        return json.load(open(f"results/{name}.json"))["points"]
    bad = 0
    for label, (fname, kind, vals) in QUOTED.items():
        try:
            pts = load(fname)
            refB = load("static_sel_B")["0.5"]["R"]["F"]
            collR = load("collective_R") if kind == "coll" else None
        except FileNotFoundError as e:
            print(f"  {label}: missing {e.filename}")
            bad += 1
            continue
        for d, q in vals.items():
            rec = pts[str(float(d))]
            if kind == "sel":
                got = rec["P"]["F"] / pts["0.5"]["R"]["F"]
            elif kind == "selZ":
                got = rec["Z"]["F"] / refB
            elif kind == "sym":
                got = rec["P"]["echo_best"] / rec["R"]["F"]
            elif kind == "sym4":
                got = rec["P"]["four_pulse"] / rec["R"]["F"]
            elif kind == "cm":
                got = rec["P"]["F"] / rec["R"]["F"]
            else:
                got = rec["echo_best"] / collR[str(float(d))]["F"]
            # quoted values are rounded to two or three significant figures
            ok = abs(got / q - 1) <= tol
            bad += not ok
            print(f"  {label:38s} delta/J = {d / 10:<5g} paper {q:<7g} result {got:<9.4g} {'ok' if ok else 'MISMATCH'}")
    print(f"\n{'all quoted values reproduced' if not bad else f'{bad} mismatches'} (tolerance {tol:.0%})")
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--main", action="store_true", help="main text only")
    g.add_argument("--supplement", action="store_true", help="Supplement only (needs existing results)")
    g.add_argument("--figures", action="store_true", help="figures, tables and checks from existing results")
    g.add_argument("--check", action="store_true", help="compare results with the values quoted in the paper")
    ap.add_argument("--parallel", action="store_true", help="run the three independent optimization chains at once")
    ap.add_argument("--dry-run", action="store_true", help="list the steps only")
    ap.add_argument("--copy-figures", metavar="DIR", help="copy figs/*.pdf to DIR at the end")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figs", exist_ok=True)
    if a.check:
        sys.exit(0 if check() else 1)

    t0 = time.time()
    if not (a.supplement or a.figures):
        if a.parallel and not a.dry_run:
            chains = (("markovian", MARKOVIAN), ("static", STATIC), ("symmetric", STATIC_SYM))
            th = [threading.Thread(target=run_chain, args=(n, s, False)) for n, s in chains]
            [t.start() for t in th]
            [t.join() for t in th]
        else:
            run_chain("markovian", MARKOVIAN, a.dry_run)
            run_chain("static", STATIC, a.dry_run)
            run_chain("symmetric", STATIC_SYM, a.dry_run)
    if not a.supplement:
        run_chain("main outputs", MAIN_OUTPUTS, a.dry_run)
    if not a.main:
        run_chain("supplement", SUPPLEMENT, a.dry_run)
    if a.dry_run:
        return
    if a.copy_figures:
        os.makedirs(a.copy_figures, exist_ok=True)
        for f in sorted(os.listdir("figs")):
            if f.endswith(".pdf"):
                shutil.copy(os.path.join("figs", f), a.copy_figures)
        print(f"figures copied to {a.copy_figures}")
    print(f"\nfinished in {(time.time() - t0) / 60:.1f} min\n")
    if not a.supplement:
        check()


if __name__ == "__main__":
    main()
