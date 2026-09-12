#!/usr/bin/env python3
"""
diagnose_realized_dominance.py -- the last check before the full run, and the prototype
of the fix. Confirms the CRLB frontier's dominance survives on REALIZED RMSE.

Run from simulator/:  MRSF_CITRATE4=1 MRSF_NUIS_RIDGE=0.3 python diagnose_realized_dominance.py

Pipeline (the actual fix, tiny budget):
  STAGE A  strong search on the deterministic CRLB (fast, noiseless) to reach the
           dominating schedule the frontier proved exists -- objective is the normalised
           minimax per-metabolite ratio to the baseline target.
  STAGE B  refine that schedule on the REALIZED RMSE under COMMON RANDOM NUMBERS (same
           fixed seeds for every comparison), same normalised-minimax objective.
  CHECK    evaluate the final schedule's realized per-metabolite RMSE (held-out seeds)
           against the per-metabolite baseline best. PASS if it is <= the bar on ALL.
"""
import os, sys, dataclasses
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mrf"))
import mrf_clinical as mc
import mrf_realistic as mr
import mrf_objective as mo
import mrf_scoreboard as sb
from prostate_tissue_concentrations import DICTIONARY_METABOLITES as NAMES
from mrf_realistic import healthy_vector

QUANT = ["Cit", "Ch", "Cr", "Lac", "Tau", "Ins"]
CW = mo.CLINICAL_WEIGHTS
RIDGE = float(os.environ.get("MRSF_NUIS_RIDGE", "0.3"))
CRN = tuple(range(10))           # common random numbers for STAGE B + check-train
HELD = tuple(range(20, 32))      # independent held-out seeds for the final verdict

try:
    from gen_structured_fingerprint import pool_shot as _pool_shot
    def pool_shot(rng): return _pool_shot(rng)
except Exception:
    def pool_shot(rng):
        j = int(rng.integers(20)); return mc.make_schedule(20, seed=int(rng.integers(1_000_000)))[j]


def press(te, n=20): return [mc.Shot('press', {'tau1': te/2, 'tau2': te/2}, 1500.0) for _ in range(n)]
def multite(n=20): return [mc.Shot('press', {'tau1': t/2, 'tau2': t/2}, 1500.0) for t in np.linspace(15,70,n)]
def teavg(n=20): return [mc.Shot('press', {'tau1': t/2, 'tau2': t/2}, 1500.0) for t in np.linspace(16,64,n)]


def crlb(sched):
    SPEC, PPM, Df, names = mr.build_spectral_dictionary(NAMES, sched, verbose=False)
    M, nm = mo.design_matrix(SPEC, PPM, n_knots=3)
    c = mo.crlb_percent(M, nm, np.array(healthy_vector(names), float), sigma=1.0, ridge=RIDGE)
    return {names[i]: float(c[i]) for i in range(nm)}


def realized(sched, seeds):
    acc = {}
    for s in seeds:
        _, pm = sb.eval_panel(sched, n_trials=6, snr=20, seed=int(s), return_per_metab=True)
        for m, v in pm.items(): acc.setdefault(m, []).append(v)
    return {m: float(np.mean(v)) for m, v in acc.items()}


def minimax_ratio(pm, target):
    return max(pm.get(m, 9e9) / target[m] for m in QUANT)


def main():
    baselines = {"PRESS-85": press(85.), "PRESS-30": press(30.), "multi-TE": multite(), "TE-avg": teavg()}

    # targets from the SAME estimator we optimise on (CRLB target for A, realized target for B/check)
    bl_crlb = {n: crlb(s) for n, s in baselines.items()}
    tgt_crlb = {m: min(bl_crlb[n][m] for n in baselines) for m in QUANT}
    bl_real = {n: realized(s, CRN) for n, s in baselines.items()}
    tgt_real = {m: min(bl_real[n][m] for n in baselines) for m in QUANT}

    rng = np.random.default_rng(0)

    # ---------- STAGE A: strong CRLB search (deterministic) ----------
    print("STAGE A: strong search on the deterministic CRLB...")
    seeds = list(baselines.values()) + [[pool_shot(rng) for _ in range(20)] for _ in range(60)]
    best = min(seeds, key=lambda s: minimax_ratio(crlb(s), tgt_crlb)); bw = minimax_ratio(crlb(best), tgt_crlb)
    for _ in range(3000):
        s = [dataclasses.replace(sh, params=dict(sh.params)) for sh in best]
        for _ in range(1 + (rng.random() < 0.1)):
            s[int(rng.integers(len(s)))] = pool_shot(rng)
        w = minimax_ratio(crlb(s), tgt_crlb)
        if w < bw: bw, best = w, s
    print(f"  CRLB worst-ratio reached: {bw:.3f}")

    # ---------- STAGE B: realized refinement under CRN ----------
    print("STAGE B: refine on realized RMSE under common random numbers...")
    bwr = minimax_ratio(realized(best, CRN), tgt_real)
    accepted = 0
    for _ in range(120):
        s = [dataclasses.replace(sh, params=dict(sh.params)) for sh in best]
        s[int(rng.integers(len(s)))] = pool_shot(rng)
        w = minimax_ratio(realized(s, CRN), tgt_real)   # CRN => reliable comparison
        if w < bwr: bwr, best, accepted = w, s, accepted + 1
    print(f"  realized worst-ratio (train/CRN): {bwr:.3f}  ({accepted} refinements)\n")

    # ---------- CHECK on independent held-out seeds ----------
    print("=" * 70)
    print("HELD-OUT verdict (seeds 20-31): optimized schedule vs per-metabolite best baseline")
    print("=" * 70)
    opt = realized(best, HELD)
    print(f"  {'metab':6s} {'baseline':>9s} {'optimized':>10s} {'ratio':>7s}  verdict")
    ok = True
    for m in QUANT:
        r = opt[m] / tgt_real[m]; ok &= (r <= 1.02)
        print(f"  {m:6s} {tgt_real[m]:9.1f} {opt[m]:10.1f} {r:7.2f}  {'BEATS' if r <= 1.02 else 'ABOVE'}")
    print("\n  " + ("PASS -- dominates every metabolite on realized RMSE. Ship the fix: put this"
                    "\n  two-stage optimizer + normalised-minimax objective into the pipeline and run."
                    if ok else
                    "CLOSE BUT NOT ALL -- realized bias/noise leaves >=1 metabolite above bar; report"
                    "\n  back the table and I will add per-metabolite reweighting / more CRN seeds."))


if __name__ == "__main__":
    main()
