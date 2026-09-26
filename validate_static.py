"""Appendix C checks of the static-offset pathway method: with Gamma = 0 the pathway sum must reproduce
direct propagation, and for a plain free-induction decay the static and Markovian descriptions must coincide.
Also prints the homogeneous lifetimes and the timing of one Fisher evaluation."""
import numpy as np, time, model as M, protocols as Pr, static_runs as SR
rng = np.random.default_rng(1)
# (1),(2) Gamma = 0 must reproduce the direct propagation (same 360-sample acquisition)
M.NF = SR.NF
for arm in ("P", "R"):
    L0, rows_i = SR.setup(2.0, arm, M.MEAS_INDIV); _, rows_c = SR.setup(2.0, arm, M.MEAS_COLL)
    Ff = M.aug(L0, M.DT)
    ps = np.array([rng.uniform(lo, hi) for lo, hi in Pr.BND_S]); pc = np.array([rng.uniform(lo, hi) for lo, hi in Pr.BND_C])
    a = SR.fisher_selective_static(ps, L0, rows_i, 0.0); b = Pr.fisher_selective(ps, L0, Ff)
    c = SR.fisher_collective_static(pc, L0, 2, 4, rows_c, 0.0); e = Pr.fisher_collective(pc, L0, 2, 4, Ff)
    print(f"{arm}: selective pathways {a:.10g} vs direct {b:.10g}; collective pathways {c:.10g} vs direct {e:.10g}")
# (3) single-pathway check: 90-degree pulses then FID; static Lorentzian = Markovian longitudinal dephasing at rate Gamma
Gam = SR.GAMMA["B"]; p = np.zeros(15); p[0] = p[2] = np.pi / 2; p[1] = p[3] = np.pi / 2
L0, rows = SR.setup(2.0, "R", M.MEAS_INDIV)
g_mark = dict(SR.G_REF); g_mark["uncorr_z"] = 2 * Gam
L0m = M.liouvillian(2.0, 0.0, g_mark)
print(f"FID only: static {SR.fisher_selective_static(p, L0, rows, Gam):.10g} vs Markovian {Pr.fisher_selective(p, L0m, M.aug(L0m, M.DT)):.10g}")
# (4) echo check: with static offsets a collective 90-tau-180-tau-FID echo recovers signal; Markovian cannot
print("lifetimes, homogeneous environment:", {k: round(v, 3) for k, v in M.lifetimes(SR.G_HOM).items()}, " Gamma_B =", round(Gam, 4))
# timing
L0, rows = SR.setup(2.0, "P", M.MEAS_INDIV); t = time.time()
for _ in range(50): SR.fisher_selective_static(ps, L0, rows, Gam)
print(f"selective static: {(time.time()-t)/50*1e3:.2f} ms/eval")
L0, rows = SR.setup(2.0, "P", M.MEAS_COLL); t = time.time()
for _ in range(20): SR.fisher_collective_static(pc, L0, 4, 4, rows, Gam)
print(f"collective static (4,4): {(time.time()-t)/20*1e3:.2f} ms/eval")
t = time.time(); SR.fid_rows(L0, M.MEAS_INDIV); print(f"fid_rows: {(time.time()-t)*1e3:.1f} ms")
