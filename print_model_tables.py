"""Relaxation-model tables (Appendix B): secular decay rates per mechanism (Table III) and the strengths and
lifetimes implied by fitting (T1, T2, T_S) for the environments used in the paper."""
import numpy as np
import model as M

fmt = lambda v: "inf" if not np.isfinite(v) else f"{v:.3f}"
print("Secular decay rates per unit mechanism strength (s^-1), delta = J = 0")
print(f"{'mechanism':10s}" + "".join(f"{k:>8s}" for k in M.OBS))
for m, v in M.unit_rates().items():
    print(f"{m:10s}" + "".join(f"{v[k]:8.3f}" for k in M.OBS))
envs = {"homogeneous relaxation of limits A and B": (1.2, 1.2, 5.0),
        "Markovian common-mode comparison (Appendix D)": (1.2, 0.2, 5.0),
        "homogeneous relaxation with T_S -> infinity (bound T_c = 3 T1)": (1.2, 1.2, 1e12)}
for name, (T1, T2, TS) in envs.items():
    g = M.solve_environment(T1, T2, TS); lt = M.lifetimes(g)
    print(f"\n{name}: T1 = {T1} s, T2 = {T2} s, T_S = {TS:g} s")
    print("  strengths (s^-1):", {k: round(float(v), 4) for k, v in g.items()})
    print("  lifetimes (s):   ", {k: fmt(v) for k, v in lt.items()}, f"  P_c = T_c/T2 = {lt['Tc']/T2:.2f}")
print()
for T1, T2 in [(1.2, 1.2), (1.2, 0.2)]:
    print(f"independent-spin reference, T1 = {T1} s, T2 = {T2} s:", {k: round(float(v), 4) for k, v in M.solve_reference(T1, T2).items()})
