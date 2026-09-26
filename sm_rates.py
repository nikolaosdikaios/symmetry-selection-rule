"""Supplement S2: secular rates for six noise mechanisms, and the coherence bound in the fitted homogeneous environment."""
import numpy as np
import model as M
mechs = dict(M.MECH, anticorr_z=M.diss(M.Iz[0] - M.Iz[1]), corr_iso=M.diss(M.Fx) + M.diss(M.Fy) + M.diss(M.Fz))
names = {"corr_z": "common-mode longitudinal", "corr_iso": "common-mode isotropic", "anticorr_z": "anticorrelated longitudinal",
         "uncorr_z": "independent longitudinal", "uncorr": "independent isotropic", "dipolar": "intra-pair dipolar"}
print(f"{'mechanism':28s}{'R1':>7s}{'R2':>7s}{'RS/R2':>8s}{'Rc/R2':>8s}{'RZQ/R2':>8s}")
for m in names:
    r = {k: M.decay_rate(mechs[m], O)[0] for k, O in M.OBS.items()}
    print(f"{names[m]:28s}{abs(r['R1']):7.3f}{r['R2']:7.3f}{abs(r['RS'])/r['R2']:8.3f}{abs(r['Rc'])/r['R2']:8.3f}{abs(r['RZQ'])/r['R2']:8.3f}")
g = M.solve_environment(1.2, 1.2, 5.0); L = M.dissipator(g)
B = {"S": M.S, "T+": M.Tp, "T0": M.T0, "T-": M.Tm}
def w(a, b, L=L):
    out = (L @ M.vec(M.ket(B[a], B[a]))).reshape(4, 4, order="F")
    return float(np.real(B[b].conj() @ out @ B[b]))
Gam = {a: sum(w(a, b) for b in B if b != a) for a in B}
Rc = M.decay_rate(L, M.OBS["Rc"])[0]; half = (Gam["S"] + Gam["T0"]) / 2
print("\nfitted homogeneous environment: population out-rates (s^-1)", {k: round(v, 4) for k, v in Gam.items()})
print(f"R_c = {Rc:.4f} = (Gamma_S + Gamma_T0)/2 [{half:.4f}] + pure dephasing [{Rc - half:.4f}] s^-1")
print(f"T_c = {1/Rc:.3f} s <= 2/(Gamma_S + Gamma_T0) = {1/half:.3f} s <= 3 T1 = 3.6 s;  2 R1/3 = {2/3/1.2:.4f} s^-1")

Ld = M.MECH["dipolar"]; rd = {k: M.decay_rate(Ld, O)[0] for k, O in M.OBS.items()}
Gd = {a: sum(w(a, b, Ld) for b in B if b != a) for a in B}
Dop = M.ket(M.Tp, M.Tp) + M.ket(M.Tm, M.Tm) - 2 * M.ket(M.T0, M.T0)
print("\ndipolar term per unit strength: out-rates", {k: round(v, 4) for k, v in Gd.items()},
      f"| R1 = {rd['R1']:.4f}, dipolar-order rate = {M.decay_rate(Ld, Dop)[0]:.4f}, Gamma_T0/R1 = {Gd['T0']/rd['R1']:.4f}")
print(f"  R_c = {rd['Rc']:.4f} = Gamma_T0/2 [{Gd['T0']/2:.4f}] + secular dephasing [{rd['Rc'] - Gd['T0']/2:.4f}];  R_c/R1 = {rd['Rc']/rd['R1']:.4f}")
print("R_c >= R1/3 for every mechanism:", {m: bool(M.decay_rate(L_, M.OBS['Rc'])[0] >= M.decay_rate(L_, M.OBS['R1'])[0] / 3 - 1e-12) for m, L_ in mechs.items()})
