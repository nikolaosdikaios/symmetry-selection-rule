#!/usr/bin/env python3
"""
verify_selection_rule.py
========================

Run:  python3 verify_selection_rule.py
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar, brentq

np.set_printoptions(precision=5, suppress=True)

# ----------------------------------------------------------------------------
# operators
# ----------------------------------------------------------------------------
sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
sp, sm, e2 = sx + 1j * sy, sx - 1j * sy, np.eye(2, dtype=complex)
kr = np.kron
I1 = {k: kr(v, e2) for k, v in dict(x=sx, y=sy, z=sz, p=sp, m=sm).items()}
I2 = {k: kr(e2, v) for k, v in dict(x=sx, y=sy, z=sz, p=sp, m=sm).items()}
up, dn = np.array([1, 0], complex), np.array([0, 1], complex)
uu, ud, du, dd = kr(up, up), kr(up, dn), kr(dn, up), kr(dn, dn)
S, T0, Tp, Tm = (ud - du) / np.sqrt(2), (ud + du) / np.sqrt(2), uu, dd
Fp, Fz, G = I1['p'] + I2['p'], I1['z'] + I2['z'], I1['z'] - I2['z']

print("=" * 78)
print("[A] Brightness/darkness identities (Eqs. 2-3)")
print("=" * 78)
print(f"  || F+ |S> ||                    = {np.linalg.norm(Fp @ S):.3e}   (expect 0)")
print(f"  || (I1+-I2+)|S> + sqrt(2)|T+> || = {np.linalg.norm((I1['p']-I2['p'])@S + np.sqrt(2)*Tp):.3e}   (expect 0)")
print(f"  || (I1z-I2z)|S> - |T0> ||        = {np.linalg.norm(G @ S - T0):.3e}   (expect 0)")
print(f"  <T0|(I1z-I2z)|S>                 = {(T0.conj() @ (G @ S)).real:.12f}   (expect 1)")

print()
print("=" * 78)
print("[B] S-T0 block: splitting and mixing angle")
print("=" * 78)
dot = I1['x'] @ I2['x'] + I1['y'] @ I2['y'] + I1['z'] @ I2['z']
J = 10.0
for d in (1.0, 3.0, 10.0):
    H = 2 * np.pi * J * dot + 2 * np.pi * d / 2 * G
    B = np.stack([S, T0], axis=1)
    Hb = B.conj().T @ H @ B
    ev = np.linalg.eigvalsh(Hb)
    print(f"  d={d:5.1f} Hz: splitting/2pi = {(ev[1]-ev[0])/2/np.pi:8.4f}"
          f"  vs sqrt(J^2+d^2) = {np.sqrt(J*J+d*d):8.4f};"
          f"  tan(2theta) = {2*abs(Hb[0,1])/abs(Hb[1,1]-Hb[0,0]):.4f} vs d/J = {d/J:.4f}")

print()
print("=" * 78)
print("[C] Intra-pair dipolar algebra (Appendix A)")
print("=" * 78)
T2m = {
    +2: 0.5 * I1['p'] @ I2['p'],
    -2: 0.5 * I1['m'] @ I2['m'],
    +1: -0.5 * (I1['p'] @ I2['z'] + I1['z'] @ I2['p']),
    -1: +0.5 * (I1['m'] @ I2['z'] + I1['z'] @ I2['m']),
     0: (1/np.sqrt(6)) * (2*I1['z']@I2['z'] - 0.5*(I1['p']@I2['m'] + I1['m']@I2['p'])),
}
print("  || T_2m |S> || , m=-2..+2 :", [f"{np.linalg.norm(T2m[m]@S):.1e}" for m in (-2,-1,0,1,2)], " (all 0)")
K = sum(A.conj().T @ A for A in T2m.values())
print(f"  || K|S> || = {np.linalg.norm(K@S):.3e};   k_T = <T0|K|T0> = {(T0.conj()@K@T0).real:.6f}"
      f" = <T+|K|T+> = {(Tp.conj()@K@Tp).real:.6f}   (scalar; = 5/12)")

# ----------------------------------------------------------------------------
# Lindblad superoperators (column-stacking vec)
# ----------------------------------------------------------------------------
def dissipator(ops):
    L = np.zeros((16, 16), complex); Id = np.eye(4, dtype=complex)
    for A in ops:
        AdA = A.conj().T @ A
        L += np.kron(A.conj(), A) - 0.5 * (np.kron(Id, AdA) + np.kron(AdA.T, Id))
    return L

def rate(L, X):
    """Initial decay rate -<X, L X>/<X,X>; residual=0 iff X is an eigenoperator."""
    v = X.reshape(-1, order='F'); Lv = L @ v; n = np.vdot(v, v)
    r = -np.vdot(v, Lv) / n
    res = np.linalg.norm(Lv + r * v) / np.linalg.norm(v)
    return r.real, res

mechanisms = {
    "intra-pair dipole-dipole": dissipator(T2m.values()),
    "correlated fields (Fz)":   dissipator([Fz]),
    "uncorrelated local fields": dissipator([I1['z'], I2['z']]),
    "anticorrelated (I1z-I2z)": dissipator([G]),
}
probes = {
    "S population |S><S|": np.outer(S, S.conj()),
    "ZQ coherence |S><T0|": np.outer(S, T0.conj()),
    "SQ  (F+)":             Fp,
    "DQ  |T+><T-|":         np.outer(Tp, Tm.conj()),
    "Fz  (T1 order)":       Fz,
}
print()
print("=" * 78)
print("[D] Mechanism-resolved rates (Table I of Appendix A); unit noise strength")
print("=" * 78)
for name, L in mechanisms.items():
    print(f"  -- {name} --")
    for pname, X in probes.items():
        r, res = rate(L, X)
        tag = "eigen-op" if res < 1e-12 else f"mixes (res {res:.1f})"
        print(f"     {pname:22s} rate = {r:8.5f}   [{tag}]")

print()
print("=" * 78)
print("[E] Dipole-limited coherence budget")
print("=" * 78)
Ldd = mechanisms["intra-pair dipole-dipole"]
r2, _ = rate(Ldd, Fp)
rc, _ = rate(Ldd, np.outer(S, T0.conj()))
vF = Fp.reshape(-1, order='F')
for tt in (0.5/r2, 1.0/r2, 2.0/r2):
    s = (np.vdot(vF, expm(Ldd*tt) @ vF) / np.vdot(vF, vF)).real
    print(f"  F+ survival at t={tt:6.3f}: {s:.5f}  (single exponential predicts {np.exp(-r2*tt):.5f})")
print(f"  ==>  R_c / R2 = {rc/r2:.6f}  (exactly 1/3)  =>  T_c = {r2/rc:.3f} x T2 under intra-pair DD")
print( "       Peak advantage cap under DD:  P_c/4 = 3/4 (asymptotic); exact law peaks at a")
print( "       marginal ~1.1 near delta~J, outside its quantitative domain -> window effectively closed.")

print()
print("=" * 78)
print("[F] Closed-form window law, Appendix E")
print("=" * 78)
print("  Corrected law  A(u) = u P^2 (1+u)/(1+Pu)^2,  u=(d/J)^2 :")
for P in (3, 25, 100, 400):
    A = lambda u: u * P**2 * (1 + u) / (1 + P*u)**2
    res = minimize_scalar(lambda lu: -A(np.exp(lu)), bounds=(np.log(1e-8), np.log(1e4)), method='bounded')
    u = np.exp(res.x); pk = A(u)
    try:
        lo = np.sqrt(brentq(lambda uu: A(uu) - 1, 1e-12, u))
    except ValueError:
        lo = float('nan')
    print(f"    P={P:4d}:  x* = {np.sqrt(u):.4f} (P^-1/2 = {P**-0.5:.4f});"
          f"  A* = {pk:8.3f} (P/4 = {P/4:7.2f});  lower edge x = {lo:.4f} (1/P = {1/P:.4f})")
print("  For P=25: x*=0.208, A*=6.5, edge 0.042 -- vs measured 0.2 / 7.6 / 0.04 (Fig. 3).")
print("  Uncorrected law u[P(1+u)/(1+Pu)]^2 at large u (why its domain is 'through the peak'):")
for P in (25,):
    A0 = lambda u: u * (P*(1+u)/(1+P*u))**2
    print(f"    P={P}: A0(u=0.048)={A0(0.048):.2f} (local peak) but A0(u=50)={A0(50):.1f} -> unbounded; "
          f"corrected law saturates: A(u=50)={50*P**2*51/(1+P*50)**2:.3f}")

print()
print("=" * 78)
print("[G] Independent baseline: collective vs individual readout (FID model)")
print("=" * 78)
T2v = 0.2
t = np.arange(5e-4, 2.0, 1e-3); env = np.exp(-t / T2v)
F_ind = 2 * 2 * np.sum((np.pi * t * env) ** 2)          # two receivers, unit noise each
for d in (0.1, 0.2, 0.4, 0.8, 1.6, 3.0):
    ds = -2 * np.pi * t * np.sin(np.pi * d * t) * env   # collective: s = 2 cos(pi d t) env
    F_coll = 2 * np.sum(ds ** 2)
    print(f"  delta = {d:4.1f} Hz: F_coll/F_ind = {F_coll/F_ind:6.3f}"
          f"   [small-d analytic 6(pi d T2)^2 = {6*(np.pi*d*T2v)**2:6.3f}]")
print("  Linewidth 1/(pi T2) = 1.6 Hz: the two baselines coincide only above it.")
print()
print("All checks complete.")
