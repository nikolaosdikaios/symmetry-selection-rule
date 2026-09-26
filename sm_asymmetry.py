"""Supplement S1: asymmetry of the protected sector (Wigner-Yanase skew information for the generator I1z - I2z)
and the J-limited quantum Fisher information of a singlet probe."""
import numpy as np
from scipy.linalg import expm, sqrtm
import model as M

def skew(rho, A):
    s = sqrtm(rho); c = s @ A - A @ s
    return float(np.real(-0.5 * np.trace(c @ c)))
print("Skew information I(rho, I1z - I2z) and brightness |F+ psi|")
for name, v in {"S": M.S, "T0": M.T0, "(ud + i du)/sqrt2": (M.UD + 1j * M.DU) / np.sqrt(2),
                "ud": M.UD, "T+": M.Tp}.items():
    print(f"  {name:18s} I = {skew(np.outer(v, v.conj()), M.G):.4f}   |F+ psi| = {np.linalg.norm(M.Fp @ v):.4f}")
eps = 1e-5
print(f"  thermal 1/4 + eps Fz          : I = {skew(np.eye(4)/4 + eps*M.Fz, M.G):.1e}")
print(f"  after a 90-degree pulse, Fx   : I / eps^2 = {skew(np.eye(4)/4 + eps*M.Fx, M.G)/eps**2:.4f}")
J = 10.0
print("\nQuantum Fisher information of a pure singlet at delta -> 0 (Hz^-2) versus 4 sin^2(pi J t)/J^2")
for t in [0.01, 0.025, 0.05, 0.075, 0.1, 0.3]:
    psi = lambda d: expm(-1j * t * (M.TWOPI * J * M.IdotI + np.pi * d * M.G)) @ M.S
    h = 1e-6; dpsi = (psi(h) - psi(-h)) / (2 * h); p0 = psi(0.0)
    F = 4 * (np.vdot(dpsi, dpsi) - abs(np.vdot(p0, dpsi))**2).real
    print(f"  t = {t:5.3f} s:  numerical {F:.6f}   analytic {4*np.sin(np.pi*J*t)**2/J**2:.6f}   (J = 0: {4*np.pi**2*t**2:.4f})")
