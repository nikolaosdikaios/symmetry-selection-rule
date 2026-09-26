"""Numerical check of Proposition 1: best exchange-invariant measurement (QFI of the twirled family) versus the
unrestricted QFI for a singlet probe of purity p (J = 0), and the crossover formula."""
import numpy as np
from scipy.linalg import expm, sqrtm
# Two spins, J = 0, generator H = pi*delta*G, singlet probe with purity p: rho0 = p|S><S| + (1-p) I/4
sz = np.diag([0.5, -0.5]); I2 = np.eye(2)
G = np.kron(sz, I2) - np.kron(I2, sz)
S = np.array([0, 1, -1, 0]) / np.sqrt(2)
SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=float)
def rho(delta, p, t):
    U = expm(-1j * np.pi * delta * t * G)
    r0 = p * np.outer(S, S) + (1 - p) * np.eye(4) / 4
    return U @ r0 @ U.conj().T
def qfi(fam, x, h=1e-6):
    r, dr = fam(x), (fam(x + h) - fam(x - h)) / (2 * h)
    w, V = np.linalg.eigh(r); F = 0
    dr = V.conj().T @ dr @ V
    for i in range(4):
        for j in range(4):
            if w[i] + w[j] > 1e-12: F += 2 * abs(dr[i, j])**2 / (w[i] + w[j])
    return F.real
t = 1.0
for p in [1.0, 0.9, 0.5, 1e-2]:
    xstar = np.sqrt((1 - p) / (4 * p)) if p < 1 else 0.0
    print(f"p = {p}:  delta* = {xstar/(np.pi*t):.3g} Hz  (t = {t} s)")
    for d in [1e-3, 1e-2, 0.05, 0.1]:
        full = qfi(lambda x: rho(x, p, t), d)                         # any measurement
        sym  = qfi(lambda x: 0.5 * (rho(x, p, t) + SWAP @ rho(x, p, t) @ SWAP), d)   # best exchange-invariant measurement
        x = np.pi * d * t
        approx = p * 4 * np.pi**2 * t**2 * x**2 / (x**2 + xstar**2) if p < 1 else 4 * np.pi**2 * t**2
        print(f"   delta = {d:6.3f} Hz:  QFI(all) = {full:8.4f}   best symmetric = {sym:8.4f}   formula = {approx:8.4f}")
