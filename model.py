"""
model.py -- two-spin model used in the paper.

Hamiltonian (frequencies in Hz, times in s):
    H = 2*pi*J (I1 . I2) + pi*delta*(I1z - I2z).

Relaxation is a sum of Lindblad dissipators, one per noise mechanism, in the
white-noise (extreme-narrowing, high-temperature) limit. Each term is completely
positive by construction, so every lifetime follows from the mechanism strengths:
    corr_z   common-mode longitudinal field      L = Fz
    dipolar  intra-pair dipolar coupling         L_q = T_{2,q},  q = -2..2
    uncorr   independent isotropic local fields  L = I_{k,alpha}
    uncorr_z independent longitudinal fields     L = I_{kz}
The dissipators do not depend on delta, so dL/d(delta) is purely Hamiltonian and
the derivative propagated below is exact.

Fisher information: F = 2 * sum over samples and receivers of |ds/d delta|^2,
the classical Fisher information of delta (in Hz^-2) for unit-variance circular
complex Gaussian noise on every sample of every receiver.
"""
import numpy as np
from scipy.linalg import expm

TWOPI = 2 * np.pi
_sx = np.array([[0, 1], [1, 0]], complex) / 2
_sy = np.array([[0, -1j], [1j, 0]], complex) / 2
_sz = np.array([[1, 0], [0, -1]], complex) / 2
_e = np.eye(2)
Ix = [np.kron(_sx, _e), np.kron(_e, _sx)]
Iy = [np.kron(_sy, _e), np.kron(_e, _sy)]
Iz = [np.kron(_sz, _e), np.kron(_e, _sz)]
Ip = [Ix[k] + 1j * Iy[k] for k in (0, 1)]
Im = [Ix[k] - 1j * Iy[k] for k in (0, 1)]
Fx, Fy, Fz = Ix[0] + Ix[1], Iy[0] + Iy[1], Iz[0] + Iz[1]
Fp = Fx + 1j * Fy
G = Iz[0] - Iz[1]                                    # exchange-odd generator
IdotI = Ix[0] @ Ix[1] + Iy[0] @ Iy[1] + Iz[0] @ Iz[1]
d, D = 4, 16

_u, _d = np.array([1, 0], complex), np.array([0, 1], complex)
S = (np.kron(_u, _d) - np.kron(_d, _u)) / np.sqrt(2)
T0 = (np.kron(_u, _d) + np.kron(_d, _u)) / np.sqrt(2)
Tp, Tm = np.kron(_u, _u), np.kron(_d, _d)
UD, DU = np.kron(_u, _d), np.kron(_d, _u)
def ket(a, b): return np.outer(a, b.conj())

# rank-2 spherical tensor operators of the pair (equal Hilbert-Schmidt norms)
T2 = [0.5 * Ip[0] @ Ip[1],
      -0.5 * (Ip[0] @ Iz[1] + Iz[0] @ Ip[1]),
      (3 * Iz[0] @ Iz[1] - IdotI) / np.sqrt(6),
      0.5 * (Im[0] @ Iz[1] + Iz[0] @ Im[1]),
      0.5 * Im[0] @ Im[1]]

def vec(M): return M.flatten(order="F")
def comm(H): return -1j * (np.kron(np.eye(d), H) - np.kron(H.T, np.eye(d)))
def diss(L):
    LdL = L.conj().T @ L
    return np.kron(L.conj(), L) - 0.5 * np.kron(np.eye(d), LdL) - 0.5 * np.kron(LdL.T, np.eye(d))

MECH = {
    "corr_z": diss(Fz),
    "dipolar": sum(diss(T) for T in T2),
    "uncorr": sum(diss(A) for A in Ix + Iy + Iz),
    "uncorr_z": diss(Iz[0]) + diss(Iz[1]),
}

# observables whose decay rates define the relaxation times
OBS = {
    "R1": Fz,                                   # Zeeman order
    "R2": Fp,                                   # single-quantum coherence
    "RS": ket(S, S) - 0.25 * np.eye(d),         # singlet order
    "Rc": ket(S, T0),                           # singlet-T0 (long-lived) coherence
    "RZQ": ket(UD, DU),                         # zero-quantum coherence of the product basis
}

def decay_rate(Lsup, O):
    """Decay rate of operator O under superoperator Lsup, with a check that O is an eigenoperator."""
    v = vec(O); v = v / np.linalg.norm(v)
    w = Lsup @ v
    lam = np.vdot(v, w)
    resid = np.linalg.norm(w - lam * v)
    return -lam.real, resid

def unit_rates():
    """Rates of each mechanism at unit strength: {mech: {R1, R2, RS, Rc, RZQ}} (secular, delta = J = 0)."""
    return {m: {k: decay_rate(L, O)[0] for k, O in OBS.items()} for m, L in MECH.items()}

def solve_environment(T1, T2, TS):
    """Mechanism strengths (corr_z, dipolar, uncorr) that reproduce T1, T2 and T_S."""
    r = unit_rates()
    A = np.array([[r[m][k] for m in ("corr_z", "dipolar", "uncorr")] for k in ("R1", "R2", "RS")])
    g = np.linalg.solve(A, [1 / T1, 1 / T2, 1 / TS])
    if np.any(g < -1e-12):
        raise ValueError(f"no non-negative mechanism mixture reproduces these lifetimes: {g}")
    return dict(zip(("corr_z", "dipolar", "uncorr"), np.clip(g, 0, None)))

def solve_reference(T1, T2):
    """Independent-spin reference: uncorrelated isotropic + longitudinal local fields matching T1, T2."""
    r = unit_rates()
    gu = (1 / T1) / r["uncorr"]["R1"]
    gz = (1 / T2 - gu * r["uncorr"]["R2"]) / r["uncorr_z"]["R2"]
    if gz < -1e-12:
        raise ValueError("T2 > 2 T1 is not allowed")
    return {"uncorr": gu, "uncorr_z": max(gz, 0.0)}

def dissipator(gammas):
    return sum(g * MECH[m] for m, g in gammas.items())

def lifetimes(gammas):
    """Relaxation times implied by a mechanism mixture (delta = J = 0 operator rates)."""
    Ls = dissipator(gammas)
    return {k.replace("R", "T"): 1 / decay_rate(Ls, O)[0] if decay_rate(Ls, O)[0] > 0 else np.inf
            for k, O in OBS.items()}

def liouvillian(delta, J, gammas):
    H = TWOPI * J * IdotI + np.pi * delta * G
    return comm(H) + dissipator(gammas)

dL = comm(np.pi * G)          # exact d/d(delta) of the Liouvillian, delta in Hz

# ---------------------------------------------------------------- propagation
def aug(L0, t):
    """Van Loan block exponential: propagates the stacked vector [rho; d rho/d delta]."""
    M = np.zeros((2 * D, 2 * D), complex)
    M[:D, :D] = L0; M[D:, D:] = L0; M[D:, :D] = dL
    return expm(M * t)

def aug_unitary(U):
    P = np.kron(U.conj(), U)
    Z = np.zeros((D, D), complex)
    return np.block([[P, Z], [Z, P]])

def pulse_collective(th, ph):
    return aug_unitary(expm(-1j * th * (np.cos(ph) * Fx + np.sin(ph) * Fy)))

def pulse_selective(th0, ph0, th1, ph1):
    H = th0 * (np.cos(ph0) * Ix[0] + np.sin(ph0) * Iy[0]) + th1 * (np.cos(ph1) * Ix[1] + np.sin(ph1) * Iy[1])
    return aug_unitary(expm(-1j * H))

RHO0 = np.concatenate([vec(Fz), np.zeros(D, complex)])   # high-temperature deviation state
MEAS_COLL = [vec(Fp.T)]                                    # inductive (collective) receiver
MEAS_INDIV = [vec(Ip[0].T), vec(Ip[1].T)]                  # two frequency-resolved receivers
NF, DT = 60, 0.01          # 60 samples, 0.6 s (Markovian comparison of Appendix D); static_runs.py uses 360

def fid_fisher(Y, L0, meas, Ffid=None):
    Ffid = aug(L0, DT) if Ffid is None else Ffid
    tot = 0.0
    for _ in range(NF):
        for m in meas:
            tot += 2 * abs(m @ Y[D:]) ** 2
        Y = Ffid @ Y
    return tot
