"""
protocols.py -- control ansaetze and the multistart optimiser.

Collective (symmetric) ansatz, 12 continuous parameters plus two integers:
    P(th1,ph1) - d1 - [echo]^n1 - P(th2,ph2) - storage - P(th3,ph3) - [echo]^n2 - d2 - P(th4,ph4) - FID
    echo = tau_e - pi_x - tau_e with a collective pi pulse. n1 = n2 = 0 is the
    four-pulse ansatz of Table II.
Selective (symmetry-breaking) ansatz, 15 parameters:
    Q(a0) - d1 - Q(a1) - storage - Q(a2) - d2 - FID, each Q an independent pulse on each spin.
Optimisation: L-BFGS-B with finite-difference gradients, several random starts
(half of them with a long initial storage) plus an optional warm start.
"""
import numpy as np
from scipy.optimize import minimize
import model as M

BND_C = [(-4, 4)] * 4 + [(0, 0.2), (0.005, 0.1), (0, 8.0)] + [(-4, 4)] * 2 + [(0, 0.2)] + [(-4, 4)] * 2
BND_S = [(-4, 4)] * 4 + [(0, 0.2)] + [(-4, 4)] * 4 + [(0, 8.0)] + [(-4, 4)] * 4 + [(0, 0.2)]
STORE_C, STORE_S = 6, 9
PI_X = M.pulse_collective(np.pi, 0.0)

def fisher_collective(p, L0, n1, n2, Ffid):
    th1, ph1, th2, ph2, d1, te, ts, th3, ph3, d2, th4, ph4 = p
    Y = M.pulse_collective(th1, ph1) @ M.RHO0
    Y = M.aug(L0, d1) @ Y
    if n1 or n2:
        Fe = M.aug(L0, te); E = Fe @ PI_X @ Fe
    if n1: Y = np.linalg.matrix_power(E, n1) @ Y
    Y = M.pulse_collective(th2, ph2) @ Y
    Y = M.aug(L0, ts) @ Y
    Y = M.pulse_collective(th3, ph3) @ Y
    if n2: Y = np.linalg.matrix_power(E, n2) @ Y
    Y = M.aug(L0, d2) @ Y
    Y = M.pulse_collective(th4, ph4) @ Y
    return M.fid_fisher(Y, L0, M.MEAS_COLL, Ffid)

def fisher_selective(p, L0, Ffid, meas=None):
    a = p
    Y = M.pulse_selective(*a[0:4]) @ M.RHO0
    Y = M.aug(L0, a[4]) @ Y
    Y = M.pulse_selective(*a[5:9]) @ Y
    Y = M.aug(L0, a[9]) @ Y
    Y = M.pulse_selective(*a[10:14]) @ Y
    Y = M.aug(L0, a[14]) @ Y
    return M.fid_fisher(Y, L0, M.MEAS_INDIV if meas is None else meas, Ffid)

def optimise(fun, bnds, store_idx, n_start, seed, warm=None, store_range=(1.0, 5.0), maxiter=200):
    rng = np.random.default_rng(seed)
    starts = [np.array(warm)] if warm is not None else []
    for s in range(n_start):
        p = np.array([rng.uniform(lo, hi) for lo, hi in bnds])
        if s % 2 == 1:
            p[store_idx] = rng.uniform(*store_range)
        starts.append(p)
    best, bp, vals = -1.0, None, []
    for p0 in starts:
        r = minimize(lambda q: -fun(q), p0, method="L-BFGS-B", bounds=bnds,
                     options={"maxiter": maxiter, "ftol": 1e-13})
        vals.append(-r.fun)
        if -r.fun > best:
            best, bp = -r.fun, r.x
    return best, bp, vals
