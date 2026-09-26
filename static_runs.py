"""
static_runs.py -- matched-T1 comparison with field inhomogeneity as static common-mode offsets.

Both limits share the homogeneous relaxation: intra-pair dipolar coupling plus weak independent
fields, fitted to T1 = T2,hom = 1.2 s and T_S = 5 s (extreme narrowing, so T2,hom = T1).
  A  homogeneous only               T2* = T2,hom = 1.2 s
  B  plus static inhomogeneity      T2* = 0.2 s  (Lorentzian half-width 4.17 s^-1)
Inhomogeneity: every molecule sees H + Delta*Fz, with Delta Lorentzian across the sample. Every
Liouvillian used here conserves coherence order p, so a free period tau multiplies order p by
exp(-i Delta p tau). Tracking p through each free period gives every pathway a phase-time
Phi = sum_k p_k tau_k, and the sample average of exp(-i Delta Phi) is exp(-Gamma |Phi|). This is exact
(no quadrature over Delta). Echoes, which return Phi to 0, refocus the inhomogeneity; a Markovian
dephasing model cannot do that. Ideal collective pi pulses map p -> -p, so echo trains add no pathways.
Acquisition: 360 samples at 10 ms (3.6 s = 3 T2* in limit A), Nyquist frequency 50 Hz.

Usage:
  python static_runs.py sel A|B P,R,Z  d1 d2 ...   symmetry-breaking sweep
  python static_runs.py sym A|B P,R    d1 d2 ...   symmetric sweep (P with echo trains, R four-pulse)
  python static_runs.py refine A|B     d1 d2 ...   extra starts for P, warm-started from every optimum
"""
import sys, os, json, datetime
import numpy as np
import model as M
import protocols as Pr

J = 10.0
T1, T2H, TS, T2STAR = 1.2, 1.2, 5.0, 0.2
G_HOM = M.solve_environment(T1, T2H, TS)          # dipolar + weak independent fields
G_REF = M.solve_reference(T1, T2H)                 # independent isotropic fields
GAMMA = {"A": 0.0, "B": 1 / T2STAR - 1 / T2H}
ARMS = {"P": (J, G_HOM), "R": (0.0, G_REF), "Z": (0.0, G_HOM)}
NF, DT = 360, 0.01
TK = np.arange(NF) * DT
GRID = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
ECHO = [0, 4]
N_SEL, N_SYM0, N_COMBO = 4, 5, 2          # values used for the paper

_m = np.array([1, 0, 0, -1])
ORDER = np.array([_m[a] - _m[b] for b in range(4) for a in range(4)])   # vec index a + 4b
ORD2 = np.concatenate([ORDER, ORDER])
MASK = {q: (ORD2 == q).astype(float) for q in range(-2, 3)}
QDET = -1                                                                # <I+>, <F+> read order -1

def fid_rows(L0, meas):
    """Per receiver: rows R_k (signal) and S_k (derivative) acting on the post-d2 augmented state."""
    Fa = M.aug(L0, DT); G = np.eye(2 * M.D, dtype=complex)
    R = [[] for _ in meas]; S = [[] for _ in meas]
    for _ in range(NF):
        for i, m in enumerate(meas):
            R[i].append(m @ G[:M.D, :M.D]); S[i].append(m @ G[M.D:, :M.D])
        G = Fa @ G
    return [(np.array(R[i]), np.array(S[i])) for i in range(len(meas))]

def fisher_paths(V, phi, rows, Gam):
    W = np.exp(-Gam * np.abs(phi[None, :] + QDET * TK[:, None]))
    tot = 0.0
    for R, S in rows:
        D = S @ V[:M.D] + R @ V[M.D:]
        tot += 2 * np.sum(np.abs((W * D).sum(axis=1)) ** 2)
    return tot

def fisher_selective_static(p, L0, rows, Gam):
    U1, U2, U3 = (M.pulse_selective(*p[0:4]), M.pulse_selective(*p[5:9]), M.pulse_selective(*p[10:14]))
    d1, ts, d2 = p[4], p[9], p[14]
    Y = M.aug(L0, d1) @ (U1 @ M.RHO0)
    Fs = M.aug(L0, ts)
    cols, phi = [], []
    for p1 in range(-2, 3):
        v = MASK[p1] * Y
        if not v.any(): continue
        v = Fs @ (U2 @ v)
        for p2 in range(-2, 3):
            w = MASK[p2] * v
            if not w.any(): continue
            cols.append(MASK[QDET] * (U3 @ w)); phi.append(p1 * d1 + p2 * ts + QDET * d2)
    V = M.aug(L0, d2) @ np.array(cols).T
    return fisher_paths(V, np.array(phi), rows, Gam)

def fisher_collective_static(p, L0, n1, n2, rows, Gam):
    th1, ph1, th2, ph2, d1, te, ts, th3, ph3, d2, th4, ph4 = p
    P2, P3, P4 = M.pulse_collective(th2, ph2), M.pulse_collective(th3, ph3), M.pulse_collective(th4, ph4)
    if n1 or n2:
        Fe = M.aug(L0, te); E = Fe @ Pr.PI_X @ Fe
        E1, E2 = np.linalg.matrix_power(E, n1), np.linalg.matrix_power(E, n2)
    Y = M.aug(L0, d1) @ (M.pulse_collective(th1, ph1) @ M.RHO0)
    Fs, F2 = M.aug(L0, ts), M.aug(L0, d2)
    if Gam == 0.0:                      # no inhomogeneity: one pathway, direct propagation
        v = E1 @ Y if n1 else Y
        v = P3 @ (Fs @ (P2 @ v))
        if n2: v = E2 @ v
        return fisher_paths((P4 @ (F2 @ v))[:, None], np.zeros(1), rows, 0.0)
    s2 = (-1) ** n2
    cols, phi = [], []
    for p1 in range(-2, 3):
        v = MASK[p1] * Y
        if not v.any(): continue
        if n1: v = E1 @ v
        v = Fs @ (P2 @ v)
        for p2 in range(-2, 3):
            w = MASK[p2] * v
            if not w.any(): continue
            w = P3 @ w
            for p3 in range(-2, 3):
                x = MASK[p3] * w
                if not x.any(): continue
                if n2: x = E2 @ x
                cols.append(MASK[QDET] * (P4 @ (F2 @ x))); phi.append(p1 * d1 + p2 * ts + s2 * p3 * d2)
    return fisher_paths(np.array(cols).T, np.array(phi), rows, Gam)

# ------------------------------------------------------------------ driver
def load(task):
    f = f"results/{task}.json"
    return json.load(open(f)) if os.path.exists(f) else {"task": task, "points": {}}

def save(task, data):
    os.makedirs("results", exist_ok=True)
    data["settings"] = dict(J=J, T1=T1, T2hom=T2H, TS=TS, T2star=T2STAR, GAMMA=GAMMA, G_HOM=G_HOM, G_REF=G_REF,
                            NF=NF, DT=DT, ECHO=ECHO, N_SEL=N_SEL, N_SYM0=N_SYM0, N_COMBO=N_COMBO,
                            updated=datetime.datetime.now().isoformat(timespec="seconds"))
    json.dump(data, open(f"results/{task}.json", "w"), indent=1)

def setup(delta, arm, meas):
    Jj, g = ARMS[arm]; L0 = M.liouvillian(delta, Jj, g)
    return L0, fid_rows(L0, meas)

def run_sel(limit, arms, deltas):
    task = f"static_sel_{limit}"; data = load(task); Gam = GAMMA[limit]
    for d in deltas:
        rec = data["points"].get(str(d), {})
        for a in arms:
            L0, rows = setup(d, a, M.MEAS_INDIV)
            f = lambda q: fisher_selective_static(q, L0, rows, Gam)
            b, bp, vals = Pr.optimise(f, Pr.BND_S, Pr.STORE_S, N_SEL, int(d * 100) + 71, data.get(f"warm_{a}"))
            rec[a] = {"F": b, "restarts": sorted(vals, reverse=True), "params": list(bp)}
            data[f"warm_{a}"] = list(bp)
        data["points"][str(d)] = rec; save(task, data)
        print(task, d, {a: f"{rec[a]['F']:.4g}" for a in arms if a in rec}, flush=True)

def refine(limit, deltas, n_cold):
    task = f"static_sel_{limit}"; data = load(task); Gam = GAMMA[limit]
    other = load(f"static_sel_{'B' if limit == 'A' else 'A'}")
    for d in deltas:
        rec = data["points"][str(d)]
        L0, rows = setup(d, "P", M.MEAS_INDIV)
        f = lambda q: fisher_selective_static(q, L0, rows, Gam)
        best, bp, _ = Pr.optimise(f, Pr.BND_S, Pr.STORE_S, n_cold, int(d * 100) + 503 + int(os.environ.get("SEED", "0")))
        warms = [r["P"]["params"] for r in data["points"].values() if "P" in r]
        warms += [r["P"]["params"] for k, r in other["points"].items() if k == str(d) and "P" in r]
        for w in warms:
            b2, bp2, _ = Pr.optimise(f, Pr.BND_S, Pr.STORE_S, 0, 0, warm=w)
            if b2 > best: best, bp = b2, bp2
        if best > rec["P"]["F"]:
            rec["P"]["F"], rec["P"]["params"] = best, list(bp)
        rec["P"]["refined"] = True; save(task, data)
        print("refine", task, d, f"{rec['P']['F']:.4g}", flush=True)

def run_sym(limit, arms, deltas):
    task = f"static_sym_{limit}"; data = load(task); Gam = GAMMA[limit]
    SEEDOFF = int(os.environ.get("SEED", "0"))
    for d in deltas:
        rec = data["points"].get(str(d), {})
        if "P" in arms:
            L0, rows = setup(d, "P", M.MEAS_COLL); combos = {}
            for n1 in ECHO:
                for n2 in ECHO:
                    ns = N_SYM0 if (n1, n2) == (0, 0) else N_COMBO
                    warm = data.get("warm00") if (n1, n2) == (0, 0) else None
                    b, bp, _ = Pr.optimise(lambda q: fisher_collective_static(q, L0, n1, n2, rows, Gam), Pr.BND_C,
                                           Pr.STORE_C, ns, int(d * 100) + 5 * n1 + n2 + 13 + SEEDOFF, warm)
                    combos[f"{n1},{n2}"] = max(b, rec.get("P", {}).get("combos", {}).get(f"{n1},{n2}", 0.0))
                    if (n1, n2) == (0, 0): data["warm00"] = list(bp)
            best = max(combos, key=combos.get)
            rec["P"] = {"four_pulse": combos["0,0"], "echo_best": combos[best], "echo_best_n": best, "combos": combos}
        if "R" in arms:
            L0, rows = setup(d, "R", M.MEAS_COLL)
            b, bp, vals = Pr.optimise(lambda q: fisher_collective_static(q, L0, 0, 0, rows, Gam), Pr.BND_C,
                                      Pr.STORE_C, N_SYM0 + 2, int(d * 100) + 29 + SEEDOFF, data.get("warmR"))
            data["warmR"] = list(bp); rec["R"] = {"F": max(b, rec.get("R", {}).get("F", 0.0)), "restarts": sorted(vals, reverse=True)}
            if d in (1.0, 5.0):
                rec["R"]["echo_check"] = max(Pr.optimise(lambda q: fisher_collective_static(q, L0, n, n, rows, Gam),
                                                         Pr.BND_C, Pr.STORE_C, 2, 7 + n)[0] for n in (2, 4))
        data["points"][str(d)] = rec; save(task, data)
        print(task, d, {k: (f"{v['echo_best']:.4g} (four-pulse {v['four_pulse']:.4g}, n={v['echo_best_n']})" if k == "P"
                            else f"{v['F']:.4g}") for k, v in rec.items()}, flush=True)

if __name__ == "__main__":
    mode, limit = sys.argv[1], sys.argv[2]
    if mode == "sel":
        run_sel(limit, sys.argv[3].split(","), [float(x) for x in sys.argv[4:]] or GRID)
    elif mode == "sym":
        run_sym(limit, sys.argv[3].split(","), [float(x) for x in sys.argv[4:]] or GRID)
    elif mode == "refine":
        refine(limit, [float(x) for x in sys.argv[3:]] or GRID, int(os.environ.get("NCOLD", "4")))
