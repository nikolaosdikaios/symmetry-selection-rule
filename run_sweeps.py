"""
run_sweeps.py -- inhomogeneity modeled as Markovian common-mode dephasing (Appendix D comparison).
The paper uses the tasks selective_cm, collective_P and collective_R. The tasks selective_dd and
selective_z are exploratory variants and are not used in the paper.

Usage:  python run_sweeps.py <task> [delta_Hz ...]
tasks:
  selective_cm   symmetry-breaking control and detection, common-mode environment (P, R)
  selective_dd   symmetry-breaking control and detection, dipolar-limited environment (P, R)
  selective_z    uncoupled pair (J = 0) in the common-mode environment (Z)
  collective_P   symmetric control and detection, protected pair, echo-train ansatz
  collective_R   symmetric control and detection, reference, four-pulse ansatz (+ echo check)
Results are appended to results/<task>.json after every delta, with the settings used.
"""
import sys, os, json, time, datetime
import numpy as np
import model as M
import protocols as P

J = 10.0
GRID = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]          # delta in Hz (delta/J = 0.02 ... 5)
ENV = {
    "cm": dict(T1=1.2, T2=0.2, TS=5.0),    # common-mode dominated T2 (the Letter's lifetimes)
    "dd": dict(T1=0.2, T2=0.2, TS=5.0),    # intra-pair dipolar limited, extreme narrowing
}
N_SEL, N_COLL0, N_COMBO = 6, 6, 2
ECHO = [0, 2, 4, 8]

def env_gammas(name):
    e = ENV[name]
    return M.solve_environment(e["T1"], e["T2"], e["TS"]), M.solve_reference(e["T1"], e["T2"])

def load(task):
    f = f"results/{task}.json"
    return json.load(open(f)) if os.path.exists(f) else {"task": task, "points": {}}

def save(task, data):
    os.makedirs("results", exist_ok=True)
    data["settings"] = dict(J=J, NF=M.NF, DT=M.DT, ENV=ENV, N_SEL=N_SEL, N_COLL0=N_COLL0,
                            N_COMBO=N_COMBO, ECHO=ECHO, BND_C=P.BND_C, BND_S=P.BND_S,
                            updated=datetime.datetime.now().isoformat(timespec="seconds"))
    json.dump(data, open(f"results/{task}.json", "w"), indent=1)

def selective(task, deltas, env, arms):
    data = load(task); g_env, g_ref = env_gammas(env)
    gam = {"P": (J, g_env), "Z": (0.0, g_env), "R": (0.0, g_ref)}
    for delta in deltas:
        rec = data["points"].get(str(delta), {})
        for a in arms:
            Jj, g = gam[a]
            L0 = M.liouvillian(delta, Jj, g); Ff = M.aug(L0, M.DT)
            warm = data.get(f"warm_{a}")
            b, bp, vals = P.optimise(lambda q: P.fisher_selective(q, L0, Ff), P.BND_S, P.STORE_S,
                                     N_SEL, int(delta * 100) + 11, warm)
            rec[a] = {"F": b, "restarts": sorted(vals, reverse=True), "params": list(bp)}
            data[f"warm_{a}"] = list(bp)
        data["points"][str(delta)] = rec; save(task, data)
        print(task, delta, {a: f"{rec[a]['F']:.3e}" for a in arms}, flush=True)

def collective_P(deltas):
    task = "collective_P"; data = load(task); g_env, _ = env_gammas("cm")
    for delta in deltas:
        L0 = M.liouvillian(delta, J, g_env); Ff = M.aug(L0, M.DT)
        rec = {"combos": {}}
        for n1 in ECHO:
            for n2 in ECHO:
                ns = N_COLL0 if (n1, n2) == (0, 0) else N_COMBO
                warm = data.get("warm00") if (n1, n2) == (0, 0) else None
                b, bp, vals = P.optimise(lambda q: P.fisher_collective(q, L0, n1, n2, Ff), P.BND_C, P.STORE_C,
                                         ns, int(delta * 100) + 17 + 5 * n1 + n2, warm)
                rec["combos"][f"{n1},{n2}"] = {"F": b, "params": list(bp)}
                if (n1, n2) == (0, 0): data["warm00"] = list(bp); rec["four_pulse"] = b
        best = max(rec["combos"].items(), key=lambda kv: kv[1]["F"])
        rec["echo_best"] = best[1]["F"]; rec["echo_best_n"] = best[0]
        data["points"][str(delta)] = rec; save(task, data)
        print(task, delta, f"four-pulse {rec['four_pulse']:.3e}  echo {rec['echo_best']:.3e} at n={best[0]}", flush=True)

def collective_R(deltas):
    task = "collective_R"; data = load(task); _, g_ref = env_gammas("cm")
    for delta in deltas:
        L0 = M.liouvillian(delta, 0.0, g_ref); Ff = M.aug(L0, M.DT)
        b, bp, vals = P.optimise(lambda q: P.fisher_collective(q, L0, 0, 0, Ff), P.BND_C, P.STORE_C,
                                 N_COLL0 + 2, int(delta * 100) + 29, data.get("warm"))
        data["warm"] = list(bp)
        rec = {"F": b, "restarts": sorted(vals, reverse=True)}
        if delta in (1.0, 5.0):          # check that collective echo trains do not help the reference
            be = max(P.optimise(lambda q: P.fisher_collective(q, L0, n, n, Ff), P.BND_C, P.STORE_C, 2, 7 + n)[0]
                     for n in (2, 4, 8))
            rec["echo_check"] = be
        data["points"][str(delta)] = rec; save(task, data)
        print(task, delta, f"{b:.3e}", rec.get("echo_check", ""), flush=True)

if __name__ == "__main__" and sys.argv[1] != "refine":
    task = sys.argv[1]
    deltas = [float(x) for x in sys.argv[2:]] or GRID
    t0 = time.time()
    if task == "selective_cm": selective(task, deltas, "cm", ["P", "R"])
    elif task == "selective_dd": selective(task, deltas, "dd", ["P", "R"])
    elif task == "selective_z": selective(task, deltas, "cm", ["Z"])
    elif task == "collective_P": collective_P(deltas)
    elif task == "collective_R": collective_R(deltas)
    print(f"[{time.time() - t0:.0f} s]")

def refine(task, deltas, n_cold=10):
    """Extra cold starts plus warm starts from every other delta's optimum; keeps the best."""
    data = load(task)
    env = "dd" if task.endswith("_dd") else "cm"
    g_env, g_ref = env_gammas(env)
    gam = {"P": (J, g_env), "Z": (0.0, g_env), "R": (0.0, g_ref)}
    for delta in deltas:
        rec = data["points"][str(delta)]
        for a in [k for k in rec if k in gam and k != "R"]:
            Jj, g = gam[a]
            L0 = M.liouvillian(delta, Jj, g); Ff = M.aug(L0, M.DT)
            f = lambda q: P.fisher_selective(q, L0, Ff)
            best, bp, vals = P.optimise(f, P.BND_S, P.STORE_S, n_cold, int(delta * 100) + 404 + int(os.environ.get("SEED", "0")))
            for other in data["points"].values():
                if a in other:
                    b2, bp2, _ = P.optimise(f, P.BND_S, P.STORE_S, 0, 0, warm=other[a]["params"])
                    if b2 > best: best, bp = b2, bp2
            if best > rec[a]["F"]:
                rec[a]["F"], rec[a]["params"] = best, list(bp)
            rec[a]["refined_restarts"] = sorted(vals, reverse=True)
        save(task, data)
        print("refine", task, delta, {a: f"{rec[a]['F']:.3e}" for a in rec if a in gam}, flush=True)

if __name__ == "__main__" and sys.argv[1] == "refine":
    refine(sys.argv[2], [float(x) for x in sys.argv[3:]], int(os.environ.get("NCOLD", "10")))
