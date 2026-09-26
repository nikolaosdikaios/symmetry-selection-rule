"""
redfield.py -- intra-pair dipolar relaxation from Redfield theory (isotropic rotational diffusion).

High field, secular in the Zeeman interaction, high temperature, no dynamic frequency shifts:
    R_dd = j(0) D[T20] + j(w0) (D[T2,+1] + D[T2,-1]) + j(2 w0) (D[T2,+2] + D[T2,-2]),
    j(w) = tau_c / (1 + (w tau_c)^2), normalised so that the q = 0 weight is 1.
In extreme narrowing all weights are equal and R_dd is the isotropic dissipator of model.py.
Each D[T2,q] conserves coherence order, so the exact static-offset treatment of static_runs.py applies.

Environment for each tau_c: dipolar strength and weak independent fields fitted to T1 = 1.2 s and
T_S = 5 s; T2,hom then follows from tau_c. Static broadening brings T2* to 0.2 s where T2,hom > 0.2 s.
The reference has independent fields with the same T1 and T2,hom and sees the same broadening.

Usage: python redfield.py rates            rate table versus tau_c, and the extreme-narrowing check
       python redfield.py run TAU_NS ...    uncoupled-control test (Z, R, P at delta = 0.5 and 5 Hz)
"""
import sys, os, json
import numpy as np
import model as M
import protocols as Pr
import static_runs as SR

NU0 = 500e6
QS = [2, 1, 0, -1, -2]                       # order of model.T2

def dd_redfield(tau_c):
    w0 = 2 * np.pi * NU0
    j = lambda w: 1.0 / (1 + (w * tau_c) ** 2)
    return sum(j(abs(q) * w0) * M.diss(T) for q, T in zip(QS, M.T2))

def unit_rates(Lsup):
    odd = M.Ip[0] - M.Ip[1]                  # exchange-odd single-spin coherence (singlet-triplet part)
    obs = dict(M.OBS, Rodd=odd)
    return {k: M.decay_rate(Lsup, O)[0] for k, O in obs.items()}

def environment(tau_c, T1=1.2, TS=5.0, T2star=0.2):
    R = dd_redfield(tau_c); r = unit_rates(R); ru = unit_rates(M.MECH["uncorr"])
    gu = (1 / TS) / ru["RS"]
    gd = (1 / T1 - gu * ru["R1"]) / r["R1"]
    T2hom = 1 / (gd * r["R2"] + gu * ru["R2"])
    Gam = max(0.0, 1 / T2star - 1 / T2hom)
    Ldiss = gd * R + gu * M.MECH["uncorr"]
    ref = M.solve_reference(T1, T2hom)
    return dict(gd=gd, gu=gu, T2hom=T2hom, Gam=Gam, Ldiss=Ldiss, ref=ref,
                Tc=1 / M.decay_rate(Ldiss, M.OBS["Rc"])[0], units=r)

def liouv(delta, J, Ldiss):
    H = M.TWOPI * J * M.IdotI + np.pi * delta * M.G
    return M.comm(H) + Ldiss

def best(L0, Gam, n_start, seed, warm=None):
    rows = SR.fid_rows(L0, M.MEAS_INDIV)
    return Pr.optimise(lambda q: SR.fisher_selective_static(q, L0, rows, Gam), Pr.BND_S, Pr.STORE_S,
                       n_start, seed, warm)[0]

if __name__ == "__main__":
    out = json.load(open("results/redfield.json")) if os.path.exists("results/redfield.json") else {}
    if sys.argv[1] == "rates":
        iso = M.MECH["dipolar"]
        print("extreme-narrowing check |R(1e-15) - isotropic| / |isotropic| =",
              f"{np.linalg.norm(dd_redfield(1e-15) - iso) / np.linalg.norm(iso):.1e}")
        print(" tau_c (ns)  w0*tau_c   R1/R2   Rc/R2  Rodd/R2   T2hom (s)  Tc (s)  Tc/T2hom")
        rows = {}
        for tc in [0.01, 0.1, 0.3, 1.0, 2.0]:
            e = environment(tc * 1e-9); u = e["units"]
            rows[tc] = dict(R1=u["R1"] / u["R2"], Rc=u["Rc"] / u["R2"], Rodd=u["Rodd"] / u["R2"], T2hom=e["T2hom"], Tc=e["Tc"])
            print(f"  {tc:6.2f}   {2*np.pi*NU0*tc*1e-9:7.3f}   {u['R1']/u['R2']:.3f}   {u['Rc']/u['R2']:.3f}   {u['Rodd']/u['R2']:.3f}"
                  f"     {e['T2hom']:.3f}   {e['Tc']:.3f}   {e['Tc']/e['T2hom']:.2f}")
        out["rates"] = rows
    elif sys.argv[1] == "run":
        warmB = json.load(open("results/static_sel_B.json"))["points"]
        for tns in [float(x) for x in sys.argv[2:]]:
            e = environment(tns * 1e-9); rec = {"T2hom": e["T2hom"], "Gam": e["Gam"], "Tc": e["Tc"]}
            Lr = liouv(0.5, 0.0, M.dissipator(e["ref"]))
            rec["R"] = best(Lr, e["Gam"], 6, 5)
            for d in (0.5, 5.0):
                rec[f"Z{d}"] = best(liouv(d, 0.0, e["Ldiss"]), e["Gam"], 6, 7, warmB[str(d)].get("Z", {}).get("params"))
                rec[f"P{d}"] = best(liouv(d, 10.0, e["Ldiss"]), e["Gam"], 6, 9, warmB[str(d)]["P"]["params"])
            out[str(tns)] = rec
            print(f"tau_c = {tns} ns: T2hom = {e['T2hom']:.3f} s, T2* = {1/(1/e['T2hom']+e['Gam']):.3f} s, Tc = {e['Tc']:.2f} s | "
                  f"Z/R: {rec['Z0.5']/rec['R']:.2f} (0.5 Hz), {rec['Z5.0']/rec['R']:.2f} (5 Hz) | "
                  f"P/R: {rec['P0.5']/rec['R']:.2f} (0.5 Hz), {rec['P5.0']/rec['R']:.2f} (5 Hz)", flush=True)
            json.dump(out, open("results/redfield.json", "w"), indent=1)
    json.dump(out, open("results/redfield.json", "w"), indent=1)
