"""Fig. 4 and the values of Tables I, II and V: limits A and B at matched T1 (results/static_*.json) and the
Markovian comparison of Appendix D (results/selective_cm.json, collective_P.json, collective_R.json).
Writes figs/fig_static.pdf and prints every tabulated ratio."""
import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
J = 10.0
ld = lambda t: json.load(open(f"results/{t}.json"))["points"]
A, B, SB, cm, cP, cR = (ld(t) for t in ("static_sel_A", "static_sel_B", "static_sym_B", "selective_cm", "collective_P", "collective_R"))
RA, RB = A["0.5"]["R"]["F"], B["0.5"]["R"]["F"]
dA = sorted(float(k) for k in A if "P" in A[k]); dB = sorted(float(k) for k in B if "P" in B[k]); dS = sorted(float(k) for k in SB)
rA = {d: A[str(d)]["P"]["F"] / RA for d in dA}; rB = {d: B[str(d)]["P"]["F"] / RB for d in dB}
rM = {float(k): v["P"]["F"] / v["R"]["F"] for k, v in cm.items() if float(k) <= 20}
rZ = {float(k): v["Z"]["F"] / RB for k, v in B.items() if "Z" in v}
sB = {d: SB[str(d)]["P"]["echo_best"] / SB[str(d)]["R"]["F"] for d in dS}
s4 = {d: SB[str(d)]["P"]["four_pulse"] / SB[str(d)]["R"]["F"] for d in dS}
sM = {float(k): cP[k]["echo_best"] / cR[k]["F"] for k in cP}
SA = ld("static_sym_A"); sA = {float(k): v["P"]["echo_best"] / v["R"]["F"] for k, v in SA.items()}
fmt = lambda x: f"{x:.3g}"
sel = [f"{d/J:g} & {fmt(rA[d]) if d in rA else '--'} & {fmt(rB[d])} & {fmt(rM[d])} & {fmt(rZ[d]) if d in rZ else '--'}\\\\" for d in dB]
sym = [f"{d/J:g} & {fmt(sA[d]) if d in sA else '--'} & {s4[d]:.2g} & {fmt(sB[d])} ({SB[str(d)]['P']['echo_best_n']}) & {sM[d]:.2g}\\\\" for d in dS]
open("static_tables.tex", "w").write(
 "\\begin{table}[h]\\centering\n\\caption{Symmetry-breaking control and detection at matched $T_1$: protected pair relative to the reference. "
 f"References: $F_R={RA:.0f}$ (limit~A) and ${RB:.2f}$ (limit~B), flat in $\\delta$.}}\n\\label{{tab:static_sel}}\n"
 "\\begin{tabular}{c|cccc}\\hline\n$\\delta/J$ & limit A (dipolar only) & limit B (+ static broadening) & Markovian common-mode & uncoupled pair, limit B\\\\\\hline\n"
 + "\n".join(sel) + "\n\\hline\n\\end{tabular}\n\\end{table}\n\n"
 "\\begin{table}[h]\\centering\n\\caption{Symmetric control and inductive detection relative to the reference with the same readout. "
 "Echo lengths $n_{1,2}\\in\\{0,4\\}$. Last column: the Markovian result of Table~\\ref{tab:sym}.}\n\\label{tab:static_sym}\n"
 "\\begin{tabular}{c|cccc}\\hline\n$\\delta/J$ & limit A, echo trains & limit B, four-pulse & limit B, echo trains $(n_1,n_2)$ & Markovian\\\\\\hline\n" + "\n".join(sym) + "\n\\hline\n\\end{tabular}\n\\end{table}\n")
print("\n".join(sel)); print("\n".join(sym))
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
a = ax[0]
a.loglog([d / J for d in sorted(sM)], [sM[d] for d in sorted(sM)], "o:", c="0.45", ms=3.5, label="Markovian common-mode")
a.loglog([d / J for d in dS], [sB[d] for d in dS], "o-", c="C0", ms=4, label="limit B, echo trains")
a.loglog([d / J for d in dS], [s4[d] for d in dS], "s--", c="C0", mfc="none", ms=4, label="limit B, four-pulse")
a.loglog([d / J for d in sorted(sA)], [sA[d] for d in sorted(sA)], "^-", c="C2", ms=4, label="limit A, echo trains")
a.axhline(1, c="k", lw=0.6); a.set_ylim(1e-3, 80); a.set_xlabel(r"$\delta/J$"); a.set_ylabel("protected / reference")
a.set_title("(a) symmetric control and readout", fontsize=9); a.legend(fontsize=6.5, frameon=False, loc="upper left")
a = ax[1]
a.loglog([d / J for d in dA], [rA[d] for d in dA], "^-", c="C2", ms=4, label="limit A (dipolar only)")
a.loglog([d / J for d in dB], [rB[d] for d in dB], "o-", c="C3", ms=4, label="limit B (static broadening)")
a.loglog([d / J for d in sorted(rM)], [rM[d] for d in sorted(rM)], ":", c="0.45", label="Markovian model")
a.loglog([d / J for d in sorted(rZ)], [rZ[d] for d in sorted(rZ)], "x", c="k", ms=5, label="uncoupled pair (B)")
a.axhline(1, c="k", lw=0.6); a.set_ylim(5e-3, 200); a.set_xlabel(r"$\delta/J$")
a.set_title("(b) symmetry-breaking control and readout", fontsize=9); a.legend(fontsize=6.5, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("figs/fig_static.pdf"); fig.savefig("figs/fig_static.png", dpi=110)

print("\nMarkovian comparison (Appendix D): delta/J | symmetry-breaking Markovian, limit B | symmetric Markovian, limit B")
for d in sorted(set(rM) | set(sM)):
    g = lambda m: f"{m[d]:.3g}" if d in m else "--"
    print(f"  {d/J:5.2f} | {g(rM):>6s} {g(rB):>6s} | {g(sM):>6s} {g(sB):>6s}")

# Gains per unit experiment time (Sec. VI C): sequence = d1 + storage + d2 + acquisition, plus a
# repolarization delay of 5 T1 between shots, for the optimal controls stored with each value.
TACQ, TREP = 3.6, 5 * 1.2
dur = lambda p: p[4] + p[9] + p[14] + TACQ
pR = B["0.5"]["R"]["params"]; TR = dur(pR) + TREP
fac = {d: TR / (dur(B[str(d)]["P"]["params"]) + TREP) for d in dB}
st = [B[str(d)]["P"]["params"][9] for d in dB]
print(f"\nlimit B, per unit time: storage {min(st):.1f}-{max(st):.1f} s, gain per unit time / gain per acquisition = "
      f"{min(fac.values()):.2f}-{max(fac.values()):.2f}")
