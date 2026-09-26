"""Supplement S5: semi-analytic estimate A = sin^2(2 theta) (T_c^eff / T2*)^2 of the symmetry-breaking gain,
compared with the optimized values of limit B. Reads results/static_sel_B.json and results/lifetimes_hom.json;
writes figs/figS3_gain.pdf."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
J, T2s = 10.0, 0.2
B = json.load(open("results/static_sel_B.json"))["points"]; lt = json.load(open("results/lifetimes_hom.json"))
RB = B["0.5"]["R"]["F"]
ds = sorted(float(k) for k in B if "P" in B[k] and float(k) <= 20)
x = np.array(ds) / J; num = np.array([B[str(d)]["P"]["F"] / RB for d in ds])
law = lambda xx: xx**2 / (1 + xx**2) * (np.interp(xx, lt["dj"], lt["Tc"]) / T2s)**2
print(" delta/J   optimized   estimate   optimized/estimate")
for xi, ni in zip(x, num):
    print(f"  {xi:5.2f}   {ni:8.3g}   {law(xi):8.3g}   {ni/law(xi):6.2f}")
xx = np.logspace(-2, np.log10(2), 100)
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, a = plt.subplots(figsize=(3.4, 2.6))
a.loglog(x, num, "o", c="C3", ms=4, label="optimized (limit B)")
a.loglog(xx, law(xx), "-", c="C3", lw=1, label=r"$\sin^22\theta\,(T_c^{\rm eff}/T_2^\ast)^2$")
a.axhline(1, c="k", lw=0.6); a.set_xlabel(r"$\delta/J$"); a.set_ylabel("gain over reference"); a.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig("figs/figS3_gain.pdf")
