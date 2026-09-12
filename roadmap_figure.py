#!/usr/bin/env python3
"""
Roadmap / concept-map figure for the Supplemental Material.
Links the paper's theorems, exact reduction, interpretation and the two
numerical realizations (two-spin and many-body), with pointers to the defining
equations, tables and figures, and the contribution tags (i)-(v).

Figure numbers referenced here (S2/S3/S4) assume this roadmap is Fig. S1, i.e.
the FIRST figure in the Supplement. Output: figroadmap.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY="#1f3a5f"; LBLUE="#e7eef7"; PBLUE="#dbe6f4"
TAN="#f5eee0"; BROWN="#7a5427"; ORANGE="#b0601f"; GRAY="#5f5f5f"

TITLE_H, LINE_H, PAD_V = 0.40, 0.345, 0.17

N = {}
def add(name, x, y, w, title, lines, kind="thm", tag=None):
    h = 2*PAD_V + (TITLE_H if title else 0) + LINE_H*len(lines)
    N[name] = dict(x=x, y=y, w=w, h=h, title=title, lines=lines, kind=kind, tag=tag)

add("prin", 8.0, 10.15, 13.8, "Duality", [
    "A symmetry that protects a coherence from noise also hides the symmetry-breaking parameter $\\delta$ from",
    "every symmetric measurement: the accessible Fisher information $F^{C}=O(\\delta^{2})$, whatever the lifetime."],
    kind="prin")

add("thm1", 2.95, 8.20, 3.9, "Theorem 1",
    ["invariant probes;", "grading of operator space"], tag="(i)")
add("interp", 8.0, 8.20, 4.1, "Interpretation",
    ["dynamical WAY analogue;", "skew information localizes", "the resource in the dark sector"],
    kind="interp", tag="(ii)")
add("thm2", 13.20, 8.20, 3.9, "Theorem 2",
    ["decoherence-free sectors of", "any size; Hilbert-space blocks"], tag="(i)")

add("exact", 2.95, 6.30, 3.9, "Exact reduction",
    ["(exchange)   [Eq. 1]", "$F^{C}=4\\delta^{2}F_{u}(\\delta^{2})$", "only $\\delta^{2}$ is estimable"],
    tag="(iii)")
add("manybody", 13.20, 6.30, 3.9, "Many-body dark states",
    ["$\\bigotimes N/2$ singlet pairs;", "total spin zero"], kind="num", tag="(v)")

add("twospin", 2.95, 4.65, 3.9, "Two-spin realization", ["[Eq. 2]"],
    kind="num", tag="(v)")
add("mbqfi", 13.20, 4.35, 3.9, "",
    ["Inaccessible quantum Fisher", "information grows with $N$:",
     "$F\\propto N$ (adjacent), $N^{3}$ (spanning)", "[Fig. S4]"], kind="num")

add("collective", 2.95, 2.95, 3.9, "",
    ["Collective readout: no advantage", "(ratio $\\to 0.51$); reference",
     "globally converged  [Fig. 2(a), S2]"], kind="num")
add("advquant", 8.25, 2.55, 5.4, "Quantify the advantage",
    ["$\\bullet$ window law, peak $(P_c{+}1)/4$   [Eqs. 5, 6]",
     "$\\bullet$ resource is $T_c$, not $T_S$   [Table I]",
     "$\\bullet$ robust to RF, $B_0$ errors   [Fig. S3]"], kind="num", tag="(iv)")
add("breaksym", 2.95, 1.35, 3.9, "Break the symmetry",
    ["individual addressing $\\to$", "bounded window  [Fig. 2(b)]"], kind="num")

EDGES=[("prin","thm1"),("prin","interp"),("prin","thm2"),
       ("thm1","exact"),("exact","twospin"),("twospin","collective"),
       ("collective","breaksym"),("breaksym","advquant"),
       ("thm2","manybody"),("manybody","mbqfi")]

STYLE={"prin":dict(fc=PBLUE,ec=NAVY,lw=1.7,tc=NAVY),
       "thm":dict(fc=LBLUE,ec=NAVY,lw=1.0,tc=NAVY),
       "interp":dict(fc="#eef3fa",ec=NAVY,lw=1.0,tc=NAVY,dashed=True),
       "num":dict(fc=TAN,ec=BROWN,lw=1.0,tc="#2a2118")}

def clip(cx,cy,w,h,tx,ty):
    dx,dy=tx-cx,ty-cy
    if dx==0 and dy==0: return cx,cy
    sx=(w/2)/abs(dx) if dx else 1e9
    sy=(h/2)/abs(dy) if dy else 1e9
    s=min(sx,sy); return cx+dx*s, cy+dy*s

fig,ax=plt.subplots(figsize=(7.15,5.0))
ax.set_xlim(0,16.2); ax.set_ylim(0,11.0); ax.set_aspect("equal"); ax.axis("off")

for a,b in EDGES:
    A,B=N[a],N[b]
    x0,y0=clip(A["x"],A["y"],A["w"],A["h"],B["x"],B["y"])
    x1,y1=clip(B["x"],B["y"],B["w"],B["h"],A["x"],A["y"])
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",
                 mutation_scale=11,lw=1.1,color=GRAY,shrinkA=0,shrinkB=0,zorder=1))

for d in N.values():
    st=STYLE[d["kind"]]
    ax.add_patch(FancyBboxPatch((d["x"]-d["w"]/2,d["y"]-d["h"]/2),d["w"],d["h"],
        boxstyle="round,pad=0.02,rounding_size=0.12",fc=st["fc"],ec=st["ec"],
        lw=st["lw"],zorder=2,linestyle=(":" if st.get("dashed") else "-")))
    y=d["y"]+d["h"]/2-PAD_V
    if d["title"]:
        ax.text(d["x"],y-TITLE_H/2+0.02,d["title"],ha="center",va="center",
                fontsize=8.0,fontweight="bold",color=st["tc"],zorder=3)
        y-=TITLE_H
    for ln in d["lines"]:
        ax.text(d["x"],y-LINE_H/2+0.01,ln,ha="center",va="center",
                fontsize=6.9,color=st["tc"],zorder=3)
        y-=LINE_H
    if d["tag"]:
        ax.text(d["x"]+d["w"]/2-0.12,d["y"]+d["h"]/2+0.05,d["tag"],
                ha="right",va="bottom",fontsize=7.4,fontweight="bold",
                color=ORANGE,zorder=4)

ax.text(8.1,0.32,
    "Blue: theory   $\\cdot$   tan: numerics   $\\cdot$   "
    "$(i)$ selection rule   $(ii)$ WAY analogue   $(iii)$ exact reduction   "
    "$(iv)$ window law and resource   $(v)$ numerical confirmation",
    ha="center",va="center",fontsize=6.6,color=GRAY)

plt.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01)
fig.savefig("figroadmap.pdf",bbox_inches="tight",pad_inches=0.03)
fig.savefig("figroadmap_preview.png",dpi=200,bbox_inches="tight",pad_inches=0.03)
print("wrote figroadmap.pdf")
