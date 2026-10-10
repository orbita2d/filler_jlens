"""Compact version of fig21's top row (all items, seed ensembles) sized for a 16:9 beamer slide.
    ./py.sh slide_seed_chart.py   -> slides/seed_ensembles_chart.pdf
"""
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot import CAT, INK2, INK3, HERE, ROOT

S = json.load(open(ROOT / "seed_patch/results_ens/all/summary.json"))
A = S["subsets"]["all"]
mpl.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 5.4,
                     "ytick.labelsize": 6.5, "lines.linewidth": 1})

NSEL = [16, 50, 100]
shade = dict(zip(NSEL, [0.35, 0.65, 1.0]))
fixed = {"none", "seed", "dots100", "none_greedy", "dots100_greedy"}
t1 = [("none", "no\nfiller", INK3), ("seed", "single\nseed", CAT[1]), ("logitavg", "log-prob\navg", CAT[0]),
      ("mostconf", "most\nconfident", CAT[4]), ("bestof", "best of n\n(oracle)", INK2), ("dots100", "100\ndots", CAT[2])]
gr = [("none_greedy", "no\nfiller", INK3), ("single_greedy", "single\nseed", CAT[1]), ("vote", "majority\nvote", CAT[3]),
      ("dots100_greedy", "100\ndots", CAT[2])]

fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.05), gridspec_kw=dict(width_ratios=[len(t1), len(gr) + 0.3], wspace=0.28))
w = 0.27
for a, cats, title, yl in ((axes[0], t1, "T=1: P(correct answer)", "P(correct)"), (axes[1], gr, "greedy: accuracy", "accuracy")):
    for i, (k, _, c) in enumerate(cats):
        for j, n in enumerate([None] if k in fixed else NSEL):
            d = A[k] if k in fixed else A[f"ens{n}_{k}"]
            x = i if n is None else i + (j - 1) * (w + 0.02)
            a.bar(x, d["mean"], w * (1.5 if n is None else 1), color=c, alpha=1.0 if n is None else shade[n], lw=0, zorder=2)
            a.errorbar(x, d["mean"], yerr=d["se"], color=INK2, lw=0.6, capsize=0, zorder=3)
            if n in (None, 100):
                a.text(x, d["mean"] + d["se"] + 0.012, f"{d['mean']:.2f}", ha="center", va="bottom", fontsize=5.3, color=INK2)
    a.set_xticks(range(len(cats)), [l for _, l, _ in cats])
    a.set_ylim(0, 0.8); a.grid(axis="x", visible=False); a.set_ylabel(yl, labelpad=2)
    a.set_title(title, fontsize=7.2)
axes[0].legend(handles=[Patch(color=INK2, alpha=shade[n], label=f"{n} seeds") for n in NSEL], loc="upper left",
               fontsize=5.5, handlelength=1.2, borderaxespad=0.2, labelspacing=0.25)
out = HERE / "slides"; out.mkdir(exist_ok=True)
fig.savefig(out / "seed_ensembles_chart.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out / "seed_ensembles_chart.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("n items", A["n"])
