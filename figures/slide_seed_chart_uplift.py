"""Uplift version of the slide chart: each bar = method − no filler (same problems, paired), all 598 items.
    ./py.sh slide_seed_chart_uplift.py   -> slides/seed_ensembles_uplift.pdf
"""
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plot import CAT, INK2, INK3, HERE, ROOT

S = json.load(open(ROOT / "seed_patch/results_ens/all/summary.json"))
I = S["items"]
# T=1 bars all in the same space: P(answer) within the number distribution (0..999, renormalised), which is what the
# ensembles use. The full-vocabulary P(correct) of no-filler/single-seed/dots runs is 1-3% lower (number mass 0.97-0.99).
D = ROOT / "seed_patch/results_ens/all/dist"
for s in I:
    z = np.load(D / f"{s['idx']:04d}.npz")
    a_ = int(json.loads(next(l for l in open(D.parent / "rows.jsonl") if f'"idx": {s["idx"]},' in l))["answer"]) if "answer" not in s else s["answer"]
    for host, key in (("none", "none_num"), ("dots100", "dots100_num")):
        x = z[f"{host}|base"].astype(np.float64); x -= np.logaddexp.reduce(x)
        s[key] = float(np.exp(x[a_]))
mpl.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 5.4,
                     "ytick.labelsize": 6.5, "lines.linewidth": 1})


def up(k, base):
    d = np.array([s[k] - s[base] for s in I])
    return 100 * d.mean(), 100 * d.std(ddof=1) / np.sqrt(len(d))


NSEL = [16, 50, 100]
shade = dict(zip(NSEL, [0.35, 0.65, 1.0]))
fixed = {"dots100_num", "dots100_greedy"}
t1 = [("single", "single\nseed", CAT[1]), ("logitavg", "log-prob\navg", CAT[0]), ("mostconf", "most\nconfident", CAT[4]),
      ("bestof", "best of n\n(oracle)", INK2), ("dots100_num", "100\ndots", CAT[2])]
gr = [("single_greedy", "single\nseed", CAT[1]), ("vote", "majority\nvote", CAT[3]), ("dots100_greedy", "100\ndots", CAT[2])]

fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.05), gridspec_kw=dict(width_ratios=[len(t1), len(gr) + 0.3], wspace=0.3))
w = 0.27
for a, cats, base, title in ((axes[0], t1, "none_num", "T=1: P(correct answer)"), (axes[1], gr, "none_greedy", "greedy accuracy")):
    ref = up("dots100_num" if base == "none_num" else "dots100_greedy", base)[0]
    for i, (k, _, c) in enumerate(cats):
        for j, n in enumerate([None] if k in fixed else NSEL):
            m, e = up(k if n is None else f"ens{n}_{k}", base)
            x = i if n is None else i + (j - 1) * (w + 0.02)
            a.bar(x, m, w * (1.5 if n is None else 1), color=c, alpha=1.0 if n is None else shade[n], lw=0, zorder=2)
            a.errorbar(x, m, yerr=e, color=INK2, lw=0.6, capsize=0, zorder=3)
            if n in (None, 100):
                lab = f"{m:+.1f}" + (f"\n{m / ref:.0%}" if "dots100" not in k else "")
                a.text(x, max(m, 0) + e + 0.4, lab, ha="center", va="bottom", fontsize=5.0, color=INK2, linespacing=0.95)
    a.axhline(0, color=INK3, lw=0.6)
    a.set_xticks(range(len(cats)), [l for _, l, _ in cats])
    a.set_ylim(-2, 24); a.grid(axis="x", visible=False)
    a.set_title(title, fontsize=7.2)
axes[0].set_ylabel("uplift over no filler (points)", labelpad=2)
axes[0].legend(handles=[Patch(color=INK2, alpha=shade[n], label=f"{n} seeds") for n in NSEL], loc="upper left",
               fontsize=5.5, handlelength=1.2, borderaxespad=0.2, labelspacing=0.25)
out = HERE / "slides"; out.mkdir(exist_ok=True)
fig.savefig(out / "seed_ensembles_uplift.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out / "seed_ensembles_uplift.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
for k, b in (("ens100_single", "none_num"), ("ens100_logitavg", "none_num"), ("ens100_mostconf", "none_num"), ("ens100_bestof", "none_num"), ("dots100_num", "none_num"),
             ("ens100_single_greedy", "none_greedy"), ("ens100_vote", "none_greedy"), ("dots100_greedy", "none_greedy")):
    print(k, "%+.1f ± %.1f" % up(k, b))
