"""Figure 21: seed ensembles over all items (../seed_patch/results_ens/all/summary.json from analyze_ens.py).

    ./py.sh plot_seed_ens.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from plot import CAT, INK2, INK3, ROOT, footnote, save

S = json.load(open(ROOT / "seed_patch/results_ens/all/summary.json"))
N = S["config"]["N"]


def fig21():
    A = S["subsets"]["all"]
    NSEL = [n for n in (16, 50, 100) if n <= N]
    shade = dict(zip(NSEL, [0.35, 0.65, 1.0]))
    t1 = [("none", "no filler,\nno seed", INK3), ("seed", "single\nseed", CAT[1]), ("logitavg", "log-prob\naverage", CAT[0]),
          ("probavg", "probability\naverage", CAT[6]), ("mostconf", "most\nconfident seed", CAT[4]),
          ("bestof", "best of n\n(oracle)", INK2), ("dots100", "100 dots,\nno seed", CAT[2])]
    gr = [("none_greedy", "no filler,\nno seed", INK3), ("single_greedy", "single\nseed", CAT[1]), ("vote", "majority\nvote", CAT[3]),
          ("dots100_greedy", "100 dots,\nno seed", CAT[2])]
    fixed = {"none", "seed", "dots100", "none_greedy", "dots100_greedy"}
    fig = plt.figure(figsize=(15, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[len(t1), len(gr) + 0.6], hspace=0.55, wspace=0.14)
    w = 0.26
    for col, (cats, title, yl) in enumerate(((t1, "T=1: probability of the correct answer", "P(correct)"),
                                             (gr, "greedy: accuracy (0/1)", "accuracy"))):
        a = fig.add_subplot(gs[0, col])
        for i, (k, _, c) in enumerate(cats):
            for j, n in enumerate([None] if k in fixed else NSEL):
                d = A[k] if k in fixed else A[f"ens{n}_{k}"]
                x = i if n is None else i + (j - 1) * (w + 0.02)
                a.bar(x, d["mean"], w * (1.4 if n is None else 1), color=c, alpha=1.0 if n is None else shade[n], lw=0, zorder=2)
                a.errorbar(x, d["mean"], yerr=d["se"], color=INK2, lw=1, capsize=0, zorder=3)
                a.text(x, d["mean"] + d["se"] + 0.012, f"{d['mean']:.2f}", ha="center", va="bottom", fontsize=6.8, color=INK2)
        a.set_xticks(range(len(cats)), [l for _, l, _ in cats], fontsize=8)
        a.set_ylim(0, 1.0); a.grid(axis="x", visible=False); a.set_ylabel(yl)
        a.set_title(f"All {A['n']} items, {title}", fontsize=9.5)
        if col == 0:
            from matplotlib.patches import Patch
            a.legend(handles=[Patch(color=INK2, alpha=shade[n], label=f"{n} seeds") for n in NSEL], loc="upper left",
                     fontsize=8, title="ensemble size", title_fontsize=8)
    # regression to the mean: by the H200 uplift bin used for selection
    b = fig.add_subplot(gs[1, :])
    B = S["uplift_bins"]
    names = list(B)
    x = np.arange(len(names))
    items = S["items"]
    edges = [tuple(float(v) for v in nm.strip("(]").split(", ")) for nm in names]
    h200 = {k: [np.mean([s[k] for s in items if lo < s["h200_uplift"] <= hi]) for lo, hi in edges] for k in ("h200_k0", "h200_dots100")}
    b.plot(x, h200["h200_k0"], color=INK3, ls=":", marker="s", ms=5, label="no filler, H200 run (selection data)")
    b.plot(x, h200["h200_dots100"], color=CAT[2], ls=":", marker="s", ms=5, label="100 dots, H200 run (selection data)")
    for k, lab, c in (("none", "no filler, no seed (this run)", INK3), ("seed", "single seed", CAT[1]),
                      ("ens100_logitavg", "log-prob average of 100", CAT[0]), ("ens50_mostconf", "most confident of 50", CAT[4]),
                      ("dots100", "100 dots (this run)", CAT[2])):
        m = np.array([B[nm][k]["mean"] for nm in names]); e = np.array([B[nm][k]["se"] for nm in names])
        b.fill_between(x, m - e, m + e, color=c, alpha=0.12, lw=0)
        b.plot(x, m, color=c, marker="o", ms=4, label=lab)
    b.set_xticks(x, [f"uplift {nm}\nn = {B[nm]['n']}" for nm in names], fontsize=8)
    b.set_ylim(0, 1.02); b.set_ylabel("P(correct), T=1")
    b.set_xlabel("100-dot uplift measured on the H200 run (P(correct), 100 dots − no filler)")
    b.set_title("By the selection criterion: how much of the seed gain on high-uplift items is regression to the mean?", fontsize=9.5)
    b.legend(fontsize=7.5, loc="upper left", ncol=2)
    fig.suptitle(f"Seed ensembles over all items ({N} WikiText seeds, no patching)", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", y=0.99)
    footnote(fig, f"10-shot, no filler, {N} WikiText seeds (128 tokens, no digits) inserted before the examples; items with "
                  "single-token answers. P(correct) = P(answer) × P(EOS | answer), teacher-forced (= expected T=1 pass rate); "
                  "ensembles over the first n seeded runs' answer distributions (numbers 0–999, renormalised). Vote = 1 if the "
                  "most common argmax is correct; most confident = the seed whose top answer has the highest probability; best "
                  "of n needs the label. Greedy and T=1 numbers are not comparable with each other. Error bars / bands ±1 SE "
                  "over items. 2×H200, FP8 checkpoint.", y=0.02)
    save(fig, "fig21_seed_ensembles_all")


if __name__ == "__main__":
    fig21()
