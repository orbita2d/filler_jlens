"""Figures 17-19: seed-mean patching round 2 (unseeded host, ../seed_patch/results2/main/summary.json).

    ./py.sh plot_seed_patch2.py [results dir name, default main]
"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

from plot import CAT, INK2, INK3, ROOT, footnote, save

RUN = sys.argv[1] if len(sys.argv) > 1 else "main"
S = json.load(open(ROOT / "seed_patch/results2" / RUN / "summary.json"))
cfg = S["config"]
NS = [int(x) for x in cfg["ns"].split(",")]
DEPTHS = [int(x) for x in cfg["depths"].split(",")]
GROUPS = [(g, f"{S['groups'][g]['n']} {'high-uplift' if g == 'uplift' else 'control'} items") for g in ("uplift", "control")
          if g in S["groups"]]
SCOPES = [("ans", "answer position only"), ("all", "every position after the system prompt")]
SETUP = ("Host = the 10-shot prompt with no filler and no seed. Patch at depth l = its residual after layers 0..l "
         "replaced by the seeded prompts' residuals at the matching tokens (each seeded prompt = system + 128-token "
         f"WikiText seed + the same examples and test turn). {cfg['N']} seeds; single seed = averaged over "
         f"{cfg['n_donors']} donor seeds. P(correct) = P(answer) × P(EOS | answer), teacher-forced (= expected T=1 "
         "pass rate). 2×H200, FP8 checkpoint.")
SEQ = ["#86b6ef", "#3987e5", "#0d366b"]                                     # mean of 16 / 50 / 100: light -> dark


def items(g):
    return [s for s in S["items"] if s["group"] == g]


def mse(G, k):
    x = np.array([s[k] for s in G], float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def fig17():
    fig, axes = plt.subplots(len(GROUPS), 2, figsize=(11.5, 3.6 * len(GROUPS)), sharey="row", squeeze=False,
                             gridspec_kw=dict(wspace=0.08, hspace=0.45))
    for r, (g, glab) in enumerate(GROUPS):
        G = items(g)
        for c, (sc, slab) in enumerate(SCOPES):
            a = axes[r][c]
            for key, rl, col, ls in (("dots100", "100 dots, no seed", CAT[2], "--"), ("seed", "seeded prompt, unpatched", CAT[1], ":"),
                                     ("none", "host, unpatched", INK3, "-.")):
                m, e = mse(G, key)
                a.axhspan(m - e, m + e, color=col, alpha=0.12, lw=0)
                a.axhline(m, color=col, ls=ls, lw=1.3, label=f"{rl} ({m:.2f})")
            lines = [("single", "one seed's residuals", CAT[1])] + [(f"mean{n}", f"mean of {n} seeds", SEQ[i]) for i, n in enumerate(NS)]
            for j, (k, lab, col) in enumerate(lines):
                m = np.array([mse(G, f"{k}_{sc}@{d}") for d in DEPTHS])
                x = np.array(DEPTHS) + (j - 1.5) * 0.3
                a.errorbar(x, m[:, 0], yerr=m[:, 1], color=col, marker="o", ms=3.5, lw=1.6, elinewidth=1, capsize=0,
                           mec="white", mew=0.6, label=lab)
            a.set_xticks(DEPTHS); a.set_xlim(0, 44); a.set_ylim(0, 1.02)
            a.set_title(f"{glab}: patch {slab}")
            if c == 0:
                a.set_ylabel("P(correct)")
            if r == len(GROUPS) - 1:
                a.set_xlabel("patched layers 0..l")
        axes[r][1].legend(loc="center right", fontsize=7.5, bbox_to_anchor=(1.0, 0.6))
    fig.suptitle("Patching the unseeded prompt with the seed mean does no better than patching it with a single seed",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=0.99)
    footnote(fig, SETUP + " Error bars ±1 SE over items; points offset horizontally for legibility.", y=-0.01)
    save(fig, "fig17_seed_patch2_levels")


def fig18():
    T = S["takeover"]
    fig, a = plt.subplots(figsize=(7.5, 3.8))
    for sc, lab, col in (("ans", "answer position only", CAT[0]), ("all", "every position", CAT[4])):
        a.plot(DEPTHS, [T[str(d)][f"{sc}_host"] for d in DEPTHS], color=col, marker="o", ms=4, label=f"{lab}: KL(host ‖ patched)")
        a.plot(DEPTHS, [T[str(d)][f"{sc}_donor"] for d in DEPTHS], color=col, marker="o", ms=4, ls="--",
               label=f"{lab}: KL(donor seed ‖ patched)")
    a.set_xticks(DEPTHS); a.set_xlabel("patched layers 0..l"); a.set_ylabel("KL over numbers 0–999 (nats, median)")
    a.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    a.set_title("Single-seed patch: when does the output become the donor's?")
    footnote(fig, SETUP + " A patch that fully takes over drives KL(donor ‖ patched) to the numerics floor and "
                  "KL(host ‖ patched) up to the host-vs-seeded distance. Medians over items and donors.", y=-0.04)
    save(fig, "fig18_seed_patch2_takeover")


def fig19():
    ens_n = sorted({int(k.split("_")[0]) for k in S["groups"][GROUPS[0][0]]["ens"]})
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(11.5, 4), sharey=True, squeeze=False, gridspec_kw=dict(wspace=0.08))
    kinds = [("single", "single seed", CAT[1], "-"), ("probavg", "probability average", CAT[6], "-"),
             ("logitavg", "log-prob average", CAT[0], "-"), ("mostconf", "most confident seed", CAT[4], "-"),
             ("bestof", "best of n (oracle)", INK2, "-"), ("single_greedy", "single seed, greedy", CAT[1], ":"),
             ("vote", "majority vote (greedy)", CAT[3], ":")]
    for a, (g, glab) in zip(axes[0], GROUPS):
        E = S["groups"][g]["ens"]
        for k, lab, col, ls in kinds:
            m = np.array([E[f"{n}_{k}"]["mean"] for n in ens_n]); e = np.array([E[f"{n}_{k}"]["se"] for n in ens_n])
            a.fill_between(ens_n, m - e, m + e, color=col, alpha=0.12, lw=0)
            a.plot(ens_n, m, color=col, ls=ls, marker="o", ms=3.5, label=lab)
        d = S["groups"][g]["dots100"]
        a.axhline(d, color=CAT[2], ls="--", lw=1.3, label=f"100 dots, no seed ({d:.2f})")
        a.set_xscale("log"); a.set_xticks(ens_n, [str(n) for n in ens_n]); a.minorticks_off()
        a.set_ylim(0, 1.02); a.set_xlabel("number of seeds n")
        a.set_title(glab)
    axes[0][0].set_ylabel("P(correct) / accuracy")
    axes[0][-1].legend(fontsize=7.5, loc="upper left", ncol=2)
    fig.suptitle("More seeds: picking the most confident one helps up to ~50 seeds; averaging never helps",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.03)
    footnote(fig, SETUP + " Ensembles over the first n unpatched seeded runs' answer distributions (numbers 0–999, "
                  "renormalised). Solid = T=1 probability of the correct answer; dotted = greedy (0/1) accuracy. "
                  "Bands ±1 SE over items.", y=-0.03)
    save(fig, "fig19_seed_ensembles_n")


def fig20():
    """fig15 redrawn on the round-2 seeds: bars per ensemble, shades = n seeds; T=1 and greedy panels separate."""
    from matplotlib.patches import Patch
    NSEL = [16, 50, 100]
    shade = dict(zip(NSEL, [0.35, 0.65, 1.0]))
    t1 = [("none", "no filler,\nno seed", INK3), ("single", "single\nseed", CAT[1]), ("logitavg", "log-prob\naverage", CAT[0]),
          ("probavg", "probability\naverage", CAT[6]), ("mostconf", "most\nconfident seed", CAT[4]),
          ("bestof", "best of n\n(oracle)", INK2), ("dots100", "100 dots,\nno seed", CAT[2])]
    gr = [("none_greedy", "no filler,\nno seed", INK3), ("single_greedy", "single\nseed", CAT[1]), ("vote", "majority\nvote", CAT[3]),
          ("dots100_greedy", "100 dots,\nno seed", CAT[2])]
    fig, axes = plt.subplots(len(GROUPS), 2, figsize=(14, 3.9 * len(GROUPS)), squeeze=False,
                             gridspec_kw=dict(width_ratios=[len(t1), len(gr) + 0.6], wspace=0.12, hspace=0.55))
    w = 0.26
    for r, (g, glab) in enumerate(GROUPS):
        G, E = items(g), S["groups"][g]["ens"]
        for a, cats, title in ((axes[r][0], t1, "T=1: probability of the correct answer"), (axes[r][1], gr, "greedy: accuracy (0/1)")):
            for i, (k, _, c) in enumerate(cats):
                fixed = k in ("none", "dots100", "none_greedy", "dots100_greedy")
                for j, n in enumerate([None] if fixed else NSEL):
                    if fixed:
                        m, e = mse(G, k); x = i
                    else:
                        m, e = E[f"{n}_{k}"]["mean"], E[f"{n}_{k}"]["se"]; x = i + (j - 1) * (w + 0.02)
                    a.bar(x, m, w if not fixed else w * 1.4, color=c, alpha=1.0 if fixed else shade[n], zorder=2, lw=0)
                    a.errorbar(x, m, yerr=e, color=INK2, lw=1, capsize=0, zorder=3)
                    a.text(x, m + e + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=6.8, color=INK2)
            a.set_xticks(range(len(cats)), [l for _, l, _ in cats], fontsize=8)
            a.set_ylim(0, 1.08); a.grid(axis="x", visible=False)
            a.set_title(f"{glab}, {title}", fontsize=9.5)
        axes[r][0].set_ylabel("P(correct)"); axes[r][1].set_ylabel("accuracy")
    axes[0][0].legend(handles=[Patch(color=INK2, alpha=shade[n], label=f"{n} seeds") for n in NSEL],
                      loc="upper left", fontsize=8, title="bars per ensemble", title_fontsize=8)
    fig.suptitle("Ensembles over 16, 50 and 100 seeds: picking the most confident seed helps, averaging does not",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=0.99)
    footnote(fig, SETUP + " Ensembles over the first n unpatched seeded runs' answer distributions (numbers 0–999, "
                  "renormalised). Vote = 1 if the most common argmax is correct; most confident = the seed whose top answer "
                  "has the highest probability; best of n = the seed giving the correct answer the most mass (needs the "
                  "label). Greedy and T=1 numbers are not comparable with each other. Error bars ±1 SE over items.", y=0.0)
    save(fig, "fig20_seed_ensembles_bars")


if __name__ == "__main__":
    fig17()
    fig18()
    fig19()
    fig20()
