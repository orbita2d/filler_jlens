"""Figures 13-16: seed-mean patching (../seed_patch, results in ../seed_patch/results/main).

Reads summary.json written by seed_patch/analyze.py; same style as plot.py.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from plot import CAT, INK, INK2, INK3, ROOT, footnote, save

SP = ROOT / "seed_patch/results/main"
S = json.load(open(SP / "summary.json"))
ITEMS = S["items"]
DEPTHS = sorted(int(k.split("@")[1]) for k in S["groups"]["uplift"]["gain"] if k.startswith("mean@"))
GROUPS = [("uplift", "50 high-uplift items"), ("control", "50 control items")]
N_SEEDS = S["config"]["N"]
SETUP = (f"10-shot, no filler, {N_SEEDS} WikiText seeds (128 tokens, no digits) inserted before the examples. "
         "Patch at depth l = the answer position's residual after layers 0..l replaced, then the rest of the forward "
         "pass runs normally. P(correct) = P(answer) × P(EOS | answer), teacher-forced (= expected T=1 pass rate). "
         "High-uplift items: top 50 by 100-dot uplift (≥0.75 on the H200 run); controls matched on no-filler P(correct) "
         "with |uplift| < 0.05. 4×H100, FP8 checkpoint.")


def col(group, key):
    return np.array([s[key] for s in ITEMS if s["group"] == group], float)


def mse(x):
    x = np.asarray(x, float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


# ---------------------------------------------------------------------------------------------
# 13. Absolute P(correct) after patching vs depth, against the reference levels
# ---------------------------------------------------------------------------------------------
def fig13():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True, gridspec_kw=dict(wspace=0.08))
    for a, (g, lab) in zip(axes, GROUPS):
        refs = [("dots100", "100 dots, no seed", CAT[2], "--"), ("seed_base", "single seed (unpatched)", CAT[1], ":"),
                ("none", "no filler, no seed", INK3, "-.")]
        for key, rl, c, ls in refs:
            m, se = mse(col(g, key))
            a.axhspan(m - se, m + se, color=c, alpha=0.12, lw=0)
            a.axhline(m, color=c, ls=ls, lw=1.4, label=f"{rl} ({m:.2f})")
        m = np.array([mse(col(g, f"mean@{d}")) for d in DEPTHS])
        a.fill_between(DEPTHS, m[:, 0] - m[:, 1], m[:, 0] + m[:, 1], color=CAT[0], alpha=0.18, lw=0)
        a.plot(DEPTHS, m[:, 0], color=CAT[0], marker="o", ms=4, mec="white", mew=0.8,
               label=f"seed host patched with mean of other {N_SEEDS - 1} seeds ({m[:, 0].mean():.2f})")
        a.set_xlim(0, 42)
        a.set_xticks(DEPTHS)
        a.set_ylim(0, 1)
        a.set_xlabel("patched layers 0..l")
        a.set_title(lab)
        a.legend(loc="center left", fontsize=8, bbox_to_anchor=(0, 0.62))
        if g == "uplift":
            a.set_ylabel("P(correct)")
    fig.suptitle("Averaging the seeds' answer-token residuals leaves P(correct) exactly at the single-seed level",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.03)
    footnote(fig, SETUP + " Bands: ±1 SE over items (each item averaged over its 16 seeded hosts).", y=-0.03)
    save(fig, "fig13_seed_patch_levels")


# ---------------------------------------------------------------------------------------------
# 14. Gain vs depth for every patch kind, uplift vs control, plus item-level scatter
# ---------------------------------------------------------------------------------------------
def fig14():
    kinds = [("mean", f"mean of {N_SEEDS - 1} other seeds", CAT[0]), ("mean_nm", "same, norm-matched", CAT[4]),
             ("k4", "mean of 4 other seeds", CAT[3]), ("k1", "1 other seed (swap)", CAT[1])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 0.95], wspace=0.32))
    for a, (g, lab) in zip(axes[:2], GROUPS):
        base = col(g, "seed_base")
        for j, (k, kl, c) in enumerate(kinds):
            m = np.array([mse(100 * (col(g, f"{k}@{d}") - base)) for d in DEPTHS])
            x = np.array(DEPTHS) + (j - 1.5) * 0.35
            a.errorbar(x, m[:, 0], yerr=m[:, 1], color=c, marker="o", ms=3.5, lw=1.4, elinewidth=1, capsize=0,
                       label=kl, mec="white", mew=0.6)
        a.axhline(0, color=INK3, lw=0.8)
        a.set_ylim(-2, 2)
        a.set_xticks(DEPTHS)
        a.set_xlabel("patched layers 0..l")
        a.set_title(lab)
        need = 100 * (col(g, "dots100").mean() - base.mean())
        a.text(0.02, 0.97, f"100 dots − single seed: {need:+.0f} points" + (" (off scale)" if abs(need) > 2 else ""), transform=a.transAxes, fontsize=8,
               color=CAT[2], va="top")
    axes[0].set_ylabel("ΔP(correct) vs unpatched seed host (points)")
    axes[1].legend(loc="lower left", fontsize=7.5, ncol=2)
    # item-level: gain averaged over depths vs this pod's 100-dot uplift
    b = axes[2]
    for g, c in (("uplift", CAT[0]), ("control", CAT[1])):
        gain = np.mean([col(g, f"mean@{d}") for d in DEPTHS], 0) - col(g, "seed_base")
        up = col(g, "dots100") - col(g, "none")
        b.scatter(up, 100 * gain, s=16, color=c, alpha=0.8, lw=0, label=f"{g} items")
    b.axhline(0, color=INK3, lw=0.8)
    rho = S.get("specificity", {})
    b.set_xlabel("100-dot uplift on this pod (ΔP(correct))")
    b.set_ylabel("mean-patch gain, avg over depths (points)")
    b.set_title("No item-level link to the filler uplift")
    b.legend(loc="upper left", fontsize=8)
    rhos = [v["rho"] for v in rho.values()]
    if rhos:
        b.text(0.98, 0.03, f"Spearman ρ per depth: {min(rhos):+.2f} … {max(rhos):+.2f}", transform=b.transAxes,
               ha="right", fontsize=8, color=INK2)
    fig.suptitle("Every patch variant is flat at every depth, on filler-sensitive items and controls alike",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.03)
    footnote(fig, SETUP + " Error bars ±1 SE of the paired difference over items; points offset horizontally for "
                  "legibility. None of the 88 tests is significant (one-sided, Bonferroni over 11 depths). Norm-matched: the "
                  "mean rescaled per stream and layer to the seeds' mean RMS.", y=-0.03)
    save(fig, "fig14_seed_patch_gain")


# ---------------------------------------------------------------------------------------------
# 15. Output-level ensembles over the 16 seeds
# ---------------------------------------------------------------------------------------------
def fig15():
    ens = [("ens_none_num", "no filler,\nno seed", INK3), ("ens_seed_num", "single\nseed", CAT[1]),
           ("ens_logitavg", "log-prob\naverage", CAT[0]), ("ens_probavg", "probability\naverage", CAT[6]),
           ("ens_vote", "majority\nvote", CAT[3]), ("ens_mostconf", "most\nconfident seed", CAT[4]),
           ("ens_bestof", f"best of {N_SEEDS}\n(oracle)", INK2), ("ens_dots100_num", "100 dots,\nno seed", CAT[2])]
    fig, a = plt.subplots(figsize=(11, 4))
    w = 0.38
    for gi, (g, lab) in enumerate(GROUPS):
        for i, (k, _, c) in enumerate(ens):
            m, se = mse(col(g, k))
            x = i + (gi - 0.5) * (w + 0.03)
            a.bar(x, m, w, color=c, alpha=1.0 if g == "uplift" else 0.38, zorder=2,
                  hatch=None if g == "uplift" else "///", edgecolor="white" if g == "uplift" else c, lw=0)
            a.errorbar(x, m, yerr=se, color=INK2, lw=1, capsize=0, zorder=3)
            a.text(x, m + se + 0.012, f"{m:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    a.set_xticks(range(len(ens)), [l for _, l, _ in ens])
    a.set_ylim(0, 1.05)
    a.set_ylabel("P(correct) among numbers 0–999")
    a.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    a.legend(handles=[Patch(color=INK2, label="50 high-uplift items (solid)"),
                      Patch(facecolor="none", edgecolor=INK2, hatch="///", label="50 controls (hatched)")],
             loc="upper left", fontsize=8)
    a.set_title("Across seeds, selection helps and averaging does not; neither gets near 100 dots")
    footnote(fig, SETUP + " Ensembles are over the 16 unpatched seeded runs' answer distributions (numbers 0–999, "
                  "renormalised). Vote = 1 if the most common argmax is correct; most confident = the seed whose top "
                  "answer has the highest probability; best of 16 = the seed giving the correct answer the most mass "
                  "(needs the label). Error bars ±1 SE over items.", y=-0.03)
    save(fig, "fig15_seed_ensembles")


# ---------------------------------------------------------------------------------------------
# 16. Geometry: how far the seeds move the answer residual, relative to what 100 dots does
# ---------------------------------------------------------------------------------------------
def fig16():
    G = {k: np.array(v) for k, v in S["geometry"].items()}
    L = np.arange(len(G["spread"]))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw=dict(wspace=0.28))
    norm = G["none_norm"]
    for k, lab, c in (("dots_dist", "‖dots100 − no filler‖", CAT[2]), ("none_to_mean", "‖seed mean − no filler‖", CAT[0]),
                      ("spread", "seed spread (mean ‖seed − seed mean‖)", CAT[1])):
        a.plot(L, G[k] / norm, color=c, label=lab)
    a.set_xlabel("layer")
    a.set_ylabel("distance / ‖no-filler residual‖")
    a.set_xlim(0, L[-1])
    a.legend(loc="upper left", fontsize=8)
    a.set_title("Seed mean moves about half as far as 100 dots")
    b.plot(L, G["spread"] / G["dots_dist"], color=CAT[1], label="seed spread / ‖dots100 − no filler‖")
    b.plot(L, G["none_to_mean"] / G["dots_dist"], color=CAT[0], label="‖seed mean − no filler‖ / ‖dots100 − no filler‖")
    b.plot(L, G["cos_mean_dots"], color=CAT[4], label="cos(seed mean − no filler, dots100 − no filler)")
    b.axhline(0, color=INK3, lw=0.8)
    b.set_ylim(-0.1, 1.25)
    b.set_xlim(0, L[-1])
    b.set_xlabel("layer")
    b.legend(loc="upper left", fontsize=8)
    b.set_title("…and only partly in the same direction")
    fig.suptitle("The perturbation is real and partly aligned with filler, but averaging it buys nothing",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.03)
    footnote(fig, SETUP + " Residual after each layer at the answer position, all 4 hyper-connection streams "
                  "concatenated (4×4096). Medians over all 100 items.", y=-0.03)
    save(fig, "fig16_seed_geometry")


if __name__ == "__main__":
    fig13()
    fig14()
    fig15()
    fig16()
