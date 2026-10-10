"""Figures for the V4 Flash filler-token run, compared with Brauer, Verdun & Marks (arXiv 2607.03502).

Run compute_decode.py first (writes cache/). Reads results/ and repo/ only; writes PNG + PDF here.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
CACHE = HERE / "cache"

# Palette: reference categorical order (blue, orange, aqua, yellow, magenta, ...), sequential blue ramp.
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e4e3df"
SURF = "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TARGET_COL = dict(zip(["x", "c1x", "y", "c2y", "answer"], CAT[:5]))
TARGET_LAB = {"x": "x", "c1x": "c₁·x", "y": "y", "c2y": "c₂·y", "answer": "answer"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
COND_LAB = {"k0": "no filler", "dots10": "10 dots", "dots100": "100 dots"}
CONDS = ["k0", "dots10", "dots100"]
LENS_NOTE = "J-lens (neuronpedia deepseek-v4-flash, mean of the 4 streams), top numeric token among 0–999"

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": INK3, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False, "lines.linewidth": 2,
    "text.color": INK, "figure.dpi": 110,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def rollouts(shots):
    by = {}
    for l in open(RES / f"rollout_{shots}shot.jsonl"):
        r = json.loads(l)
        by.setdefault(r["idx"], {})[r["cond"]] = r
    return by


def footnote(fig, text, y=-0.02):
    fig.text(0.01, y, text, ha="left", va="top", fontsize=7.5, color=INK3, wrap=True)


# ---------------------------------------------------------------------------------------------
# 1. Accuracy vs filler, and against the paper's system-of-equations numbers
# ---------------------------------------------------------------------------------------------
def fig_accuracy():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.35))
    xs = np.arange(3)
    series = []
    for shots, ls in ((10, "-"), (0, "--")):
        by = rollouts(shots)
        for metric, col in (("greedy", CAT[0]), ("T=1", CAT[1])):
            m, se = [], []
            for c in CONDS:
                v = np.array([(r[c]["greedy_correct"] if metric == "greedy" else r[c]["n_correct"] / len(r[c]["samples"]))
                              for r in by.values()], float)
                m.append(100 * v.mean()); se.append(100 * v.std(ddof=1) / np.sqrt(len(v)))
            series.append((f"{shots}-shot, {'greedy' if metric == 'greedy' else 'T=1 pass rate'}", m, se, col, ls))
    for lab, m, se, col, ls in series:
        eb = a.errorbar(xs, m, yerr=se, color=col, ls=ls, marker="o", ms=5, capsize=0, elinewidth=1.2,
                        mec=SURF, mew=1.2, label=lab)
        eb[2][0].set_linestyle("-")
        a.annotate(f"{m[-1]:.0f}%", (xs[-1], m[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                   fontsize=8, color=INK2)
    a.set_xticks(xs, [COND_LAB[c] for c in CONDS])
    a.set_xlim(-0.25, 2.45)
    a.set_ylim(0, 72)
    a.set_ylabel("accuracy (%)")
    a.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    a.set_title("DeepSeek V4 Flash: accuracy vs filler (600 items)")
    a.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=8)
    a.grid(axis="x", visible=False)

    # Paper: system-of-equations task, baseline -> best filler. V3/K2 from the text (Sec. 3), K2.5 Sec. 3.
    rows = [
        ("DeepSeek V3 (paper)", 31.1, 61.0, CAT[6]),
        ("Kimi K2 (paper)", 18.4, 36.4, CAT[5]),
        ("Kimi K2.5, no-think (paper)", 31.0, 46.0, CAT[7]),
        ("V4 Flash (orvelte repo)", 53.5, 63.5, INK3),
        ("V4 Flash (this run)", 54.2, 64.0, CAT[0]),
    ]
    label_y = {"V4 Flash (this run)": 67.5, "V4 Flash (orvelte repo)": 63.3, "DeepSeek V3 (paper)": 59.1}
    for lab, y0, y1, col in rows:
        lw = 2.6 if "this run" in lab else 1.8
        b.plot([0, 1], [y0, y1], color=col, lw=lw, marker="o", ms=6, mec=SURF, mew=1.2, zorder=3)
        b.text(1.04, label_y.get(lab, y1), f"{lab}  +{y1 - y0:.0f}", va="center", fontsize=8, color=INK)
    b.set_xticks([0, 1], ["no filler", "with filler"])
    b.set_xlim(-0.15, 1.0)
    b.set_ylim(10, 72)
    b.set_ylabel("accuracy (%)")
    b.grid(axis="x", visible=False)
    b.set_title("Same task family across models")
    footnote(fig, "Left: error bars ±1 SE over items. 0-shot k0 is mostly a formatting failure: only 50% of its "
                  "samples are a bare number (96% with 10 dots, 99% with 100). Right: system-of-equations task "
                  "(Brauer et al. Fig. 2 & Sec. 3; their 'with filler' is the best k, 5-shot prompts with filler in the "
                  "shots; ours is 100 dots, 10-shot, greedy). Item draws differ: the paper's 500 items vs the repo's 600.")
    save(fig, "fig1_accuracy")


# ---------------------------------------------------------------------------------------------
# 2. Per-item transitions: T=1 pass count without vs with 100 dots
# ---------------------------------------------------------------------------------------------
def fig_items():
    by = rollouts(10)
    M = np.zeros((9, 9), int)
    for r in by.values():
        M[r["k0"]["n_correct"], r["dots100"]["n_correct"]] += 1
    fig, ax = plt.subplots(figsize=(5.6, 4.9))
    norm = mpl.colors.LogNorm(vmin=1, vmax=M.max())
    im = ax.imshow(np.where(M > 0, M, np.nan), origin="lower", cmap=BLUES, norm=norm)
    for i in range(9):
        for j in range(9):
            if M[i, j]:
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=7.5,
                        color="white" if M[i, j] > 25 else INK)
    ax.plot([-0.5, 8.5], [-0.5, 8.5], color=INK3, lw=0.8, ls=":")
    ax.set_xticks(range(9)); ax.set_yticks(range(9))
    ax.set_xlabel("correct samples out of 8, with 100 dots")
    ax.set_ylabel("correct samples out of 8, no filler")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    up = M[np.triu_indices(9, 1)].sum(); down = M[np.tril_indices(9, -1)].sum()
    ax.set_title(f"Which items filler helps (10-shot, T=1)")
    ax.add_patch(mpl.patches.Rectangle((7.5, -0.5), 1, 1, fill=False, ec=CAT[1], lw=2))
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("items", color=INK2); cb.outline.set_visible(False)
    footnote(fig, f"600 items, 8 samples each at T=1 per condition. Better with dots: {up} items, worse: {down}, "
                  f"same count: {600 - up - down}. Off-diagonal mass is partly sampling noise. Orange box: {M[0, 8]} items "
                  "go 0/8 → 8/8 (the lens 'uplift' subset uses 25 of them).", y=0.0)
    save(fig, "fig2_item_transitions")


# ---------------------------------------------------------------------------------------------
# Lens decode maps
# ---------------------------------------------------------------------------------------------
LENSES = {"jlens": "J-lens", "logit_hchead": "logit lens (hc_head)", "logit_mean": "logit lens (stream mean)"}
LOGIT_NOTE = ("Logit lens = lm_head(final norm(collapse(h))), no J; hc_head applies the model's own final stream mix "
              "at every layer, mean averages the 4 streams. Its last layer (L42) is the model's own output")


def load_decode(cond, mode="hits_raw", lens="jlens"):
    z = np.load(CACHE / (f"decode_{cond}.npz" if lens == "jlens" else f"decode_{lens}_{cond}.npz"))
    return z, z[mode]


def target_rate(z, h, t, mask=None):
    v = z["valid"][:, t]
    if mask is not None:
        v = v & mask
    return h[v, t].mean(0), v.sum()


def pos_labels(z):
    nq, nf, na = int(z["n_q"]), int(z["nf"]), int(z["na"])
    return nq, nf, na


def fig_heatmaps():
    """Paper Fig. 3 analog: decode rate per (layer, position) for each chain quantity and condition."""
    targets = ["x", "c1x", "y", "c2y", "answer"]
    zs = {c: load_decode(c) for c in CONDS}
    widths = [zs[c][1].shape[-1] for c in CONDS]
    fig, axes = plt.subplots(len(targets), 3, figsize=(13, 10.5), sharey=True,
                             gridspec_kw=dict(width_ratios=[w + 4 for w in widths], hspace=0.28, wspace=0.06))
    vmax = 0.8
    for i, tn in enumerate(targets):
        for j, c in enumerate(CONDS):
            ax = axes[i, j]
            z, h = zs[c]
            r, n = target_rate(z, h, i)
            im = ax.imshow(r, origin="lower", aspect="auto", cmap=BLUES, vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_ylim(17.5, 41.5)
            nq, nf, na = pos_labels(z)
            w = r.shape[1]
            for x in (nq - 0.5, nq + nf - 0.5):
                ax.axvline(x, color=INK2, lw=0.8, ls="--")
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == len(targets) - 1:
                ticks = [nq // 2 + 0, nq + nf // 2, nq + nf + na // 2]
                narrow = w < 30
                ax.set_xticks([1, nq + (nf - 1) / 2, w - 1], ["question", "filler", "answer"], fontsize=8,
                              rotation=90 if narrow else 0)
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(f"{TARGET_LAB[tn]}\nlayer", color=TARGET_COL[tn] if False else INK, fontsize=9)
            if i == 0:
                ax.set_title(COND_LAB[c], loc="center")
            L, p = np.unravel_index(r.argmax(), r.shape)
            ax.text(0.99, 0.04, f"peak {r.max():.0%}\n@ L{L}" if r.shape[1] < 30 else f"peak {r.max():.0%} @ L{L}",
                    transform=ax.transAxes, ha="right", fontsize=7,
                    color=INK2, bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.85))
    cax = fig.add_axes([0.92, 0.3, 0.012, 0.4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("fraction of items where the value is the lens's top number", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("Where each chain value is readable: layer × position, 600 items", x=0.125, ha="left",
                 fontweight="bold", y=0.93)
    footnote(fig, f"{LENS_NOTE}. Window = last 3 question tokens | 'Filler:' + dots + newline | "
                  "'Answer:<Assistant></think>' (answer position is the last column). Chain: x → c₁·x → y "
                  "→ c₂·y → answer; x is written in the prompt, y and c₂·y never are "
                  "(y literally appears in 9 prompts). Chance ≈ 0.1%. Layers 0–17 decode nothing and are cropped.", y=0.03)
    save(fig, "fig3_decode_heatmaps")


def fig_composite(lens="jlens", name="fig4_chain_composite"):
    """Paper Fig. 4 analog: one panel per condition, each cell coloured by the best-decoded chain value,
    opacity = its decode rate."""
    targets = ["x", "c1x", "y", "c2y", "answer"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True,
                             gridspec_kw=dict(width_ratios=[16, 22, 110], wspace=0.04))
    for ax, c in zip(axes, CONDS):
        z, h = load_decode(c, lens=lens)
        R = np.stack([target_rate(z, h, t)[0] for t in range(5)])  # 5, L, w
        best = R.argmax(0)
        strength = np.clip(R.max(0) / 0.6, 0, 1)
        cols = np.array([to_rgb(TARGET_COL[t]) for t in targets])
        bg = np.array(to_rgb(SURF))
        img = bg + (cols[best] - bg) * strength[..., None]
        ax.imshow(img, origin="lower", aspect="auto", interpolation="nearest")
        ax.set_ylim(17.5, R.shape[1] - 0.5)
        nq, nf, na = pos_labels(z)
        for x in (nq - 0.5, nq + nf - 0.5):
            ax.axvline(x, color=INK2, lw=0.8, ls="--")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        w = R.shape[-1]
        ax.set_xticks([1, nq + (nf - 1) / 2, w - 1], ["question", "filler", "answer"], fontsize=8,
                      rotation=90 if w < 30 else 0)
        ax.set_title(COND_LAB[c], loc="center")
    axes[0].set_ylabel("layer")
    handles = [mpl.patches.Patch(color=TARGET_COL[t], label=TARGET_LAB[t]) for t in targets]
    fig.legend(handles=handles, loc="upper left", ncol=5, bbox_to_anchor=(0.12, 1.01), fontsize=9)
    title = "The chain unrolls in depth order, and with filler it spreads across the dots"
    if lens != "jlens":
        title = f"Same map with the {LENSES[lens]}: the picture does not depend on the J-lens"
    fig.suptitle(title, x=0.125, ha="left", fontweight="bold", y=1.08)
    footnote(fig, f"Layers 0–17 decode nothing and are cropped. Compare Brauer et al. Fig. 4 (DeepSeek V3, logit lens). Colour = chain value with the highest decode "
                  f"rate in that cell; opacity = that rate (full at ≥60%). "
                  f"{LENS_NOTE if lens == 'jlens' else LOGIT_NOTE}.", y=-0.1)
    save(fig, name)


def fig_ladder():
    """Peak layer of each chain value vs the paper's DeepSeek V3 ladder, as relative depth."""
    targets = ["x", "c1x", "y", "c2y", "answer"]
    z, h = load_decode("dots100")
    z0, h0 = load_decode("k0")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.55))
    nq, nf, na = pos_labels(z)
    for t, tn in enumerate(targets):
        r = target_rate(z, h, t)[0]
        # best position in the question/filler/answer window at each layer
        a.plot(np.arange(r.shape[0]), r.max(1), color=TARGET_COL[tn], label=TARGET_LAB[tn])
        r0 = target_rate(z0, h0, t)[0]
        a.plot(np.arange(r0.shape[0]), r0.max(1), color=TARGET_COL[tn], lw=1, ls=":", alpha=0.9)
    a.set_xlabel("layer"); a.set_ylabel("decode rate at best position")
    a.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    a.set_title("V4 Flash: each value peaks one layer band after the last")
    a.legend(loc="upper left", fontsize=8, ncol=1)
    a.text(0.99, 0.97, "solid: 100 dots\ndotted: no filler", transform=a.transAxes, ha="right", va="top",
           fontsize=8, color=INK2)
    # relative depth comparison
    v4 = []
    for t in range(5):
        r = target_rate(z, h, t)[0].max(1)
        v4.append(r.argmax() / 41)
    v3 = np.array([33, 38, 44, 51, 60]) / 60  # Brauer et al. repo, scripts/kimi_k25/RUNBOOK.md (DeepSeek V3 peaks)
    for t, tn in enumerate(targets):
        b.plot([v3[t], v4[t]], [1, 0], color=TARGET_COL[tn], lw=1.2, alpha=0.6)
        for y, xv in ((1, v3[t]), (0, v4[t])):
            b.plot(xv, y, "o", color=TARGET_COL[tn], ms=9, mec=SURF, mew=1.5, zorder=3)
        dx = {"y": -6, "c2y": 6}.get(tn, 0)
        b.annotate(TARGET_LAB[tn], (v4[t], 0), xytext=(dx, -14), textcoords="offset points",
                   ha={"y": "right", "c2y": "left"}.get(tn, "center"), fontsize=8)
    b.set_yticks([0, 1], ["V4 Flash\n(J-lens, 42 layers)", "DeepSeek V3\n(logit lens, 61 layers)"])
    b.set_ylim(-0.4, 1.3)
    b.set_xlim(0.45, 1.03)
    b.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    b.set_xlabel("peak layer / last layer")
    b.grid(axis="y", visible=False)
    b.set_title("Same order, similar relative depth")
    footnote(fig, "Left: max over the 110-position window per layer (600 items). Right: DeepSeek V3 peak layers "
                  "x@L33, c₁x@L38, y@L44, c₂y@L51, answer@L60 from the paper's repository "
                  "(kimi_k25/RUNBOOK.md); V4 Flash peaks from the left panel.")
    save(fig, "fig5_depth_ladder")


def fig_correct_wrong():
    """Paper Fig. 3 bottom analog: decode on the filler for items the model gets right vs wrong (100 dots)."""
    z, h = load_decode("dots100")
    nq, nf, na = pos_labels(z)
    dots = slice(nq + 2, nq + nf - 1)  # the 100 dot tokens
    cor = z["correct"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, gridspec_kw=dict(wspace=0.1))
    for ax, tn in zip(axes, ["x", "y", "answer"]):
        t = ["x", "c1x", "y", "c2y", "answer"].index(tn)
        for mask, lab, col in ((cor, "correct", CAT[0]), (~cor, "wrong", CAT[1])):
            r, n = target_rate(z, h, t, mask)
            prof = r[:, dots].mean(1)
            ax.plot(np.arange(len(prof)), prof, color=col, label=f"{lab} (n={n})")
            ans = r[:, -1]
            ax.plot(np.arange(len(ans)), ans, color=col, lw=1, ls=":")
        ax.set_title(f"{TARGET_LAB[tn]}", loc="left")
        ax.set_xlabel("layer")
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    axes[0].set_ylabel("decode rate")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].text(0.02, 0.97, "solid: mean over the 100 dots\ndotted: answer position", transform=axes[1].transAxes,
                 va="top", fontsize=8, color=INK2)
    fig.suptitle("Right vs wrong answers (100 dots, greedy): x and y look the same, only the answer differs", x=0.125, ha="left", fontweight="bold", y=1.03)
    footnote(fig, f"{LENS_NOTE}. Brauer et al. found the inputs present and the composed value missing on wrong "
                  "2-fact items. Here the intermediates decode equally on the dots for right and wrong answers; "
                  "the correct answer forms late at the answer position only when the model gets it right.", y=-0.03)
    save(fig, "fig6_correct_vs_wrong")


def fig_subset():
    """25 uplift items (0/8 -> 8/8) vs 25 matched no-uplift items: does y show up on the dots?"""
    sub = json.load(open(RES / "subset.json"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, c in zip(axes, ["k0", "dots100"]):
        z, h = load_decode(c)
        nq, nf, na = pos_labels(z)
        for grp, col in (("uplift", CAT[2]), ("no_uplift", CAT[4])):
            m = np.zeros(600, bool); m[sub[grp]] = True
            for tn, ls in (("y", "-"), ("answer", "--")):
                t = ["x", "c1x", "y", "c2y", "answer"].index(tn)
                r, n = target_rate(z, h, t, m)
                prof = r[:, : -na].max(1) if tn == "y" else r[:, -1]
                ax.plot(prof, color=col, ls=ls,
                        label=f"{'uplift' if grp == 'uplift' else 'no uplift'}: {TARGET_LAB[tn]}")
        ax.set_title(COND_LAB[c], loc="left")
        ax.set_xlabel("layer")
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    axes[0].set_ylabel("decode rate")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Both groups compute y; with 100 dots only the rescued items turn it into the answer", x=0.125, ha="left",
                 fontweight="bold", y=1.03)
    footnote(fig, "Uplift: 0/8 → 8/8 at T=1 with 100 dots; no uplift: 0/8 → ~0/8, matched on the no-filler "
                  "pass rate. y: best position before the answer segment; answer: at the answer position. "
                  "n = 25 per group, so a single item is 4 points.", y=-0.03)
    save(fig, "fig7_uplift_subset")


# ---------------------------------------------------------------------------------------------
# 2026-10-09 additions: logit lens vs J-lens, MoE routing freeze
# ---------------------------------------------------------------------------------------------
def fig_lens_compare():
    """Per chain value, best-position decode rate by layer (100 dots) for the J-lens and both logit lenses."""
    targets = ["x", "c1x", "y", "c2y", "answer"]
    styles = {"jlens": (CAT[0], "-", 2.2), "logit_hchead": (CAT[1], "-", 1.6), "logit_mean": (CAT[2], "--", 1.6)}
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.6), sharey=True, sharex=True, gridspec_kw=dict(hspace=0.35, wspace=0.08))
    for row, cond in enumerate(["k0", "dots100"]):
        for t, tn in enumerate(targets):
            ax = axes[row, t]
            for lens, (col, ls, lw) in styles.items():
                z, h = load_decode(cond, lens=lens)
                r = target_rate(z, h, t)[0]
                curve = r[:, -1] if tn == "answer" else r.max(1)
                ax.plot(np.arange(len(curve)), curve, color=col, ls=ls, lw=lw, label=LENSES[lens])
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
            if row == 0:
                ax.set_title(TARGET_LAB[tn] + (" (answer position)" if tn == "answer" else ""), loc="left")
            if row == 1:
                ax.set_xlabel("layer")
        axes[row, 0].set_ylabel(f"{COND_LAB[cond]}\ndecode rate")
    axes[0, 0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("J-lens vs logit lens: same values, same layers", x=0.125, ha="left", fontweight="bold", y=0.99)
    footnote(fig, f"600 items, 10-shot. Decode rate = fraction of items where the value is the lens's top number among "
                  "0–999; x, c₁·x, y, c₂·y at the best position in the question/filler/answer "
                  f"window per layer, the answer at the answer position. J-lens: 42 layers (0–41). {LOGIT_NOTE}.",
             y=0.0)
    save(fig, "fig9_lens_compare")


def fig_freeze():
    """MoE routing freeze: uplift in P(correct) over no filler, per variant, plus natural expert overlap."""
    by = {}
    for l in open(RES / "freeze_10shot.jsonl"):
        r = json.loads(l)
        by.setdefault(r["idx"], {})[(r["cond"], r["variant"])] = r
    items = list(by.values())
    base = np.array([np.exp(v[("k0", "natural")]["logp_correct"]) for v in items])
    variants = [("natural", "natural routing"), ("answer_only", "freeze answer token only"),
                ("freeze_idx", "freeze experts\n(dots + answer)"), ("freeze_idxw", "freeze experts + weights\n(dots + answer)")]
    vcol = [CAT[0], CAT[2], CAT[1], CAT[3]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.3))
    width = 0.19
    for ci, cond in enumerate(["dots100", "dots10"]):
        nat = np.mean([np.exp(v[(cond, "natural")]["logp_correct"]) for v in items]) - base.mean()
        for vi, (var, lab) in enumerate(variants):
            d = np.array([np.exp(v[(cond, var)]["logp_correct"]) for v in items]) - base
            m, se = 100 * d.mean(), 100 * d.std(ddof=1) / np.sqrt(len(d))
            x = ci + (vi - 1.5) * (width + 0.02)
            a.bar(x, m, width, color=vcol[vi], label=lab.replace("\n", " ") if ci == 0 else None, zorder=2)
            a.errorbar(x, m, yerr=se, color=INK2, lw=1.1, capsize=0, zorder=3)
            txt = f"{m:+.1f}" + (f"\n{d.mean() / nat:.0%}" if cond == "dots100" and var != "natural" else "")
            a.text(x, max(m, 0) + se + 0.3, txt, ha="center", va="bottom", fontsize=7.5, color=INK)
    a.axhline(0, color=INK3, lw=0.8)
    a.set_xticks([0, 1], ["100 dots", "10 dots"])
    a.set_ylabel("uplift in P(correct) over no filler (points)")
    a.set_ylim(-2.5, 19)
    a.grid(axis="x", visible=False)
    a.legend(loc="upper right", fontsize=8)
    a.set_title("Freezing the dots' experts removes about half the uplift")
    for cond, col in (("dots100", CAT[0]), ("dots10", CAT[1])):
        ov = np.array([[v[(cond, "natural")]["overlap_dots"][k] for k in sorted(v[(cond, "natural")]["overlap_dots"], key=int)]
                       for v in items])
        m = ov.mean(0)
        b.plot(np.arange(len(m)), m, color=col, label=f"{COND_LAB[cond]} (mean {ov.mean():.0%})")
    b.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    b.set_ylim(0, 1)
    b.set_xlabel("MoE layer")
    b.set_ylabel("share of dot tokens' experts in the frozen set")
    b.legend(loc="upper left", fontsize=8)
    b.set_title("Freezing is a big change: dots naturally use other experts")
    footnote(fig, "600 items, 10-shot, teacher-forced P(correct) = expected T=1 pass rate; error bars ±1 SE of the "
                  "paired difference. Frozen set = the experts the answer position uses with no filler, at every MoE layer "
                  "(shared experts and attention untouched). Percentages above the 100-dot bars = share of the natural uplift "
                  "kept. Caveat: no control yet for freezing to some other, equally foreign expert set (random experts or "
                  "another item's), so this cannot yet separate 'these experts matter' from 'any forced off-distribution "
                  "routing hurts'.", y=-0.02)
    save(fig, "fig8_moe_freeze")


def fig_freeze2():
    """MoE routing freeze round 2 (freeze_moe2.py): which fixed expert set the dots are forced through, with
    another-item and random-expert controls."""
    by = {}
    for l in open(RES / "freeze2_10shot.jsonl"):
        r = json.loads(l)
        by.setdefault(r["idx"], {})[(r["cond"], r["variant"])] = r
    items = list(by.values())
    base = np.array([np.exp(v[("k0", "natural")]["logp_correct"]) for v in items])
    # (variant, label, colour); colour = family
    rows = [
        ("natural", "natural routing", CAT[0]),
        ("first_idx", "every dot \u2192 first dot's experts", CAT[2]),
        ("first_idxw", "  + first dot's gate weights", CAT[2]),
        ("ans_idx_nohash", "dots + answer \u2192 no-filler answer experts\n  (hash layers 0\u20132 untouched)", CAT[1]),
        ("ans_idxw_nohash", "  + its gate weights (hash untouched)", CAT[1]),
        ("ans_idx", "dots + answer \u2192 no-filler answer experts\n  (all layers; round 1's 'freeze experts')", CAT[1]),
        ("foreign_idx", "control: another item's answer experts", CAT[3]),
        ("random_idx", "control: random experts", CAT[4]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True, gridspec_kw=dict(wspace=0.06))
    ys = np.arange(len(rows))[::-1]
    for ax, cond in zip(axes, ["dots100", "dots10"]):
        nat = np.mean([np.exp(v[(cond, "natural")]["logp_correct"]) for v in items]) - base.mean()
        for y, (var, lab, col) in zip(ys, rows):
            d = np.array([np.exp(v[(cond, var)]["logp_correct"]) for v in items]) - base
            m, se = 100 * d.mean(), 100 * d.std(ddof=1) / np.sqrt(len(d))
            hatch = "////" if "weights" in lab else None
            ax.barh(y, m, 0.68, color=col, alpha=0.55 if hatch else 1, hatch=hatch, edgecolor=SURF, lw=0, zorder=2)
            ax.errorbar(m, y, xerr=se, color=INK2, lw=1.1, capsize=0, zorder=3)
            txt = f"{m:+.1f}"
            if cond == "dots100" and var != "natural" and m > 0:
                txt += f"  ({d.mean() / nat:.0%})"
            ax.text(m + (se + 0.4) * (1 if m >= 0 else -1), y, txt, va="center", ha="left" if m >= 0 else "right",
                    fontsize=8, color=INK)
        ax.axvline(0, color=INK3, lw=0.8)
        ax.axvline(100 * nat, color=CAT[0], lw=0.8, ls=":")
        ax.set_xlim(-27, 19)
        ax.set_title(COND_LAB[cond], loc="left")
        ax.set_xlabel("uplift in P(correct) over no filler (points)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, [r[1] for r in rows], fontsize=8.5)
    fig.suptitle("Forcing the dots through any plausible fixed expert set keeps 40–73% of the uplift; random experts break the model", x=0.125, ha="left",
                 fontweight="bold", y=1.0)
    footnote(fig, "Round 2, 600 items, 10-shot, teacher-forced P(correct) = expected T=1 pass rate; error bars \u00b11 SE "
                  "of the paired difference; percentages = share of the natural 100-dot uplift kept. 'First dot' variants "
                  "leave the answer position routed normally. Hatched bars also copy the gate weights instead of letting "
                  "each token weight its frozen experts. Run on 4\u00d7H100 (round 1 used B200s), so the natural uplift "
                  "differs slightly from fig8 (+13.5 vs +15.0) from FP8 kernel numerics.", y=-0.02)
    save(fig, "fig11_moe_freeze2")


def fig_composite_uplift(lens="jlens"):
    """fig4 restricted to items the model mostly gets wrong without filler and mostly right with 100 dots
    (10-shot T=1: <=2/8 -> >=6/8), above items that stay wrong (<=2/8 -> <=2/8) as a contrast."""
    targets = ["x", "c1x", "y", "c2y", "answer"]
    by = rollouts(10)
    k0 = np.array([by[i]["k0"]["n_correct"] for i in range(600)])
    d100 = np.array([by[i]["dots100"]["n_correct"] for i in range(600)])
    groups = [(f"uplift: \u22642/8 \u2192 \u22656/8", (k0 <= 2) & (d100 >= 6)),
              (f"stuck: \u22642/8 \u2192 \u22642/8", (k0 <= 2) & (d100 <= 2))]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4), sharey=True,
                             gridspec_kw=dict(width_ratios=[16, 22, 110], wspace=0.04, hspace=0.62))
    cols = np.array([to_rgb(TARGET_COL[t]) for t in targets])
    bg = np.array(to_rgb(SURF))
    for row, (gname, mask) in enumerate(groups):
        for j, c in enumerate(CONDS):
            ax = axes[row, j]
            z, h = load_decode(c, lens=lens)
            R = np.stack([target_rate(z, h, t, mask)[0] for t in range(5)])
            strength = np.clip(R.max(0) / 0.6, 0, 1)
            ax.imshow(bg + (cols[R.argmax(0)] - bg) * strength[..., None], origin="lower", aspect="auto",
                      interpolation="nearest")
            ax.set_ylim(17.5, R.shape[1] - 0.5)
            nq, nf, na = pos_labels(z)
            for x in (nq - 0.5, nq + nf - 0.5):
                ax.axvline(x, color=INK2, lw=0.8, ls="--")
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            w = R.shape[-1]
            ax.set_xticks([1, nq + (nf - 1) / 2, w - 1], ["question", "filler", "answer"], fontsize=8,
                          rotation=90 if w < 30 else 0)
            ax.set_title(COND_LAB[c], loc="center", fontsize=9)
        axes[row, 0].set_ylabel("layer")
        axes[row, 0].text(0, 1.22, f"{gname} at T=1 ({mask.sum()} items)", transform=axes[row, 0].transAxes,
                          fontsize=10, fontweight="bold", ha="left")
    handles = [mpl.patches.Patch(color=TARGET_COL[t], label=TARGET_LAB[t]) for t in targets]
    fig.legend(handles=handles, loc="upper left", ncol=5, bbox_to_anchor=(0.12, 1.0), fontsize=9)
    fig.suptitle("Chain map on the items filler rescues, vs items it doesn't", x=0.125, ha="left",
                 fontweight="bold", y=1.04)
    footnote(fig, "Items grouped by 10-shot T=1 correct samples out of 8, no filler \u2192 100 dots. Same encoding as "
                  "fig4: colour = chain value with the highest decode rate in that cell, opacity = that rate (full at "
                  f"\u226560%). Layers 0\u201317 cropped. {LENS_NOTE if lens == 'jlens' else LOGIT_NOTE}.", y=0.03)
    save(fig, "fig12_chain_composite_uplift")


if __name__ == "__main__":
    fig_accuracy()
    fig_items()
    fig_heatmaps()
    fig_composite()
    fig_ladder()
    fig_correct_wrong()
    fig_subset()
    fig_freeze()
    fig_freeze2()
    fig_composite_uplift()
    fig_lens_compare()
    fig_composite("logit_hchead", "fig10_chain_composite_logit")
