"""J-lens vs logit lens on the 50-item subset (10-shot): at the answer position, and at the best
position over the filler segment, how often each chain quantity is the top number (0..999) of the readout.
Numbers are read through their canonical column (texts.index(str(n))). Writes figures/fig_lens_compare.png
and prints a table."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "results/subset/10shot"
texts = json.load(open(SUB / "number_tokens.json"))["texts"]
col = {t: i for i, t in enumerate(texts)}
COLS = np.array([col[str(n)] for n in range(1000)])
items = [json.loads(l) for l in open(ROOT / "repo/data/tasks/eval.jsonl")]
subset = json.load(open(ROOT / "results/subset.json"))
TARGETS = ["x", "c1x", "y", "c2y", "answer"]
LENSES = {"J-lens": "jlens", "logit lens (hc_head)": "logitlens_hchead", "logit lens (mean)": "logitlens_mean"}
CONDS = ["k0", "dots10", "dots100"]


def hits(lens_dir, cond, group):
    """[targets, layers] fraction of items where the target is the top number: at the answer position,
    and anywhere in the window (question + filler + answer segments, max over positions)."""
    at_ans, anywhere = [], []
    for idx in subset[group]:
        z = np.load(SUB / lens_dir / cond / f"{idx:04d}.npz")
        lp = z["number_logprobs"][:, :, COLS]  # [L, S, 1000]
        top = lp.argmax(-1)  # [L, S]
        win = z["seg"] >= 2
        a, w = [], []
        for t in TARGETS:
            v = items[idx]["chain"][t]
            ok = v < 1000
            a.append(top[:, -1] == v if ok else np.full(top.shape[0], np.nan))
            w.append((top[:, win] == v).any(1) if ok else np.full(top.shape[0], np.nan))
        at_ans.append(a); anywhere.append(w)
    return np.nanmean(np.array(at_ans, float), 0), np.nanmean(np.array(anywhere, float), 0)


fig, axes = plt.subplots(len(LENSES), 6, figsize=(24, 10), sharey=True, constrained_layout=True)
colors = dict(zip(TARGETS, ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#222222"]))
print("best layer hit rate at the ANSWER position (uplift / no-uplift)")
for r, (name, d) in enumerate(LENSES.items()):
    for c, cond in enumerate(CONDS):
        for g, group in enumerate(("uplift", "no_uplift")):
            ans, anyw = hits(d, cond, group)
            ax = axes[r, 2 * c + g]
            for t, row in zip(TARGETS, ans):
                ax.plot(row, color=colors[t], label=t)
            ax.set_title(f"{name}\n{cond}, {group.replace('_', '-')} (answer pos)", fontsize=9)
            if r == len(LENSES) - 1:
                ax.set_xlabel("layer")
            if g == 0:
                print(f"  {name:22s} {cond:6s} " + "  ".join(f"{t}" for t in TARGETS))
            print(f"    {group:9s} answer pos: " + "  ".join(f"{np.nanmax(row):.2f}@L{int(np.nanargmax(row))}" for row in ans))
            print(f"    {group:9s} any window pos: " + "  ".join(f"{np.nanmax(row):.2f}@L{int(np.nanargmax(row))}" for row in anyw))
axes[0, 0].legend(fontsize=8)
axes[0, 0].set_ylabel("fraction of items: target is top number")
fig.suptitle("V4 Flash, 10-shot, 50-item subset: which chain value is the lens's top number at the answer position")
out = ROOT / "figures/fig_lens_compare.png"
fig.savefig(out, dpi=130)
print("wrote", out)
