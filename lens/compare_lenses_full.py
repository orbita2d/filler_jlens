"""J-lens vs logit lens (hc_head, mean) on the condensed 10-shot dumps, all 600 items x 3 conditions:
fraction of items where each chain value is the top number (0..999) at the answer position, by layer,
split by greedy-correct vs wrong at that condition. Numbers read through texts.index(str(n)).
Writes figures/fig_lens_compare_full.png and figures/cache/lens_compare_full.npz."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
texts = json.load(open(RES / "10shot/number_tokens.json"))["texts"]
col = {t: i for i, t in enumerate(texts)}
COLS = np.array([col[str(n)] for n in range(1000)])
items = [json.loads(l) for l in open(ROOT / "repo/data/tasks/eval.jsonl")]
roll = {(r["idx"], r["cond"]): r for r in map(json.loads, open(RES / "rollout_10shot.jsonl"))}
TARGETS = ["x", "c1x", "y", "c2y", "answer"]
LENSES = {"J-lens": "jlens", "logit lens (hc_head)": "logit_hchead", "logit lens (mean)": "logit_mean"}
CONDS = ["k0", "dots10", "dots100"]
colors = dict(zip(TARGETS, ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#222222"]))

store = {}
fig, axes = plt.subplots(len(LENSES), len(CONDS), figsize=(16, 11), sharey=True, constrained_layout=True)
for r, (name, d) in enumerate(LENSES.items()):
    for c, cond in enumerate(CONDS):
        hit, ok = [], []
        for i in range(600):
            z = np.load(RES / f"10shot/{d}/{cond}/{i:04d}.npz")
            top = z["number_logprobs_q"][:, -1, COLS].argmin(-1)  # uint8 q = -10 logp: min q = max logp
            hit.append([top == items[i]["chain"][t] if items[i]["chain"][t] < 1000 else np.full(len(top), np.nan) for t in TARGETS])
            ok.append(roll[(i, cond)]["greedy_correct"])
        hit, ok = np.array(hit, float), np.array(ok)
        store[f"{d}_{cond}"] = hit
        ax = axes[r, c]
        for t, k in enumerate(TARGETS):
            ax.plot(np.nanmean(hit[ok, t], 0), color=colors[k], label=f"{k}")
            ax.plot(np.nanmean(hit[~ok, t], 0), color=colors[k], ls=":")
        ax.set_title(f"{name}, {cond}  (solid: greedy right n={ok.sum()}, dotted: wrong)", fontsize=9)
        if r == len(LENSES) - 1:
            ax.set_xlabel("layer")
        best = {k: f"{np.nanmax(np.nanmean(hit[:, t], 0)):.2f}@L{int(np.nanargmax(np.nanmean(hit[:, t], 0)))}" for t, k in enumerate(TARGETS)}
        print(f"{name:22s} {cond:8s} best layer (all items): {best}")
axes[0, 0].legend(fontsize=8); axes[0, 0].set_ylabel("fraction of items: value is the top number at the answer position")
fig.suptitle("V4 Flash, 10-shot, all 600 items: J-lens vs logit lens at the answer position")
fig.savefig(ROOT / "figures/fig_lens_compare_full.png", dpi=120)
np.savez_compressed(ROOT / "figures/cache/lens_compare_full.npz", **store)
print("wrote figures/fig_lens_compare_full.png")
