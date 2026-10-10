"""Report for a seed_patch.py run: report.md, summary.json, fig_depth.png in OUT/.

    python analyze.py OUT items.json

P(correct) = P(answer token) * P(EOS | answer) at T=1, from rows.jsonl. Gains are paired within host:
P(row) - P(base) on the same seeded host (both rows of one batch), averaged over the hosts of an item, so an
item is one observation. Tests are one-sided paired t-tests over items, Bonferroni over depths.
Output-level ensembles use dist/*.npz (log-probs of the numbers 0..999, renormalised over numbers)."""
import json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

out, items_f = Path(sys.argv[1]), sys.argv[2]
meta = json.load(open(items_f))
group = {int(i): m["group"] for i, m in meta["meta"].items()}
H200_uplift = {int(i): m["uplift"] for i, m in meta["meta"].items()}
cfg = json.load(open(out / "config.json"))
N = cfg["N"]

rows = defaultdict(dict)   # idx -> (host, row) -> record
for l in open(out / "rows.jsonl"):
    r = json.loads(l)
    rows[r["idx"]][(r["host"], r["row"])] = r
idxs = sorted(rows)
P = lambda r: math.exp(r["logp_correct"])
answer = {i: next(iter(rows[i].values()))["answer"] for i in idxs}
seeds = [str(s) for s in range(N)]
names = sorted({k[1] for i in idxs for k in rows[i] if k[0] == "0"})
depths = sorted({int(n.split("@")[1]) for n in names if "@" in n})
kinds = [k for k in ["mean", "mean_nm"] + sorted({n.split("@")[0] for n in names if n.startswith("k")}) if f"{k}@{depths[0]}" in names]


def item_stats(i):
    R = rows[i]
    s = {"idx": i, "group": group.get(i, "?"), "answer": answer[i],
         "none": P(R[("none", "base")]), "dots100": P(R[("dots100", "base")]) if ("dots100", "base") in R else float("nan"),
         "seed_base": np.mean([P(R[(h, "base")]) for h in seeds]),
         "seed_self": np.mean([P(R[(h, "self")]) for h in seeds]),
         "dup_ok": all(R[(h, "base")]["logp_correct"] == R[(h, "base2")]["logp_correct"] for h in seeds + ["none"]),
         "number_mass_seed": np.mean([R[(h, "base")]["number_mass"] for h in seeds]),
         "number_mass_none": R[("none", "base")]["number_mass"]}
    for k in kinds:
        for d in depths:
            s[f"{k}@{d}"] = np.mean([P(R[(h, f"{k}@{d}")]) for h in seeds])
    for d in depths:
        s[f"none_mean@{d}"] = P(R[("none", f"mean@{d}")])
        s[f"none_k1@{d}"] = P(R[("none", f"k1@{d}")])
    # output-level ensembles over the N seeded hosts (number distribution)
    z = np.load(out / "dist" / f"{i:04d}.npz")
    L = np.stack([z[f"{h}|base"].astype(np.float64) for h in seeds])          # [N, 1000]
    L = L - np.logaddexp.reduce(L, axis=1, keepdims=True)
    a = answer[i]
    pr = np.exp(L)
    s["ens_seed_num"] = pr[:, a].mean()                                         # individual seeds, number-renormalised
    s["ens_vote"] = float(np.bincount(L.argmax(1), minlength=1000).argmax() == a)
    s["ens_probavg"] = pr.mean(0)[a]
    la = L.mean(0); la -= np.logaddexp.reduce(la)
    s["ens_logitavg"] = math.exp(la[a])
    s["ens_mostconf"] = pr[L.max(1).argmax(), a]
    s["ens_bestof"] = pr[:, a].max()
    n0 = z["none|base"].astype(np.float64); n0 -= np.logaddexp.reduce(n0)
    s["ens_none_num"] = math.exp(n0[a])
    if "dots100|base" in z:
        dd = z["dots100|base"].astype(np.float64); dd -= np.logaddexp.reduce(dd)
        s["ens_dots100_num"] = math.exp(dd[a])
    return s


S = [item_stats(i) for i in idxs]
groups = {"uplift": [s for s in S if s["group"] == "uplift"], "control": [s for s in S if s["group"] == "control"]}
groups = {g: v for g, v in groups.items() if v}


def ttest(x):
    x = np.asarray(x, float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return float("nan"), float("nan")
    t, p = stats.ttest_1samp(x, 0.0, alternative="greater")
    return float(t), float(p)


summary = {"config": cfg, "n_items": len(S), "groups": {}}
lines = [f"# Seed-mean patching: `{out.name}`", "",
         f"{len(S)} items ({', '.join(f'{len(v)} {g}' for g, v in groups.items())}), {N} seeds, seed position "
         f"`{cfg['seed_pos']}`, prefix cache {'off' if cfg['no_prefix_cache'] else 'on'}. "
         "P = P(answer) x P(EOS | answer), teacher-forced, T=1. Gains are paired within each seeded host and "
         "averaged over hosts per item; t-tests are one-sided over items, Bonferroni over "
         f"{len(depths)} depths.", ""]

for g, G in groups.items():
    col = lambda k: np.array([s[k] for s in G], float)
    gs = {"n": len(G)}
    lines += [f"## {g} items (n = {len(G)})", "",
              "| quantity | mean P(correct) |", "|---|---|"]
    for k, lab in [("none", "no seed (baseline)"), ("seed_base", "individual seeds (mean over hosts)"),
                   ("seed_self", "self-patch (numerics floor)"), ("dots100", "100 dots, no seed (this pod)")]:
        gs[k] = float(np.nanmean(col(k)))
        lines.append(f"| {lab} | {gs[k]:.3f} |")
    lines += ["", f"H200 selection numbers for comparison: no filler {np.mean([meta['meta'][str(s['idx'])]['p_k0'] for s in G]):.3f}, "
              f"100 dots {np.mean([meta['meta'][str(s['idx'])]['p_dots100'] for s in G]):.3f}.", "",
              "**Patch gain vs the unpatched seeded host** (mean over items, with one-sided p, Bonferroni-corrected):", "",
              "| depth | " + " | ".join(kinds) + " | no-seed host + seed mean |", "|---" * (len(kinds) + 2) + "|"]
    base = col("seed_base")
    gs["gain"] = {}
    for d in depths:
        cells = []
        for k in kinds:
            x = col(f"{k}@{d}") - base
            t, p = ttest(x)
            gs["gain"][f"{k}@{d}"] = {"mean": float(x.mean()), "se": float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan"), "p_bonf": min(1.0, p * len(depths)) if p == p else p}
            cells.append(f"{x.mean():+.3f} ± {gs['gain'][f'{k}@{d}']['se']:.3f} (p={gs['gain'][f'{k}@{d}']['p_bonf']:.2g})")
        x = col(f"none_mean@{d}") - col("none")
        gs["gain"][f"none_mean@{d}"] = {"mean": float(x.mean()), "se": float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")}
        cells.append(f"{x.mean():+.3f}")
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    lines += ["", "**Output-level ensembles over the seeds** (P(answer) within the number distribution):", "",
              "| ensemble | mean P | gain vs individual seeds |", "|---|---|---|"]
    gs["ens"] = {}
    ref = col("ens_seed_num")
    for k, lab in [("ens_none_num", "no seed"), ("ens_seed_num", "individual seeds"), ("ens_vote", "majority vote (0/1)"),
                   ("ens_probavg", "probability average"), ("ens_logitavg", "log-prob average"),
                   ("ens_mostconf", "most confident seed"), ("ens_bestof", f"best of {N} (oracle)"),
                   ("ens_dots100_num", "100 dots")]:
        if k in G[0]:
            gs["ens"][k] = float(col(k).mean())
            lines.append(f"| {lab} | {col(k).mean():.3f} | {(col(k) - ref).mean():+.3f} |")
    lines += ["", f"Number mass: seeded hosts {col('number_mass_seed').mean():.4f}, no seed {col('number_mass_none').mean():.4f}. "
              f"Duplicate rows identical on {sum(s['dup_ok'] for s in G)}/{len(G)} items.", ""]
    summary["groups"][g] = gs

# Specificity: uplift vs control, and item-level correlation with the dots uplift.
if len(groups) == 2:
    lines += ["## Uplift vs control", "", "| depth | mean gain, uplift − control (Welch, one-sided p) | Spearman(gain, dots100 uplift on this pod) |", "|---|---|---|"]
    up_here = np.array([s["dots100"] - s["none"] for s in S])
    summary["specificity"] = {}
    for d in depths:
        xu = np.array([s[f"mean@{d}"] - s["seed_base"] for s in groups["uplift"]])
        xc = np.array([s[f"mean@{d}"] - s["seed_base"] for s in groups["control"]])
        t, p = stats.ttest_ind(xu, xc, equal_var=False, alternative="greater")
        rho, pr = stats.spearmanr([s[f"mean@{d}"] - s["seed_base"] for s in S], up_here)
        summary["specificity"][d] = {"diff": float(xu.mean() - xc.mean()), "p": float(p), "rho": float(rho), "p_rho": float(pr)}
        lines.append(f"| {d} | {xu.mean() - xc.mean():+.3f} (p={p:.2g}) | {rho:+.2f} (p={pr:.2g}) |")
    lines.append("")

# Geometry: does the seed spread reach the answer position, and does the seed mean point towards dots100?
rp = sorted((out / "resid").glob("*.pt"))
if rp:
    import torch
    geo = defaultdict(list)
    for f in rp:
        r = torch.load(f)
        Rs = torch.stack([r[s].float() for s in seeds]).flatten(2)      # [N, L, 4D]
        m = Rs.mean(0)
        none = r["none"].float().flatten(1)
        spread = (Rs - m).norm(dim=-1).mean(0)                           # [L]
        geo["spread"].append(spread); geo["none_to_mean"].append((m - none).norm(dim=-1))
        geo["none_norm"].append(none.norm(dim=-1))
        if "dots100" in r:
            dots = r["dots100"].float().flatten(1)
            dv = dots - none
            geo["dots_dist"].append(dv.norm(dim=-1))
            geo["cos_mean_dots"].append(torch.nn.functional.cosine_similarity(m - none, dv, dim=-1))
    med = {k: torch.stack(v).median(0).values for k, v in geo.items()}
    lines += ["## Geometry at the answer position (median over items)", "",
              "Seed spread = mean distance of a seed's residual from the seed mean. cos = cosine between "
              "(seed mean − no seed) and (dots100 − no seed).", "",
              "| layer | ‖none‖ | seed spread | ‖seed mean − none‖ | ‖dots100 − none‖ | spread / dots dist | cos |", "|---|---|---|---|---|---|---|"]
    summary["geometry"] = {k: v.tolist() for k, v in med.items()}
    for l in sorted(set(depths) | {0}):
        if "dots_dist" in med:
            lines.append(f"| {l} | {med['none_norm'][l]:.0f} | {med['spread'][l]:.1f} | {med['none_to_mean'][l]:.1f} | "
                         f"{med['dots_dist'][l]:.1f} | {med['spread'][l] / med['dots_dist'][l]:.2f} | {med['cos_mean_dots'][l]:+.2f} |")
    lines.append("")

if (out / "check.jsonl").exists() and (out / "check.jsonl").stat().st_size:
    ch = [json.loads(l) for l in open(out / "check.jsonl")]
    summary["checks"] = ch
    lines += ["## Numerics checks", "", "| item | dup rows max|Δ| | self-floor max|Δ| | self-floor KL (no seed) | path KL (numbers) | path argmax match |",
              "|---|---|---|---|---|---|"]
    for c in ch:
        lines.append(f"| {c['idx']} | {c['dup_rows_maxabs']:.3g} | {c['self_floor_maxabs']:.3g} | {c['self_floor_kl_none']:.3g} | {c['path_kl_numbers']:.3g} | {c['path_argmax_match']} |")
    lines.append("")

lines += ["## Per item", "", "| idx | group | answer | none | seeds | " + " | ".join(f"mean@{d}" for d in depths) + " | dots100 |",
          "|---" * (5 + len(depths) + 1) + "|"]
for s in S:
    lines.append(f"| {s['idx']} | {s['group']} | {s['answer']} | {s['none']:.3f} | {s['seed_base']:.3f} | "
                 + " | ".join(f"{s[f'mean@{d}']:.3f}" for d in depths) + f" | {s['dots100']:.3f} |")

(out / "report.md").write_text("\n".join(lines) + "\n")
summary["items"] = [{k: (float(v) if isinstance(v, (np.floating, np.bool_)) else v) for k, v in s.items()} for s in S]
json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4.2), sharey=True, squeeze=False)
colors = {"mean": "#2a6fdb", "mean_nm": "#7aa6ef", "k1": "#d9822b", "k4": "#9b59b6"}
for ax, (g, gs) in zip(axes[0], summary["groups"].items()):
    for k in kinds:
        m = [gs["gain"][f"{k}@{d}"]["mean"] for d in depths]
        e = [gs["gain"][f"{k}@{d}"]["se"] for d in depths]
        ax.errorbar(depths, m, yerr=e, marker="o", ms=4, capsize=2, label=k, color=colors.get(k))
    ax.axhline(0, color="k", lw=0.6)
    ens_gain = gs["ens"].get("ens_logitavg", np.nan) - gs["ens"].get("ens_seed_num", np.nan)
    ax.axhline(ens_gain, color="#2a6fdb", ls=":", lw=1, label="log-prob avg (output)")
    ax.axhline(gs["dots100"] - gs["seed_base"], color="#2e9e5b", ls="--", lw=1, label="100 dots − seeds")
    ax.set_title(f"{g} items (n={gs['n']})"); ax.set_xlabel("patched layers 0..l")
    ax.grid(alpha=0.3)
axes[0][0].set_ylabel("ΔP(correct) vs unpatched seeded host"); axes[0][-1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(out / "fig_depth.png", dpi=130)
print((out / "report.md").read_text())
