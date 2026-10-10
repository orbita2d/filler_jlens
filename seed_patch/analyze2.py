"""Report for a seed_patch2.py run: report.md + summary.json in OUT/.   python analyze2.py OUT items.json

Host = the unseeded prompt. P(correct) = P(answer) x P(EOS | answer), teacher-forced. For each patch scope
(ans = answer position only, all = every position after the system prompt) and depth l:
  mean{n}   host patched with the mean over the first n seeds
  single    host patched with one seed (averaged over the donor seeds) - patching without averaging
Gains are against the unpatched host; "averaging gain" = mean{n} - single. Ensembles: unpatched seeded runs."""
import json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

out, items_f = Path(sys.argv[1]), sys.argv[2]
meta = json.load(open(items_f))["meta"]
cfg = json.load(open(out / "config.json"))
N, ND = cfg["N"], cfg["n_donors"]
ns = [int(x) for x in cfg["ns"].split(",")]
depths = [int(x) for x in cfg["depths"].split(",")]
rows = defaultdict(dict)
for l in open(out / "rows.jsonl"):
    r = json.loads(l)
    rows[r["idx"]][(r["host"], r["row"])] = r
idxs = sorted(rows)
P = lambda r: math.exp(r["logp_correct"])
nrm = lambda x: (lambda y: y - np.logaddexp.reduce(y))(x.astype(np.float64))
kl = lambda p, q: float((np.exp(p) * (p - q)).sum())
ENS_N = sorted({1, 4, 16, 50, 100} & set(range(1, N + 1)))
SCOPES = ["ans", "all"]

S = []
for i in idxs:
    R = rows[i]; ans = next(iter(R.values()))["answer"]
    s = {"idx": i, "group": meta[str(i)]["group"], "answer": ans, "none": P(R[("none", "base")]),
         "dots100": P(R[("dots100", "base")]), "seed": np.mean([P(R[(str(h), "seed")]) for h in range(N)])}
    z = np.load(out / "dist" / f"{i:04d}.npz")
    hb = nrm(z["none|base"])
    for sc in SCOPES:
        for d in depths:
            for n in ns:
                s[f"mean{n}_{sc}@{d}"] = P(R[("none", f"mean{n}_{sc}@{d}")])
            s[f"single_{sc}@{d}"] = np.mean([P(R[("none", f"k1_{sc}@{d}/{k}")]) for k in range(ND)])
            s[f"kl_host_single_{sc}@{d}"] = np.median([kl(hb, nrm(z[f"none|k1_{sc}@{d}/{k}"])) for k in range(ND)])
            s[f"kl_donor_single_{sc}@{d}"] = np.median([kl(nrm(z[f"{k}|seed"]), nrm(z[f"none|k1_{sc}@{d}/{k}"])) for k in range(ND)])
            s[f"kl_host_mean{ns[-1]}_{sc}@{d}"] = kl(hb, nrm(z[f"none|mean{ns[-1]}_{sc}@{d}"]))
    s["kl_host_seed"] = np.median([kl(hb, nrm(z[f"{k}|seed"])) for k in range(ND)])
    s["kl_seedpair"] = np.median([kl(nrm(z[f"{k}|seed"]), nrm(z[f"{(k + 1) % N}|seed"])) for k in range(ND)])
    Ls = np.stack([nrm(z[f"{h}|seed"]) for h in range(N)]); pr = np.exp(Ls)
    for n in ENS_N:
        Ln, pn = Ls[:n], pr[:n]
        s[f"ens{n}_single"] = pn[:, ans].mean()
        s[f"ens{n}_single_greedy"] = (Ln.argmax(1) == ans).mean()
        s[f"ens{n}_vote"] = float(np.bincount(Ln.argmax(1), minlength=1000).argmax() == ans)
        s[f"ens{n}_probavg"] = pn.mean(0)[ans]
        s[f"ens{n}_logitavg"] = math.exp(nrm(Ln.mean(0))[ans])
        s[f"ens{n}_mostconf"] = pn[Ln.max(1).argmax(), ans]
        s[f"ens{n}_bestof"] = pn[:, ans].max()
    s["none_greedy"] = float(hb.argmax() == ans)
    s["dots100_greedy"] = float(z["dots100|base"].argmax() == ans)
    S.append(s)

groups = {g: [s for s in S if s["group"] == g] for g in ("uplift", "control")}
groups = {g: v for g, v in groups.items() if v}
col = lambda G, k: np.array([s[k] for s in G], float)
se = lambda x: float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def pone(x):
    return float(stats.ttest_1samp(x, 0, alternative="greater").pvalue) if len(x) > 1 and x.std() > 0 else float("nan")


summary = {"config": cfg, "n_items": len(S), "groups": {}}
L = [f"# Seed-mean patching, round 2 (unseeded host): `{out.name}`", "",
     f"{len(S)} items. Host = the no-seed 10-shot prompt. Its residuals after layers 0..l are replaced by the mean over "
     f"the first n seeds' (n = {', '.join(map(str, ns))}), either at the answer position only (`ans`) or at every "
     f"position after the system prompt (`all`). `single` = the same patch from one seed, averaged over {ND} donor "
     f"seeds. Ensembles use the unpatched seeded runs of the first n of {N} seeds. P = P(answer) x P(EOS | answer).", ""]
for g, G in groups.items():
    gs = {"n": len(G), "rows": {}, "ens": {}}
    base = col(G, "none")
    for k in ("none", "dots100", "seed", "none_greedy", "dots100_greedy"):
        gs[k] = float(col(G, k).mean())
    L += [f"## {g} items (n = {len(G)})", "",
          f"Unpatched host (no seed) {gs['none']:.3f}; single seeds {gs['seed']:.3f}; 100 dots {gs['dots100']:.3f}.", ""]
    for sc in SCOPES:
        L += [f"**Patch scope `{sc}`**: P(correct) after patching (gain over the unpatched host; one-sided p, "
              f"Bonferroni over {len(depths)} depths, for mean{ns[-1]} − single)", "",
              "| depth | single seed | " + " | ".join(f"mean of {n}" for n in ns) + f" | mean{ns[-1]} − single |",
              "|---" * (len(ns) + 3) + "|"]
        for d in depths:
            cells = []
            for k in ["single"] + [f"mean{n}" for n in ns]:
                x = col(G, f"{k}_{sc}@{d}")
                gs["rows"][f"{k}_{sc}@{d}"] = {"P": float(x.mean()), "gain": float((x - base).mean()), "se": se(x - base)}
                cells.append(f"{x.mean():.3f} ({(x - base).mean():+.3f})")
            dx = col(G, f"mean{ns[-1]}_{sc}@{d}") - col(G, f"single_{sc}@{d}")
            p = min(1.0, pone(dx) * len(depths))
            gs["rows"][f"avg_gain_{sc}@{d}"] = {"mean": float(dx.mean()), "se": se(dx), "p_bonf": p}
            L.append(f"| {d} | " + " | ".join(cells) + f" | {dx.mean():+.3f} ± {se(dx):.3f} (p={p:.2g}) |")
        L.append("")
    L += ["**Output-level ensembles** (first n seeds; T=1 probabilities unless marked greedy):", "",
          "| n | single seed | single seed, greedy | vote (greedy) | prob avg | log-prob avg | most confident | best of n |",
          "|---|---|---|---|---|---|---|---|"]
    ks = ["single", "single_greedy", "vote", "probavg", "logitavg", "mostconf", "bestof"]
    for n in ENS_N:
        for k in ks:
            x = col(G, f"ens{n}_{k}")
            gs["ens"][f"{n}_{k}"] = {"mean": float(x.mean()), "se": se(x)}
        L.append(f"| {n} | " + " | ".join(f"{gs['ens'][f'{n}_{k}']['mean']:.3f}" for k in ks) + " |")
    L += ["", f"Greedy accuracy: no seed {gs['none_greedy']:.2f}, 100 dots {gs['dots100_greedy']:.2f}.", ""]
    summary["groups"][g] = gs

med = lambda k: float(np.median([s[k] for s in S]))
L += ["## Does the patch take over the output?", "",
      f"Median over items. The host vs a seeded run: KL {med('kl_host_seed'):.3f}; two seeded runs: {med('kl_seedpair'):.3f}. "
      "A patch that fully takes over moves the host to the donor: KL(host ‖ row) up, KL(donor ‖ row) towards 0.", "",
      "| depth | ans: KL(host‖single) | ans: KL(donor‖single) | all: KL(host‖single) | all: KL(donor‖single) |",
      "|---|---|---|---|---|"]
summary["takeover"] = {}
for d in depths:
    v = {f"{sc}_{w}": med(f"kl_{w}_single_{sc}@{d}") for sc in SCOPES for w in ("host", "donor")}
    summary["takeover"][d] = v
    L.append(f"| {d} | {v['ans_host']:.3f} | {v['ans_donor']:.3f} | {v['all_host']:.3f} | {v['all_donor']:.3f} |")
L.append("")
(out / "report.md").write_text("\n".join(L) + "\n")
summary["items"] = [{k: (float(v) if isinstance(v, (np.floating, np.bool_, float)) else v) for k, v in s.items()} for s in S]
json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
print("\n".join(L))
