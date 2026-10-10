"""Overall rates for a seed_ens.py run: report.md + summary.json in OUT/.
    python analyze_ens.py OUT items.json ../results/10shot/results.jsonl

Per item: no seed, single seed (mean over the N seeds), ensembles of the first n seeds, 100 dots; T=1 P(correct)
from the teacher-forced logits and greedy accuracy (number distribution, renormalised over 0..999).
Items are also binned by the H200 run's teacher-forced 100-dot uplift (the selection criterion of items.json),
to show how much of the seed effect on the selected items is regression to the mean."""
import json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

out, items_f, h200_f = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sel = json.load(open(items_f))
group = {i: "uplift" for i in sel["uplift"]} | {i: "control" for i in sel["control"]}
h200 = defaultdict(dict)
for l in open(h200_f):
    r = json.loads(l)
    h200[r["idx"]][r["cond"]] = math.exp(r["logp_correct"])
cfg = json.load(open(out / "config.json")); N = cfg["N"]
rows = defaultdict(dict)
for l in open(out / "rows.jsonl"):
    r = json.loads(l)
    rows[r["idx"]][r["host"]] = r
nrm = lambda x: (lambda y: y - np.logaddexp.reduce(y))(x.astype(np.float64))
ENS_N = [n for n in (1, 4, 16, 50, 100) if n <= N]

S = []
for i in sorted(rows):
    R = rows[i]; ans = R["none"]["answer"]
    z = np.load(out / "dist" / f"{i:04d}.npz")
    Ls = np.stack([nrm(z[f"{h}|seed"]) for h in range(N)]); pr = np.exp(Ls)
    s = {"idx": i, "group": group.get(i, "rest"),
         "h200_k0": h200[i].get("k0"), "h200_dots100": h200[i].get("dots100"),
         "none": math.exp(R["none"]["logp_correct"]), "dots100": math.exp(R["dots100"]["logp_correct"]),
         "seed": float(np.mean([math.exp(R[str(h)]["logp_correct"]) for h in range(N)])),
         "none_greedy": float(z["none|base"].argmax() == ans), "dots100_greedy": float(z["dots100|base"].argmax() == ans)}
    s["h200_uplift"] = s["h200_dots100"] - s["h200_k0"]
    for n in ENS_N:
        Ln, pn = Ls[:n], pr[:n]
        s[f"ens{n}_single"] = pn[:, ans].mean()
        s[f"ens{n}_single_greedy"] = (Ln.argmax(1) == ans).mean()
        s[f"ens{n}_vote"] = float(np.bincount(Ln.argmax(1), minlength=1000).argmax() == ans)
        s[f"ens{n}_probavg"] = pn.mean(0)[ans]
        s[f"ens{n}_logitavg"] = math.exp(nrm(Ln.mean(0))[ans])
        s[f"ens{n}_mostconf"] = pn[Ln.max(1).argmax(), ans]
        s[f"ens{n}_bestof"] = pn[:, ans].max()
    S.append({k: (float(v) if isinstance(v, (np.floating, np.bool_)) else v) for k, v in s.items()})

KS = ["none", "seed", "dots100", "none_greedy", "dots100_greedy"] + \
     [f"ens{n}_{k}" for n in ENS_N for k in ("single", "single_greedy", "vote", "probavg", "logitavg", "mostconf", "bestof")]


def stat(G):
    d = {"n": len(G)}
    for k in KS:
        x = np.array([s[k] for s in G], float)
        d[k] = {"mean": float(x.mean()), "se": float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")}
    return d


subsets = {"all": S} | {g: [s for s in S if s["group"] == g] for g in ("uplift", "control", "rest")}
# bins of the H200 selection criterion
edges = [-1.01, 0.05, 0.25, 0.5, 0.75, 1.01]
bins = {}
for lo, hi in zip(edges, edges[1:]):
    G = [s for s in S if lo < s["h200_uplift"] <= hi]
    if G:
        bins[f"({lo:g}, {hi:g}]"] = G
summary = {"config": cfg, "subsets": {k: stat(v) for k, v in subsets.items() if v},
           "uplift_bins": {k: stat(v) for k, v in bins.items()}, "items": S}
json.dump(summary, open(out / "summary.json", "w"), indent=1)

f = lambda d, k: f"{d[k]['mean']:.3f}"
L = [f"# Seed ensembles over all items: `{out.name}`", "",
     f"{len(S)} items (single-token answers), {N} WikiText seeds before the 10 examples, no patching. "
     "T=1 = teacher-forced P(answer) x P(EOS | answer) for single runs; ensembles use the number distribution "
     "renormalised over 0..999. Greedy = argmax over numbers.", "",
     "## Overall and by selection group", "",
     "| subset | n | no seed | single seed | 100 dots | log-prob avg (100) | most confident (16 / 50 / 100) | best of 100 | greedy: no seed | single seed | vote (16 / 50 / 100) | 100 dots |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for name, d in summary["subsets"].items():
    L.append(f"| {name} | {d['n']} | {f(d, 'none')} | {f(d, 'seed')} | {f(d, 'dots100')} | {f(d, f'ens{ENS_N[-1]}_logitavg')} | "
             + " / ".join(f(d, f"ens{n}_mostconf") for n in ENS_N if n >= 16) + f" | {f(d, f'ens{ENS_N[-1]}_bestof')} | "
             f"{f(d, 'none_greedy')} | {f(d, f'ens{ENS_N[-1]}_single_greedy')} | "
             + " / ".join(f(d, f"ens{n}_vote") for n in ENS_N if n >= 16) + f" | {f(d, 'dots100_greedy')} |")
L += ["", "## By the H200 run's 100-dot uplift (the selection criterion)", "",
      "Regression to the mean shows up as 'no seed' on this run rising above the H200 no-filler number in the top bins.", "",
      "| H200 uplift bin | n | H200 no filler | no seed | single seed | most confident (50) | log-prob avg (100) | 100 dots | H200 100 dots |",
      "|---|---|---|---|---|---|---|---|---|"]
for name, G in bins.items():
    d = summary["uplift_bins"][name]
    L.append(f"| {name} | {d['n']} | {np.mean([s['h200_k0'] for s in G]):.3f} | {f(d, 'none')} | {f(d, 'seed')} | "
             f"{f(d, 'ens50_mostconf') if 50 in ENS_N else '-'} | {f(d, f'ens{ENS_N[-1]}_logitavg')} | {f(d, 'dots100')} | "
             f"{np.mean([s['h200_dots100'] for s in G]):.3f} |")
(out / "report.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
