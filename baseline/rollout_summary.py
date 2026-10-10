"""Summarise rollout_*.jsonl: greedy accuracy and T=1 pass rate per condition, paired against k0."""
import json, sys, math
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
by = {}
for r in rows:
    by.setdefault(r["idx"], {})[r["cond"]] = r
conds = sorted({r["cond"] for r in rows}, key=lambda c: int(c[4:]) if c.startswith("dots") else 0)
full = [v for v in by.values() if all(c in v for c in conds)]
print(f"{sys.argv[1]}: {len(full)} items with all of {conds}")
metrics = [("greedy acc", lambda r: float(r["greedy_correct"])),
           ("T=1 pass rate", lambda r: r["n_correct"] / max(1, len(r["samples"]))),
           ("T=1 parse rate", lambda r: r["n_parsed"] / max(1, len(r["samples"])))]
for name, f in metrics:
    base = np.array([f(v["k0"]) for v in full])
    line = f"  {name:15s} k0 {base.mean():.3f}"
    for c in conds[1:]:
        x = np.array([f(v[c]) for v in full]); d = x - base
        line += f" | {c} {x.mean():.3f} ({d.mean():+.3f} ± {d.std(ddof=1)/math.sqrt(len(d)):.3f})"
    print(line)
for c in conds[1:]:
    g = [(v["k0"]["greedy_correct"], v[c]["greedy_correct"]) for v in full]
    print(f"  greedy flips k0 -> {c}: wrong->right {sum(1 for x, y in g if not x and y)}, right->wrong {sum(1 for x, y in g if x and not y)}")
