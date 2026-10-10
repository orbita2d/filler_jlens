"""Summarise freeze_*shot.jsonl: mean P(correct) and greedy accuracy per cond x variant, uplift over k0 with
paired SEs, and how much of the natural uplift each freeze keeps."""
import json, sys, math
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
by = {}
for r in rows:
    by.setdefault(r["idx"], {})[(r["cond"], r["variant"])] = r
keys = sorted({(r["cond"], r["variant"]) for r in rows})
full = [v for v in by.values() if all(k in v for k in keys)]
print(f"{len(full)} items complete")
P = lambda v, k: math.exp(v[k]["logp_correct"])
base = np.array([P(v, ("k0", "natural")) for v in full])
print(f"  k0 natural: P(correct) {base.mean():.3f}, greedy {np.mean([v[('k0','natural')]['greedy_correct'] for v in full]):.3f}")
for cond in sorted({c for c, _ in keys} - {"k0"}):
    nat = np.array([P(v, (cond, "natural")) for v in full]) - base
    for var in list(dict.fromkeys(r["variant"] for r in rows if r["cond"] == cond)):
        if (cond, var) not in keys:
            continue
        p = np.array([P(v, (cond, var)) for v in full]); d = p - base
        g = np.mean([v[(cond, var)]["greedy_correct"] for v in full])
        nm = np.mean([v[(cond, var)]["number_mass"] for v in full])
        keep = d.mean() / nat.mean() if nat.mean() else float("nan")
        print(f"  {cond:7s} {var:12s} P(correct) {p.mean():.3f}  uplift {d.mean():+.3f} ± {d.std(ddof=1)/math.sqrt(len(d)):.3f}"
              f"  (keeps {keep:5.0%} of natural)  greedy {g:.3f}  number mass {nm:.3f}")
    if "overlap_dots" not in full[0][(cond, "natural")]:
        continue
    ov = np.array([[v[(cond, "natural")]["overlap_dots"][l] for l in v[(cond, "natural")]["overlap_dots"]] for v in full])
    print(f"  {cond:7s} natural: share of dot tokens' experts already in the frozen set, mean over layers {ov.mean():.3f}")
