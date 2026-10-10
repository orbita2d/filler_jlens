"""Pick 25 uplift + 25 matched no-uplift items from rollout_10shot.jsonl (T=1 pass rates, 8 samples).
Uplift: largest dots100 - k0 gain (ties broken by greedy flip wrong->right). No uplift: |gain| <= 1/8 and
greedy unchanged, chosen to match each uplift item's k0 pass rate as closely as possible.
Writes subset.json {"uplift": [...], "no_uplift": [...], "table": [...]}."""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
by = {}
for r in rows:
    by.setdefault(r["idx"], {})[r["cond"]] = r
pr = lambda r: r["n_correct"] / len(r["samples"])
items = []
for idx, v in by.items():
    if "k0" in v and "dots100" in v:
        items.append({"idx": idx, "k0": pr(v["k0"]), "dots100": pr(v["dots100"]), "gain": pr(v["dots100"]) - pr(v["k0"]),
                      "greedy_k0": v["k0"]["greedy_correct"], "greedy_dots100": v["dots100"]["greedy_correct"]})
up = sorted(items, key=lambda x: (-x["gain"], -(x["greedy_dots100"] - x["greedy_k0"])))[:25]
pool = [x for x in items if abs(x["gain"]) <= 1 / 8 and x["greedy_k0"] == x["greedy_dots100"] and x not in up]
none = []
for u in up:
    best = min(pool, key=lambda x: abs(x["k0"] - u["k0"]))
    none.append(best); pool.remove(best)
mean = lambda xs, k: sum(x[k] for x in xs) / len(xs)
print(f"uplift:    n={len(up)} k0 {mean(up,'k0'):.3f} dots100 {mean(up,'dots100'):.3f} min gain {min(x['gain'] for x in up):.3f}")
print(f"no uplift: n={len(none)} k0 {mean(none,'k0'):.3f} dots100 {mean(none,'dots100'):.3f}")
json.dump({"uplift": sorted(x["idx"] for x in up), "no_uplift": sorted(x["idx"] for x in none), "table": up + none},
          open(sys.argv[2], "w"), indent=1)
