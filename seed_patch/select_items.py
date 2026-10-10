"""items.json: the top 50 items by teacher-forced 100-dot uplift, plus 50 controls where 100 dots doesn't
help, matched (greedily, nearest first) on no-filler P(correct). Single-token answers only.
Source: results/10shot/results.jsonl (10-shot, teacher-forced, H200 run of 2026-10-08)."""
import json, math, sys
src = sys.argv[1] if len(sys.argv) > 1 else "../results/10shot/results.jsonl"
P = {}
for l in open(src):
    r = json.loads(l)
    if r["n_answer_tokens"] == 1:
        P.setdefault(r["idx"], {})[r["cond"]] = math.exp(r["logp_correct"])
P = {i: p for i, p in P.items() if {"k0", "dots100"} <= p.keys()}
up = {i: p["dots100"] - p["k0"] for i, p in P.items()}
top = sorted(up, key=lambda i: -up[i])[:50]
pool = [i for i in P if i not in top and abs(up[i]) < 0.05]
ctrl = []
for i in sorted(top, key=lambda i: P[i]["k0"]):
    j = min(pool, key=lambda j: abs(P[j]["k0"] - P[i]["k0"]))
    ctrl.append(j); pool.remove(j)
meta = {i: {"p_k0": round(P[i]["k0"], 4), "p_dots100": round(P[i]["dots100"], 4), "uplift": round(up[i], 4),
            "group": "uplift" if i in top else "control"} for i in top + ctrl}
json.dump({"items": top + ctrl, "uplift": top, "control": ctrl, "meta": meta, "source": src}, open("items.json", "w"), indent=1)
m = lambda xs, k: sum(meta[i][k] for i in xs) / len(xs)
print(f"{len(P)} single-token items; uplift: min {min(up[i] for i in top):.3f}, mean P k0 {m(top,'p_k0'):.3f} -> dots100 {m(top,'p_dots100'):.3f}")
print(f"controls: mean P k0 {m(ctrl,'p_k0'):.3f} dots100 {m(ctrl,'p_dots100'):.3f}, max |uplift| {max(abs(up[i]) for i in ctrl):.3f}")
