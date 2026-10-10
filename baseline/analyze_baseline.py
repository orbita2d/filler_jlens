"""Summarise results.jsonl: mean P(correct) at T=1, greedy accuracy and format mass per
shots x condition, plus the paired dots100 - k0 difference.

    python analyze_baseline.py OUT
"""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main(out):
    rng = np.random.default_rng(0)
    for shots_dir in sorted(Path(out).glob("*shot")):
        rows = [json.loads(l) for l in open(shots_dir / "results.jsonl")]
        by = {}
        for r in rows:
            by.setdefault(r["cond"], {})[r["idx"]] = r
        print(f"== {shots_dir.name}")
        for cond, d in by.items():
            p = np.exp([r["logp_correct"] for r in d.values()])
            print(f"  {cond:8s} n={len(d)}  mean P(correct) {p.mean():.3f}  greedy acc "
                  f"{np.mean([r['greedy_correct'] for r in d.values()]):.3f}  number mass "
                  f"{np.mean([r['number_mass_first'] for r in d.values()]):.3f} (min "
                  f"{min(r['number_mass_first'] for r in d.values()):.3f})")
        if {"k0", "dots100"} <= set(by):
            idx = sorted(set(by["k0"]) & set(by["dots100"]))
            p0 = np.exp([by["k0"][i]["logp_correct"] for i in idx])
            p1 = np.exp([by["dots100"][i]["logp_correct"] for i in idx])
            diff = p1 - p0
            boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
            g0 = np.array([by["k0"][i]["greedy_correct"] for i in idx])
            g1 = np.array([by["dots100"][i]["greedy_correct"] for i in idx])
            b, c = int((~g0 & g1).sum()), int((g0 & ~g1).sum())
            print(f"  dots100 - k0 (paired, n={len(idx)}): P(correct) {diff.mean():+.3f} "
                  f"[95% CI {np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]; "
                  f"greedy wrong->right {b}, right->wrong {c}, McNemar p {mcnemar_p(b, c):.2g}")
            print(f"  items gaining >= 0.5 in P(correct): {int((diff >= 0.5).sum())}; "
                  f"losing >= 0.5: {int((diff <= -0.5).sum())}")


if __name__ == "__main__":
    main(sys.argv[1])
