"""Aggregate the condensed 10-shot J-lens dumps into paper-style decode maps (Brauer et al. 2026, Figs 3-4).

For each condition, take a right-aligned window over the test turn: the last 3 question tokens, the filler
segment ("Filler:" + dots) and the answer segment ("Answer:<Assistant></think>", answer position last).
At every (layer, position) we ask whether each chain quantity (x, c1x, y, c2y, answer) is the top numeric
token (0..999) of the lens readout. Numbers are
read through their canonical token (str(n)), restricted to 0..999. Ties at the uint8 quantisation (0.1 nat) count fractionally.

Two readouts: raw, and "residual" (subtract the cross-item mean log-prob per (layer, position, number)
before the argmax, the paper's stage-2 trick to strip always-high tokens).

Writes figures/cache/decode_{cond}.npz for the J-lens and decode_{lens}_{cond}.npz for the logit lenses
(`compute_decode.py logit_hchead logit_mean`). Reads results/ only.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "cache"
TARGETS = ["x", "c1x", "y", "c2y", "answer"]
N_Q = 3  # question tokens kept before the filler segment


def canon_index():
    """Column of each number 0..999 in the 1110-wide arrays. The columns are sorted by numeric value with
    zero-padded forms ('00', '007') interleaved, so column n is NOT the number n."""
    texts = json.load(open(RES / "10shot/number_tokens.json"))["texts"]
    col = {t: i for i, t in enumerate(texts)}
    return np.array([col[str(n)] for n in range(1000)])


def load_items():
    items = [json.loads(l) for l in open(ROOT / "repo/data/tasks/eval.jsonl")]
    roll = {}
    for l in open(RES / "rollout_10shot.jsonl"):
        r = json.loads(l)
        roll[(r["idx"], r["cond"])] = r
    return items, roll


def window(z):
    seg = z["seg"]
    nf, na = int((seg == 3).sum()), int((seg == 4).sum())
    w = N_Q + nf + na
    return w, nf, na


def main(lens="jlens"):
    OUT.mkdir(exist_ok=True)
    items, roll = load_items()
    cols = canon_index()
    for cond in ["k0", "dots10", "dots100"]:
        files = [RES / f"10shot/{lens}/{cond}/{i:04d}.npz" for i in range(600)]
        z0 = np.load(files[0])
        w, nf, na = window(z0)
        L = z0["number_logprobs_q"].shape[0]
        # pass 1: cross-item mean log-prob over numbers 0..999 per (layer, pos)
        mean = np.zeros((L, w, 1000))
        for f in files:
            q = np.load(f)["number_logprobs_q"][:, -w:, cols]
            mean += -q.astype(np.float32) / 10
        mean /= len(files)
        hits = {m: np.zeros((len(files), len(TARGETS), L, w), np.float32) for m in ("raw", "resid")}
        rank = np.zeros((len(files), len(TARGETS), L, w), np.float32)  # raw rank among 0..999 (0 = top)
        ans_lp = np.zeros((len(files), L), np.float32)  # lens logP(answer) at answer position (lower bound -25.5)
        final_lp = np.zeros(len(files), np.float32)
        valid = np.zeros((len(files), len(TARGETS)), bool)
        correct = np.zeros(len(files), bool)
        pass_rate = np.zeros(len(files), np.float32)
        for n, f in enumerate(files):
            z = np.load(f)
            assert window(z) == (w, nf, na), f
            lp = -z["number_logprobs_q"][:, -w:, cols].astype(np.float32) / 10
            r = roll[(n, cond)]
            correct[n] = r["greedy_correct"]
            pass_rate[n] = r["n_correct"] / len(r["samples"])
            for name, x in (("raw", lp), ("resid", lp - mean)):
                top = x.max(-1, keepdims=True)
                is_top = x >= top - 1e-6
                nties = is_top.sum(-1)
                for t, key in enumerate(TARGETS):
                    v = items[n]["chain"][key]
                    if v >= 1000:
                        continue
                    valid[n, t] = True
                    hits[name][n, t] = is_top[..., v] / nties
            for t, key in enumerate(TARGETS):
                v = items[n]["chain"][key]
                if v < 1000:
                    rank[n, t] = (lp > lp[..., v : v + 1]).sum(-1)
            a = items[n]["answer"]
            if a < 1000:
                ans_lp[n] = lp[:, -1, a]
                final_lp[n] = z["final_number_logprobs"][cols[a]]
            else:
                ans_lp[n] = np.nan
                final_lp[n] = np.nan
        np.savez_compressed(OUT / (f"decode_{cond}.npz" if lens == "jlens" else f"decode_{lens}_{cond}.npz"), hits_raw=hits["raw"], hits_resid=hits["resid"], rank=rank,
                            valid=valid, correct=correct, pass_rate=pass_rate, ans_lp=ans_lp, final_lp=final_lp,
                            seg=z0["seg"][-w:], n_q=N_Q, nf=nf, na=na, targets=np.array(TARGETS))
        print(lens, cond, "window", w, "filler", nf, "acc", correct.mean())


if __name__ == "__main__":
    for lens in sys.argv[1:] or ["jlens"]:
        main(lens)
