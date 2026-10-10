"""Quick look at the condensed 10-shot lens dump: lens log-prob of the intermediate y and the answer
by layer, at the answer position and (max over) filler positions, for the 25 uplift vs 25 no-uplift items."""
import json, sys
from pathlib import Path
import numpy as np
R = Path(sys.argv[1])
sub = json.load(open(R / "subset.json"))
res = {(r["idx"], r["cond"]): r for r in map(json.loads, open(R / "10shot/results.jsonl"))}
def lp(z):  # uint8 -> logprob
    return -z["number_logprobs_q"].astype(np.float32) / 10
out = {}
for group in ("uplift", "no_uplift"):
    for cond in ("k0", "dots10", "dots100"):
        A, Y, Fy = [], [], []
        for idx in sub[group]:
            r = res[(idx, cond)]
            z = np.load(R / f"10shot/jlens/{cond}/{idx:04d}.npz")
            L = lp(z)  # [layers, S, 1110]
            y, a = r["y"], r["answer"]
            if a < 1000:
                A.append(L[:, -1, a])
            if y < 1000:
                Y.append(L[:, -1, y])
                f = z["seg"] == 3
                if cond != "k0" and f.sum():
                    Fy.append(L[:, f, y].max(1))
        out[(group, cond)] = (np.mean(A, 0), np.mean(Y, 0), np.mean(Fy, 0) if Fy else None)
layers = list(range(0, 42, 4)) + [41]
print("mean lens logP at the ANSWER position, layers", layers)
for (g, c), (A, Y, F) in out.items():
    print(f"  {g:9s} {c:7s} answer {np.round(A[layers], 1).tolist()}")
    print(f"  {g:9s} {c:7s} y      {np.round(Y[layers], 1).tolist()}")
    if F is not None:
        print(f"  {g:9s} {c:7s} y max over filler positions {np.round(F[layers], 1).tolist()}")
