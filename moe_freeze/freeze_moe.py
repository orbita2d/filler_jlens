"""Is the filler uplift an MoE routing artefact? Freeze expert selection on the filler tokens and the
answer position to what the model chose at the answer position WITHOUT filler, and see if the uplift survives.

Per item (10-shot by default, the setting where the repo's uplift lives):
  1. k0 (no filler), natural routing. Record, at every MoE layer, the routed experts (top-k indices) and
     their normalised gate weights at the answer position (the `</think>` token right before the answer).
  2. For dots10 and dots100, one teacher-forced forward per variant:
       natural      routing untouched (the baseline)
       freeze_idx   filler dots + answer position use the k0 answer-position experts; the gate weights over
                    those experts are recomputed from each token's own router scores (how hash routing works)
       freeze_idxw  same experts AND the k0 answer-position gate weights copied verbatim
       answer_only  freeze_idx at the answer position only, filler routed normally (control: separates the
                    answer token's own routing from the filler's)
     All MoE layers are frozen, including the 3 hash-routed ones (--skip-hash to leave those alone).
     Shared experts, attention and everything else run normally.
Every variant of a condition has the same length as its natural run, so the length-dependent kernel noise
found in step 0 is identical across them: differences are the intervention alone.

Output: OUT/freeze_{shots}shot.jsonl, one line per item x cond x variant: logp_correct (= expected pass rate
at T=1, log), greedy_correct (argmax answer then EOS), number_mass, top5 at the answer position, and, for
natural runs, the per-layer overlap between the filler tokens' natural experts and the frozen set.

    python freeze_moe.py --model M --repo R --out O [--shots 10] [--items 0-599] [--check 3]
"""
import argparse, json, re, time
from pathlib import Path
import numpy as np
import torch
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: common.py
import common as C

VARIANTS = ["natural", "freeze_idx", "freeze_idxw", "answer_only"]


class Router:
    """Wraps every MoE gate. mode None: pass through. mode "record": store indices/weights at `pos`.
    mode "freeze": at rows `pos`, replace indices with self.idx[l] and weights per `weights` ("own" | "copy")."""

    def __init__(self, model, skip_hash=False):
        self.mode, self.pos, self.weights = None, None, "own"
        self.idx, self.w, self.natural_idx = {}, {}, {}
        self.layers = []
        for l, layer in enumerate(model.model.layers):
            moe = layer.mlp
            if not hasattr(moe, "gate") or (skip_hash and getattr(moe, "is_hash", False)):
                continue
            self.layers.append(l)
            moe.gate.forward = self._wrap(l, moe.gate, moe.gate.forward)

    def _wrap(self, l, gate, orig):
        def forward(*args, **kwargs):
            logits, weights, indices = orig(*args, **kwargs)
            if self.mode == "record":
                self.idx[l] = indices[self.pos].clone()
                self.w[l] = weights[self.pos].clone()
            elif self.mode == "observe":  # natural experts at the filler rows, for overlap stats
                self.natural_idx[l] = indices[self.pos].clone()
            elif self.mode == "freeze":
                pos = self.pos.to(indices.device)
                idx = self.idx[l].to(indices.device).expand(len(pos), -1)
                indices = indices.clone(); weights = weights.clone()
                indices[pos] = idx
                if self.weights == "copy":
                    weights[pos] = self.w[l].to(weights.device, weights.dtype).expand(len(pos), -1)
                else:
                    s = gate.score_fn(logits[pos].float()).gather(1, idx)
                    s = s / (s.sum(-1, keepdim=True) + 1e-20)
                    weights[pos] = (s * gate.routed_scaling_factor).to(weights.dtype)
            return logits, weights, indices
        return forward


def is_dot(tok_str):
    return re.fullmatch(r"Ġ?\.(Ċ)*", tok_str) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--repo", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shots", type=int, default=10); ap.add_argument("--items", default="0-599")
    ap.add_argument("--conds", default="dots10,dots100"); ap.add_argument("--skip-hash", action="store_true")
    ap.add_argument("--check", type=int, default=3, help="sanity checks on the first N items (see below)")
    a = ap.parse_args()
    prompts, items, fewshot = C.load_repo(a.repo)
    lo, hi = map(int, a.items.split("-"))
    items = [it for it in items if lo <= it["idx"] <= hi]
    tok, model = C.load_model(a.model)
    dev = model.model.embed_tokens.weight.device
    R = Router(model, a.skip_hash)
    print(f"MoE layers wrapped: {len(R.layers)} ({R.layers[0]}..{R.layers[-1]})", flush=True)
    num_ids = torch.as_tensor(C.number_token_ids(tok)[0])
    eos = tok.convert_tokens_to_ids(C.EOS)
    out = Path(a.out) / f"freeze_{a.shots}shot.jsonl"
    done = {(r["idx"], r["cond"], r["variant"]) for r in map(json.loads, open(out))} if out.exists() else set()

    def run(ids, ans):
        """Teacher-forced prompt + answer tokens; metrics at the answer position."""
        x = torch.tensor([ids + ans], device=dev)
        lp = torch.log_softmax(model(input_ids=x, use_cache=False).logits[0, len(ids) - 1:].float(), -1).cpu()
        targets = ans + [eos]
        tl = [lp[j, t].item() for j, t in enumerate(targets)]
        return {"logp_correct": sum(tl), "greedy_correct": all(lp[j].argmax().item() == t for j, t in enumerate(targets)),
                "number_mass": lp[0, num_ids].exp().sum().item(),
                "top5": [[tok.convert_ids_to_tokens(i), round(v, 3)] for v, i in zip(*[t.tolist() for t in lp[0].topk(5)])]}, lp[0]

    t0, n = time.time(), 0
    with open(out, "a") as f, torch.inference_mode():
        for j, it in enumerate(items):
            ans = C.answer_ids(tok, it["answer"])
            ids0, _ = C.tokenize_prompt(tok, C.build_prompt(prompts, fewshot, it, "k0", a.shots)[0])
            R.mode, R.pos = "record", torch.tensor([len(ids0) - 1])
            m0, lp0 = run(ids0, ans)  # records the k0 answer-position experts
            R.mode = None
            if (it["idx"], "k0", "natural") not in done:
                f.write(json.dumps({"idx": it["idx"], "cond": "k0", "variant": "natural", **m0}) + "\n")
            if j < a.check:  # freezing k0's own answer position to its own experts must change nothing
                R.mode, R.weights = "freeze", "copy"
                m0b, lp0b = run(ids0, ans)
                R.mode = None
                print(f"  check item {it['idx']}: k0 self-freeze max |dlogp| {float((lp0 - lp0b).abs().max()):.3g} (expect 0)", flush=True)
            for cond in a.conds.split(","):
                ids, seg = C.tokenize_prompt(tok, C.build_prompt(prompts, fewshot, it, cond, a.shots)[0])
                toks = tok.convert_ids_to_tokens(ids)
                dots = [p for p in range(len(ids)) if seg[p] == C.SEG_FILLER and is_dot(toks[p])]
                assert len(dots) == C.CONDITIONS[cond][1], (cond, len(dots))
                for variant in VARIANTS:
                    if (it["idx"], cond, variant) in done:
                        continue
                    rec = {"idx": it["idx"], "cond": cond, "variant": variant, "n_frozen": 0}
                    if variant == "natural":
                        R.mode, R.pos = "observe", torch.tensor(dots + [len(ids) - 1])
                        m, _ = run(ids, ans)
                        # share of the filler dots' natural experts that are in the frozen set, per layer
                        rec["overlap_dots"] = {l: round(float(torch.isin(R.natural_idx[l][:-1], R.idx[l]).float().mean()), 3) for l in R.layers}
                        rec["overlap_answer"] = {l: round(float(torch.isin(R.natural_idx[l][-1], R.idx[l]).float().mean()), 3) for l in R.layers}
                    else:
                        pos = [len(ids) - 1] if variant == "answer_only" else dots + [len(ids) - 1]
                        R.mode, R.pos, R.weights = "freeze", torch.tensor(pos), ("copy" if variant == "freeze_idxw" else "own")
                        m, _ = run(ids, ans)
                        rec["n_frozen"] = len(pos)
                    R.mode = None
                    f.write(json.dumps({**rec, **m}) + "\n"); f.flush(); n += 1
            if j % 20 == 0:
                print(f"item {it['idx']}: {n} runs, {(time.time() - t0) / max(n, 1):.2f}s each", flush=True)


if __name__ == "__main__":
    main()
