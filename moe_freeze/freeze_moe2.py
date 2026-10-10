"""MoE routing-freeze, round 2. Same setup as freeze_moe.py (10-shot, teacher-forced P(correct) at the answer
position, every variant of a condition has the same length as its natural run). New variants, per dots condition:

  natural            routing untouched
  first_idx          every filler dot uses the experts the FIRST dot picks (natural, same forward); weights over
                     them from each dot's own router scores. The answer position is routed normally.
  first_idxw         same, and the first dot's gate weights copied too
  ans_idx            round 1's freeze_idx: dots + answer position -> no-filler answer-token experts (hash layers frozen)
  ans_idx_nohash     same but hash-routed layers 0-2 left alone (freezing those swaps token identity)
  ans_idxw_nohash    + weights copied, hash layers left alone
  foreign_idx        control: dots + answer position -> ANOTHER item's no-filler answer-token experts (item idx+300)
  random_idx         control: dots + answer position -> random experts (k distinct per layer, seeded per item)
Controls freeze the hash layers like ans_idx does, so ans_idx vs foreign_idx vs random_idx is like for like.

Pass 1 runs no-filler for every item and stores its answer-position experts (needed for foreign_idx).
Output: OUT/freeze2_{shots}shot.jsonl (one line per item x cond x variant), OUT/k0_experts_{shots}shot.pt.

    python freeze_moe2.py --model M --repo R --out O [--shots 10] [--items 0-599] [--conds dots10,dots100]
"""
import argparse, json, time
from pathlib import Path
import torch
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: common.py
import common as C
from freeze_moe import is_dot

VARIANTS = ["natural", "first_idx", "first_idxw", "ans_idx", "ans_idx_nohash", "ans_idxw_nohash", "foreign_idx", "random_idx"]


class Router:
    """mode None: pass through. "record": store indices/weights at row self.rec_pos.
    "freeze": rows self.pos get experts self.idx[l] (or, with self.first, the experts of row self.first at this
    layer in this same forward); weights "own" (renormalised own scores) or "copy"."""

    def __init__(self, model):
        self.mode, self.skip_hash, self.first, self.weights = None, False, None, "own"
        self.idx, self.w, self.rec = {}, {}, {}
        self.layers, self.hash_layers, self.n_experts = [], [], None
        for l, layer in enumerate(model.model.layers):
            moe = layer.mlp
            if not hasattr(moe, "gate"):
                continue
            self.layers.append(l)
            if getattr(moe, "is_hash", False):
                self.hash_layers.append(l)
            self.n_experts = moe.gate.num_experts
            moe.gate.forward = self._wrap(l, moe, moe.gate.forward)

    def _wrap(self, l, moe, orig):
        gate = moe.gate

        def forward(*args, **kwargs):
            logits, weights, indices = orig(*args, **kwargs)
            if self.mode == "record":
                self.rec[l] = (indices[self.rec_pos].clone(), weights[self.rec_pos].clone())
            elif self.mode == "freeze" and not (self.skip_hash and getattr(moe, "is_hash", False)):
                pos = self.pos.to(indices.device)
                if self.first is not None:
                    idx, w = indices[self.first].clone(), weights[self.first].clone()
                else:
                    idx, w = self.idx[l].to(indices.device), self.w.get(l)
                idx = idx.reshape(1, -1).expand(len(pos), -1)
                indices = indices.clone(); weights = weights.clone()
                indices[pos] = idx
                if self.weights == "copy":
                    weights[pos] = w.reshape(1, -1).to(weights.device, weights.dtype).expand(len(pos), -1)
                else:
                    s = gate.score_fn(logits[pos].float()).gather(1, idx)
                    s = s / (s.sum(-1, keepdim=True) + 1e-20)
                    weights[pos] = (s * gate.routed_scaling_factor).to(weights.dtype)
            return logits, weights, indices
        return forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--repo", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shots", type=int, default=10); ap.add_argument("--items", default="0-599")
    ap.add_argument("--conds", default="dots10,dots100")
    ap.add_argument("--check", type=int, default=2)
    a = ap.parse_args()
    prompts, items, fewshot = C.load_repo(a.repo)
    lo, hi = map(int, a.items.split("-"))
    todo = [it for it in items if lo <= it["idx"] <= hi]
    tok, model = C.load_model(a.model)
    dev = model.model.embed_tokens.weight.device
    R = Router(model)
    print(f"MoE layers {len(R.layers)}, hash layers {R.hash_layers}, experts {R.n_experts}", flush=True)
    num_ids = torch.as_tensor(C.number_token_ids(tok)[0])
    eos = tok.convert_tokens_to_ids(C.EOS)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    def run(ids, ans):
        x = torch.tensor([ids + ans], device=dev)
        lp = torch.log_softmax(model(input_ids=x, use_cache=False).logits[0, len(ids) - 1:].float(), -1).cpu()
        targets = ans + [eos]
        tl = [lp[j, t].item() for j, t in enumerate(targets)]
        return {"logp_correct": sum(tl), "greedy_correct": all(lp[j].argmax().item() == t for j, t in enumerate(targets)),
                "number_mass": lp[0, num_ids].exp().sum().item(),
                "top5": [[tok.convert_ids_to_tokens(i), round(v, 3)] for v, i in zip(*[t.tolist() for t in lp[0].topk(5)])]}, lp[0]

    # pass 1: no-filler answer-position experts for every item (all 600: foreign_idx needs other items')
    kpath = out / f"k0_experts_{a.shots}shot.pt"
    k0 = torch.load(kpath) if kpath.exists() else {}
    res_path = out / f"freeze2_{a.shots}shot.jsonl"
    done = {(r["idx"], r["cond"], r["variant"]) for r in map(json.loads, open(res_path))} if res_path.exists() else set()
    t0 = time.time()
    with open(res_path, "a") as f, torch.inference_mode():
        for it in items:
            if it["idx"] in k0:
                continue
            ans = C.answer_ids(tok, it["answer"])
            ids0, _ = C.tokenize_prompt(tok, C.build_prompt(prompts, fewshot, it, "k0", a.shots)[0])
            R.mode, R.rec_pos = "record", torch.tensor([len(ids0) - 1])
            m0, _ = run(ids0, ans)
            R.mode = None
            k0[it["idx"]] = {l: (i.cpu(), w.cpu()) for l, (i, w) in R.rec.items()}
            if (it["idx"], "k0", "natural") not in done:
                f.write(json.dumps({"idx": it["idx"], "cond": "k0", "variant": "natural", **m0}) + "\n")
            if len(k0) % 100 == 0:
                torch.save(k0, kpath); print(f"pass 1: {len(k0)} items, {time.time()-t0:.0f}s", flush=True)
        torch.save(k0, kpath)
        print(f"pass 1 done: {len(k0)} items", flush=True)

        t0, n = time.time(), 0
        for j, it in enumerate(todo):
            ans = C.answer_ids(tok, it["answer"])
            gen = torch.Generator().manual_seed(7919 * it["idx"] + 1)
            rand = {l: torch.randperm(R.n_experts, generator=gen)[:k0[it["idx"]][l][0].numel()] for l in R.layers}
            foreign = k0[(it["idx"] + 300) % 600]
            for cond in a.conds.split(","):
                ids, seg = C.tokenize_prompt(tok, C.build_prompt(prompts, fewshot, it, cond, a.shots)[0])
                toks = tok.convert_ids_to_tokens(ids)
                dots = [p for p in range(len(ids)) if seg[p] == C.SEG_FILLER and is_dot(toks[p])]
                assert len(dots) == C.CONDITIONS[cond][1], (cond, len(dots))
                P = len(ids) - 1
                ref = None
                for variant in VARIANTS + (["first_idx_selfcheck"] if j < a.check else []):
                    if (it["idx"], cond, variant) in done:
                        continue
                    R.mode, R.skip_hash, R.first, R.weights, R.w = "freeze", False, None, "own", {}
                    if variant == "natural":
                        R.mode = None
                    elif variant.startswith("first"):
                        # first_idx_selfcheck freezes only the first dot to itself: must equal natural exactly
                        R.pos = torch.tensor(dots[:1] if variant.endswith("selfcheck") else dots[1:])
                        R.first = dots[0]
                        R.weights = "copy" if variant in ("first_idxw", "first_idx_selfcheck") else "own"
                    else:
                        R.pos = torch.tensor(dots + [P])
                        src = {"foreign_idx": foreign}.get(variant, k0[it["idx"]])
                        R.idx = {l: rand[l] for l in R.layers} if variant == "random_idx" else {l: src[l][0] for l in R.layers}
                        if "idxw" in variant:
                            R.weights, R.w = "copy", {l: src[l][1] for l in R.layers}
                        R.skip_hash = variant.endswith("nohash")
                    m, lp = run(ids, ans)
                    R.mode = None
                    if variant == "natural":
                        ref = lp
                    if variant.endswith("selfcheck"):
                        print(f"  check item {it['idx']} {cond}: first-dot self-freeze max |dlogp| {float((lp - ref).abs().max()):.3g} (expect 0)", flush=True)
                        continue
                    f.write(json.dumps({"idx": it["idx"], "cond": cond, "variant": variant, **m}) + "\n"); f.flush(); n += 1
            if j % 20 == 0:
                print(f"item {it['idx']}: {n} runs, {(time.time() - t0) / max(n, 1):.2f}s each", flush=True)


if __name__ == "__main__":
    main()
