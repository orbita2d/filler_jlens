"""Seed-mean patching. Does averaging noisy answer-token computations reproduce the filler uplift?

Per item (10-shot, no filler):
  1. For each of N "seed" contexts (a fixed-length WikiText passage inserted before the 10 examples),
     run the prompt and record the residual (all 43 layers x 4 hc streams) and the answer distribution
     at the answer position (the final `</think>` token).
  2. For each seeded context as host, re-run ONLY the answer position with its residual replaced, at
     layers 0..l, by the mean over the other seeds (leave-one-out), plus controls. Everything except the
     answer position comes from the host's own KV cache, so each patched run is a one-token decode.
  3. Teacher-force the answer token after each row to get P(EOS), so P(correct) = P(ans) * P(EOS | ans).

Why it is cheap (see PLAN.md for the numbers):
  * The 10 few-shot examples are the same for every item, so system + seed + examples (~880 tokens) is
    prefilled ONCE per seed and its KV cache reused; each item only adds its ~170-token test turn.
  * Patching only touches the answer position, so all patched variants of a host are rows of ONE batched
    single-token decode on that host's cache (~50 rows), instead of ~50 full forward passes.

Rows per seeded host h (depth l in --depths; "up to l" = decoder-layer outputs 0..l replaced):
  base, base2   unpatched (two identical rows: a row-determinism check)
  self          h's own residuals from its B=1 capture run, all layers (numerics floor: B=1 vs batched)
  mean@l        mean over all other seeds (N-1)
  mean_nm@l     same, rescaled per stream and layer to the seeds' mean RMS (the mean of noisy vectors
                is shorter than any of them; this checks the effect isn't a norm artefact)
  k{k}@l        mean over k random other seeds, for k in --ks. k=1 is a swap: patching per se, no averaging
Host "none" (the plain no-seed prompt, the baseline): base, base2, self, mean@l over all N seeds, k1@l.
Reference "dots100" (no seed, 100 dots, full prefill): unpatched only, for the uplift on this hardware.

Outputs in OUT/:
  rows.jsonl        one line per item x host x row: logp_ans, logp_eos, logp_correct, argmax, number_mass
  dist/{idx}.npz    log-probs of the numbers 0..999 at the answer position for every row (float16),
                    keyed "{host}|{row}", for output-level ensembles (vote, logit average, best-of-N)
  resid/{idx}.pt    answer-position residuals [L, 4, D] bf16 for every seed, none, dots100 (--save-resid)
  check.jsonl       the sanity checks (first --check items)

    python seed_patch.py --model M --repo R --seeds seeds.json --items items.json --out O
"""
import argparse, copy, json, random, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "filler_jlens"))  # pod layout: ~/fj/filler_jlens -> code
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo layout: common.py at the root
import common as C  # noqa: E402

C.CONDITIONS.setdefault("dots100", ("dots", 100))


# ---------------------------------------------------------------------------------------------
# Hooks: capture or patch the residual after each decoder layer, at the last sequence position.
# ---------------------------------------------------------------------------------------------
class Hooks:
    def __init__(self, model):
        self.mode = None          # None | "capture" | "patch"
        self.cap = {}             # layer -> [B, 4, D] at the last position
        self.depth = None         # [B] long: patch layers 0..depth[b]; -1 = untouched
        self.vals = None          # [B, L, 4, D]
        self.handles = [layer.register_forward_hook(self._hook(l)) for l, layer in enumerate(model.model.layers)]

    def _hook(self, l):
        def f(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out          # [B, S, 4, D]
            if self.mode == "capture":
                self.cap[l] = h[:, -1].detach().clone()
            elif self.mode == "patch":
                rows = (self.depth >= l).nonzero().flatten()
                if len(rows):
                    h = h.clone()
                    h[rows.to(h.device), -1] = self.vals[rows, l].to(h.device, h.dtype)
                    return (h, *out[1:]) if isinstance(out, tuple) else h
            return None
        return f

    def stacked(self):
        return torch.stack([self.cap[l][0].cpu() for l in sorted(self.cap)])  # [L, 4, D]


# ---------------------------------------------------------------------------------------------
# KV cache helpers. V4's cache layers carry compressor buffers / compressed entries / overlap state in
# dicts besides keys/values; transformers' batch_repeat_interleave only repeats keys/values, so expand
# every batch-first tensor ourselves.
# ---------------------------------------------------------------------------------------------
def _rep(t, B):
    assert t.shape[0] == 1, f"cache tensor with batch {t.shape[0]}"
    return t.repeat(B, *[1] * (t.dim() - 1))


def expand_cache(cache, B):
    c = copy.deepcopy(cache)
    for layer in c.layers:
        shared = getattr(layer, "values", None) is getattr(layer, "keys", None)
        for k, v in list(vars(layer).items()):
            if torch.is_tensor(v) and v.dim() > 0:
                setattr(layer, k, _rep(v, B))
            elif isinstance(v, dict):
                for n, t in v.items():
                    if torch.is_tensor(t) and t.dim() > 0:
                        v[n] = _rep(t, B)
        if shared and getattr(layer, "keys", None) is not None:
            layer.values = layer.keys
    return c


# ---------------------------------------------------------------------------------------------
def canonical_number_ids(tok):
    """id of the canonical token for each n in 0..999 (never index-based: zero-padded forms exist)."""
    ids = [tok.convert_tokens_to_ids(str(n)) for n in range(1000)]
    assert len(set(ids)) == 1000 and tok.unk_token_id not in ids
    return torch.tensor(ids)


def split_prompt(text):
    """system | few-shot examples | test turn, split at <｜User｜> boundaries (special tokens, so each
    piece tokenizes the same alone as inside the whole prompt)."""
    a = text.index(C.USER)
    b = text.rfind(C.USER)
    return text[:a], text[a:b], text[b:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--repo", required=True)
    ap.add_argument("--seeds", required=True, help="seeds.json from make_seeds.py")
    ap.add_argument("--items", required=True, help="items.json from select_items.py, or a spec like 0-49")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seeds", type=int, default=16)
    ap.add_argument("--seed-pos", default="pre_examples", choices=["pre_examples", "pre_test"],
                    help="seed after the system prompt (default) or between the examples and the test turn")
    ap.add_argument("--depths", default="2,6,10,14,18,22,26,30,34,38,42")
    ap.add_argument("--ks", default="1,4")
    ap.add_argument("--no-prefix-cache", action="store_true", help="prefill every prompt from scratch")
    ap.add_argument("--no-dots-ref", action="store_true")
    ap.add_argument("--save-resid", action="store_true", default=True)
    ap.add_argument("--check", type=int, default=2, help="run sanity checks on the first N items")
    ap.add_argument("--limit", type=int, default=None, help="only the first N items (pilot)")
    ap.add_argument("--rng", type=int, default=0)
    a = ap.parse_args()

    prompts, all_items, fewshot = C.load_repo(a.repo)
    if a.items.endswith(".json"):
        sel = json.load(open(a.items))["items"]
    else:
        lo, hi = map(int, a.items.split("-")); sel = list(range(lo, hi + 1))
    sel = sel[: a.limit] if a.limit else sel
    items = [all_items[i] for i in sel]
    assert all(it["idx"] == i for it, i in zip(items, sel))
    seeds = json.load(open(a.seeds))["seeds"][: a.n_seeds]
    N = len(seeds)
    depths = [int(x) for x in a.depths.split(",")]
    ks = [int(x) for x in a.ks.split(",") if int(x) < N - 1]

    tok, model = C.load_model(a.model)
    dev = model.model.embed_tokens.weight.device
    L = len(model.model.layers)
    assert max(depths) < L
    H = Hooks(model)
    num_ids = canonical_number_ids(tok)
    all_num = torch.as_tensor(C.number_token_ids(tok)[0])
    eos = tok.convert_tokens_to_ids(C.EOS)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    sep = enc("\n\n")

    out = Path(a.out); (out / "dist").mkdir(parents=True, exist_ok=True); (out / "resid").mkdir(exist_ok=True)
    rows_f = open(out / "rows.jsonl", "a"); check_f = open(out / "check.jsonl", "a")
    done = {json.loads(l)["idx"] for l in open(out / "rows.jsonl")} if (out / "rows.jsonl").stat().st_size else set()
    json.dump(vars(a) | {"N": N, "L": L}, open(out / "config.json", "w"), indent=1)

    # Prompt pieces. The examples are the same for every item (asserted below).
    sys_t, ex_t, _ = split_prompt(C.build_prompt(prompts, fewshot, items[0], "k0", 10)[0])
    sys_ids, ex_ids = enc(sys_t), enc(ex_t)

    def prefix_ids(seed):
        if seed is None:
            return sys_ids + ex_ids
        s = seed["ids"]
        return sys_ids + sep + s + sep + ex_ids if a.seed_pos == "pre_examples" else sys_ids + ex_ids + s + sep

    @torch.inference_mode()
    def prefill(ids, cache=None):
        """Run ids (appending to cache if given); returns the cache. No logits needed."""
        o = model.model(input_ids=torch.tensor([ids], device=dev), past_key_values=cache, use_cache=True)
        return o.past_key_values

    @torch.inference_mode()
    def step(cache, tok_id, B):
        """One-token decode of tok_id for B rows on cache (mutated). Returns float32 log-probs [B, V]."""
        x = torch.full((B, 1), tok_id, device=dev)
        lg = model(input_ids=x, past_key_values=cache, use_cache=True).logits[:, -1].float()
        return torch.log_softmax(lg, -1)

    def metrics(lp1, lp2, ans):
        """lp1 [B, V] at the answer position, lp2 [B, V] after teacher-forcing ans."""
        la, le = lp1[:, ans].cpu(), lp2[:, eos].cpu()
        nm = lp1[:, all_num.to(lp1.device)].exp().sum(-1).cpu()
        am = lp1.argmax(-1).cpu()
        return [{"logp_ans": float(la[b]), "logp_eos": float(le[b]), "logp_correct": float(la[b] + le[b]),
                 "argmax": tok.convert_ids_to_tokens(int(am[b])), "number_mass": float(nm[b])} for b in range(len(la))], \
            lp1[:, num_ids.to(lp1.device)].cpu().to(torch.float16).numpy()

    # Shared prefixes, one per seed (+ the no-seed baseline).
    hosts = list(range(N)) + ["none"]
    pre_cache = {}
    if not a.no_prefix_cache:
        t0 = time.time()
        for h in hosts:
            pre_cache[h] = prefill(prefix_ids(None if h == "none" else seeds[h]))
        print(f"prefilled {len(hosts)} prefixes ({len(prefix_ids(seeds[0]))} tokens each) in {time.time() - t0:.0f}s", flush=True)

    rng = random.Random(a.rng)
    t0 = time.time()
    for j, it in enumerate(items):
        if it["idx"] in done:
            continue
        text = C.build_prompt(prompts, fewshot, it, "k0", 10)[0]
        s_t, e_t, test_t = split_prompt(text)
        assert s_t == sys_t and e_t == ex_t, "few-shot examples differ between items"
        test_ids = enc(test_t)
        assert enc(text) == sys_ids + ex_ids + test_ids, "split tokenization differs from the full prompt"
        ans = C.answer_ids(tok, it["answer"])
        assert len(ans) == 1, "4-digit answers are excluded by select_items.py"
        ans = ans[0]
        ctx, last = test_ids[:-1], test_ids[-1]

        # Phase 1: per host, context cache and an unpatched B=1 capture of the answer position.
        tp = [time.time()]
        cache, resid = {}, {}
        for h in hosts:
            if a.no_prefix_cache:
                cache[h] = prefill(prefix_ids(None if h == "none" else seeds[h]) + ctx)
            else:
                cache[h] = prefill(ctx, copy.deepcopy(pre_cache[h]))
            c1 = copy.deepcopy(cache[h])
            H.mode, H.cap = "capture", {}
            step(c1, last, 1)
            H.mode = None
            resid[h] = H.stacked()                                    # [L, 4, D] bf16
            del c1
        R = torch.stack([resid[s].float() for s in range(N)])        # [N, L, 4, D] fp32
        rms = R.pow(2).mean(-1, keepdim=True).sqrt()                 # [N, L, 4, 1]

        def mean_of(idx, norm_match=False):
            m = R[idx].mean(0)
            if norm_match:
                m = m * rms[idx].mean(0) / (m.pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)
            return m

        # Phase 2: batched patched decode per host.
        tp.append(time.time())
        dist = {}
        for h in hosts:
            others = [s for s in range(N) if s != h]
            specs = [("base", -1, None), ("base2", -1, None), ("self", L - 1, resid[h].float())]
            for l in depths:
                specs.append((f"mean@{l}", l, mean_of(others)))
                if h != "none":
                    specs.append((f"mean_nm@{l}", l, mean_of(others, True)))
                for k in (ks if h != "none" else [1]):
                    specs.append((f"k{k}@{l}", l, mean_of(rng.sample(others, k))))
            B = len(specs)
            vals = torch.zeros(B, L, *resid[h].shape[1:], dtype=torch.bfloat16)
            for b, (_, d, v) in enumerate(specs):
                if v is not None:
                    vals[b] = v.to(torch.bfloat16)
            cb = expand_cache(cache[h], B)
            H.mode, H.depth, H.vals = "patch", torch.tensor([d for _, d, _ in specs]), vals
            lp1 = step(cb, last, B)
            H.mode = None
            lp2 = step(cb, ans, B)                                   # teacher-forced answer -> P(EOS)
            ms, nd = metrics(lp1, lp2, ans)
            for b, (name, d, _) in enumerate(specs):
                rows_f.write(json.dumps({"idx": it["idx"], "answer": it["answer"], "host": str(h), "row": name, "depth": d, **ms[b]}) + "\n")
                dist[f"{h}|{name}"] = nd[b]
            del cb, cache[h]

        tp.append(time.time())
        if not a.no_dots_ref:
            ids_d, _ = C.tokenize_prompt(tok, C.build_prompt(prompts, fewshot, it, "dots100", 10)[0])
            cd = prefill(ids_d[:-1])
            H.mode, H.cap = "capture", {}
            lp1 = step(cd, ids_d[-1], 1)
            H.mode = None
            resid["dots100"] = H.stacked()
            ms, nd = metrics(lp1, step(cd, ans, 1), ans)
            rows_f.write(json.dumps({"idx": it["idx"], "answer": it["answer"], "host": "dots100", "row": "base", "depth": -1, **ms[0]}) + "\n")
            dist["dots100|base"] = nd[0]
        rows_f.flush()
        np.savez_compressed(out / "dist" / f"{it['idx']:04d}.npz", **dist)
        if a.save_resid:
            torch.save({str(k): v for k, v in resid.items()}, out / "resid" / f"{it['idx']:04d}.pt")

        tp.append(time.time())
        if j < a.check:
            check(it, tok, model, dev, check_f, dist, enc(text))
        tp.append(time.time())
        el = time.time() - t0
        ph = " ".join(f"{n}={b - a_:.1f}s" for n, a_, b in zip(["capture", "patch", "dots", "check"], tp, tp[1:]))
        print(f"item {it['idx']} ({j + 1}/{len(items)}): {el / (j + 1):.1f}s/item [{ph}] test_len={len(test_ids)}", flush=True)


def check(it, tok, model, dev, out_f, dist, ids):
    """Sanity checks on one item, written to check.jsonl and printed.
    - dup_rows:     base vs base2 in the same batch (must be 0: rows are computed independently)
    - self_floor:   base vs self (B=1 residuals pasted into a batched row): the numerics floor that any
                    patch effect must clear
    - path_kl:      no-seed prompt, prefix-cache + chunk + decode vs one full prefill: KL at the answer position.
                    Not expected to be 0 (FP8 numerics depend on chunking); should be well under the
                    ~0.3 nats a one-token length change causes."""
    def kl(p, q):
        p, q = torch.as_tensor(p, dtype=torch.float32), torch.as_tensor(q, dtype=torch.float32)
        return float((p.exp() * (p - q)).sum())
    r = {"idx": it["idx"]}
    r["dup_rows_maxabs"] = float(np.abs(dist["0|base"].astype(np.float32) - dist["0|base2"].astype(np.float32)).max())
    r["self_floor_maxabs"] = float(np.abs(dist["0|base"].astype(np.float32) - dist["0|self"].astype(np.float32)).max())
    r["self_floor_kl_none"] = kl(dist["none|base"], dist["none|self"])
    with torch.inference_mode():
        full = torch.log_softmax(model(input_ids=torch.tensor([ids], device=dev), use_cache=False).logits[0, -1].float(), -1)
    nid = canonical_number_ids(tok).to(full.device)
    f = full[nid].cpu().numpy()
    f = f - np.logaddexp.reduce(f)
    g = dist["none|base"].astype(np.float32); g = g - np.logaddexp.reduce(g)
    r["path_kl_numbers"] = kl(f, g)
    r["path_argmax_match"] = int(f.argmax() == g.argmax())
    print("CHECK", json.dumps(r), flush=True)
    out_f.write(json.dumps(r) + "\n"); out_f.flush()


if __name__ == "__main__":
    main()
