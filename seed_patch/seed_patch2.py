"""Seed-mean patching, round 2: patch the UNSEEDED prompt with the average over seeds, at the answer position
only or at every position, plus many-seed output ensembles.

Round 1 (seed_patch.py) patched seeded hosts at the answer position only. Its single-seed swap rows showed that a
swap at layers <= 18 leaves the output unchanged: later layers re-read the context through attention.

Host: the no-seed 10-shot prompt (system + examples + test turn). Every seeded prompt is system + seed + the same
examples + test turn, so each host position after the system prompt has a matching token in every seeded prompt
(same token, shifted by the seed length). The system positions come before the seed and are identical in every
prompt, so "all positions" = every position after the system prompt.

Rows on the host, depth l in --depths (layers 0..l replaced):
  base            unpatched
  mean{n}_ans@l   answer position <- mean over the first n seeds (n in --ns)
  mean{n}_all@l   every position after the system prompt <- mean over the first n seeds
  k1_ans@l/{s}    answer position <- seed s alone (s < --n-donors): patching with no averaging
  k1_all@l/{s}    every position <- seed s alone
Every seed (all N) also gets an unpatched batched run, for the output-level ensembles.

Cost: per (source, depth), once: the examples prefilled on the system-prompt cache with the patch (the examples
are item-independent). Per item: the N seeds' test turns as batched rows (capture), then the host rows batched.
    python seed_patch2.py --model M --repo R --seeds seeds.json --items items.json --out O --n-seeds 100
"""
import argparse, copy, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "filler_jlens"))  # pod layout: ~/fj/filler_jlens -> code
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo layout: common.py at the root
import common as C  # noqa: E402
from seed_patch import canonical_number_ids, expand_cache, split_prompt  # noqa: E402


class Hooks:
    """capture: self.sum[l] = sum over the rows of every cumulative group (rows < n for n in ns), all positions;
               rows listed in keep are stored; last=True also stores every row's last position.
    patch:   rows b with depth[b] >= l get positions [p0:] replaced by src(l)[srcidx[b]]."""

    def __init__(self, model):
        self.mode = None
        self.handles = [layer.register_forward_hook(self._hook(l)) for l, layer in enumerate(model.model.layers)]

    def _hook(self, l):
        def f(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out          # [B, S, 4, D]
            if self.mode == "capture":
                for g, (lo, hi) in self.groups.items():
                    if hi > lo:
                        s = h[lo:hi].float().sum(0).cpu()
                        self.sum.setdefault(g, {})
                        self.sum[g][l] = s
                for b, key in self.keep.items():
                    self.kept.setdefault(key, {})[l] = h[b].detach().to("cpu", torch.bfloat16)
                if self.last is not None:
                    self.last[l] = h[:, -1].detach().to("cpu", torch.bfloat16)
            elif self.mode == "patch":
                rows = (self.depth >= l).nonzero().flatten()
                if len(rows):
                    h = h.clone()
                    v = self.src(l)
                    v = v[self.srcidx[rows].to(v.device)]             # [r, S', 4, D]
                    h[rows.to(h.device), self.p0:] = v.to(h.device, h.dtype)
                    return (h, *out[1:]) if isinstance(out, tuple) else h
            return None
        return f

    def capture(self, groups=None, keep=None, last=False):
        self.mode, self.sum, self.groups, self.keep, self.kept = "capture", {}, groups or {}, keep or {}, {}
        self.last = {} if last else None

    def patch(self, depth, srcidx, src, p0=0):
        self.mode, self.depth, self.srcidx, self.src, self.p0 = "patch", torch.as_tensor(depth), torch.as_tensor(srcidx), src, p0

    def off(self):
        self.mode = None


def _layer_tensors(layer):
    for k, v in list(vars(layer).items()):
        if torch.is_tensor(v) and v.dim() > 0:
            yield k, None
        elif isinstance(v, dict):
            for n, t in v.items():
                if torch.is_tensor(t) and t.dim() > 0:
                    yield k, n


def concat_caches(caches):
    """Stack caches along the batch dim (V4 cache layers carry compressor state besides keys/values)."""
    c = copy.deepcopy(caches[0])
    for li, layer in enumerate(c.layers):
        shared = getattr(layer, "values", None) is getattr(layer, "keys", None)
        others = [x.layers[li] for x in caches]
        for k, n in list(_layer_tensors(layer)):
            t = torch.cat([(getattr(o, k) if n is None else getattr(o, k)[n]) for o in others], 0)
            if n is None:
                setattr(layer, k, t)
            else:
                getattr(layer, k)[n] = t
        if shared and getattr(layer, "keys", None) is not None:
            layer.values = layer.keys
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--repo", required=True)
    ap.add_argument("--seeds", required=True); ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--ns", default="16,50,100", help="seed-mean sizes (first n seeds)")
    ap.add_argument("--n-donors", type=int, default=16, help="single-seed patches (seeds 0..n-1)")
    ap.add_argument("--depths", default="2,6,10,14,18,26,34,42")
    ap.add_argument("--chunk", type=int, default=24, help="rows per batched forward")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    prompts, all_items, fewshot = C.load_repo(a.repo)
    sel = json.load(open(a.items))["items"]
    sel = sel[: a.limit] if a.limit else sel
    items = [all_items[i] for i in sel]
    seeds = json.load(open(a.seeds))["seeds"][: a.n_seeds]
    N, ND = len(seeds), a.n_donors
    assert N == a.n_seeds, f"only {N} seeds"
    ns = [int(x) for x in a.ns.split(",")]
    assert max(ns) <= N and ND <= N
    depths = [int(x) for x in a.depths.split(",")]
    # source index: 0..len(ns)-1 = seed means, len(ns) + s = donor s
    src_names = [f"mean{n}" for n in ns] + [f"k1/{s}" for s in range(ND)]

    tok, model = C.load_model(a.model)
    dev = model.model.embed_tokens.weight.device
    L = len(model.model.layers)
    H = Hooks(model)
    num_ids = canonical_number_ids(tok)
    eos = tok.convert_tokens_to_ids(C.EOS)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    sep = enc("\n\n")

    out = Path(a.out); (out / "dist").mkdir(parents=True, exist_ok=True)
    rows_f = open(out / "rows.jsonl", "a")
    done = {json.loads(l)["idx"] for l in open(out / "rows.jsonl")} if (out / "rows.jsonl").stat().st_size else set()
    json.dump(vars(a) | {"N": N, "L": L}, open(out / "config.json", "w"), indent=1)

    sys_t, ex_t, _ = split_prompt(C.build_prompt(prompts, fewshot, items[0], "k0", 10)[0])
    sys_ids, ex_ids = enc(sys_t), enc(ex_t)

    @torch.inference_mode()
    def fwd(cache, ids_rows):
        x = torch.tensor(ids_rows, device=dev)
        lg = model(input_ids=x, past_key_values=cache, use_cache=True).logits[:, -1].float()
        return torch.log_softmax(lg, -1)

    @torch.inference_mode()
    def prefill1(ids, cache=None):
        return model.model(input_ids=torch.tensor([ids], device=dev), past_key_values=cache, use_cache=True).past_key_values

    def rowname(si, kind, d):
        return f"{src_names[si]}_{kind}@{d}" if si < len(ns) else f"k1_{kind}@{d}/{si - len(ns)}"

    # --- Setup 1: per seed, system + seed + examples, capturing the examples' residuals. ------------------------
    t0 = time.time()
    P = {}
    ex_sum = {n: None for n in ns}
    ex_keep = {}
    for s in range(N):
        c = prefill1(sys_ids + sep + seeds[s]["ids"] + sep)
        H.capture(groups={"all": (0, 1)})
        P[s] = prefill1(ex_ids, c)
        H.off()
        cur = torch.stack([H.sum["all"][l] for l in range(L)])                    # [L, S_ex, 4, D] fp32
        for n in ns:
            if s < n:
                ex_sum[n] = cur.clone() if ex_sum[n] is None else ex_sum[n].add_(cur)
        if s < ND:
            ex_keep[s] = cur.to(torch.bfloat16)
    ex_src = [(ex_sum[n] / n).to(torch.bfloat16) for n in ns] + [ex_keep[s] for s in range(ND)]
    del ex_sum, ex_keep
    print(f"setup 1: {N} seed prefixes ({len(sys_ids) + 2 * len(sep) + len(seeds[0]['ids'])}+{len(ex_ids)} tokens) "
          f"in {time.time() - t0:.0f}s", flush=True)

    # --- Setup 2: host prefix (system + examples), unpatched and patched at all example positions per source/depth.
    t0 = time.time()
    sys_cache = prefill1(sys_ids)
    host = prefill1(ex_ids, copy.deepcopy(sys_cache))
    PP = {}
    for si, sT in enumerate(ex_src):
        for d in depths:
            H.patch([d], [0], lambda l, sT=sT: sT[l].unsqueeze(0))
            PP[(si, d)] = prefill1(ex_ids, copy.deepcopy(sys_cache))
            H.off()
    del ex_src
    print(f"setup 2: {len(PP)} patched host prefixes in {time.time() - t0:.0f}s", flush=True)

    def metrics(lp1, lp2, ans):
        la, le = lp1[:, ans].cpu(), lp2[:, eos].cpu()
        am = lp1.argmax(-1).cpu()
        recs = [{"logp_ans": float(la[b]), "logp_eos": float(le[b]), "logp_correct": float(la[b] + le[b]),
                 "argmax": tok.convert_ids_to_tokens(int(am[b]))} for b in range(len(la))]
        return recs, lp1[:, num_ids.to(lp1.device)].cpu().to(torch.float16).numpy()

    t0 = time.time()
    for j, it in enumerate(items):
        if it["idx"] in done:
            continue
        tp = [time.time()]
        text = C.build_prompt(prompts, fewshot, it, "k0", 10)[0]
        s_t, e_t, test_t = split_prompt(text)
        assert s_t == sys_t and e_t == ex_t
        test_ids = enc(test_t)
        assert enc(text) == sys_ids + ex_ids + test_ids
        ans = C.answer_ids(tok, it["answer"]); assert len(ans) == 1; ans = ans[0]
        ctx, last = test_ids[:-1], test_ids[-1]
        dist, recs = {}, []

        # Phase 1: all N seeds unpatched (batched): ensembles, plus test-turn and answer residual sums / donors.
        t_sum, a_sum = {n: 0 for n in ns}, {n: 0 for n in ns}
        t_keep, a_keep = {}, {}
        for c0 in range(0, N, a.chunk):
            ss = list(range(c0, min(N, c0 + a.chunk)))
            cb = concat_caches([P[s] for s in ss])
            keep = {b: s for b, s in enumerate(ss) if s < ND}
            H.capture(groups={n: (0, max(0, min(n, ss[-1] + 1) - c0)) for n in ns}, keep=keep)
            fwd(cb, [ctx] * len(ss))
            for n, g in H.sum.items():
                t_sum[n] = t_sum[n] + torch.stack([g[l] for l in range(L)])
            for s, kl in H.kept.items():
                t_keep[s] = torch.stack([kl[l] for l in range(L)])               # [L, S_ctx, 4, D]
            H.capture(groups={n: (0, max(0, min(n, ss[-1] + 1) - c0)) for n in ns}, keep=keep)
            lp1 = fwd(cb, [[last]] * len(ss))
            for n, g in H.sum.items():
                a_sum[n] = a_sum[n] + torch.stack([g[l] for l in range(L)])
            for s, kl in H.kept.items():
                a_keep[s] = torch.stack([kl[l] for l in range(L)])               # [L, 1, 4, D]
            H.off()
            lp2 = fwd(cb, [[ans]] * len(ss))
            ms, nd = metrics(lp1, lp2, ans)
            for b, s in enumerate(ss):
                recs.append({"host": str(s), "row": "seed", "depth": -1, **ms[b]})
                dist[f"{s}|seed"] = nd[b]
            del cb
        tsrc = torch.stack([(t_sum[n] / n).to(torch.bfloat16) for n in ns] + [t_keep[s] for s in range(ND)], 1).to(dev)
        asrc = torch.stack([(a_sum[n] / n).to(torch.bfloat16) for n in ns] + [a_keep[s] for s in range(ND)], 1).to(dev)
        tp.append(time.time())                                                    # [L, n_src, S, 4, D]

        # Phase 2a: answer-position rows on the host's own context (one batched single-token decode, like round 1).
        hc = prefill1(ctx, copy.deepcopy(host))
        specs = [("base", -1, 0)] + [(rowname(si, "ans", d), d, si) for d in depths for si in range(len(src_names))]
        for c0 in range(0, len(specs), 4 * a.chunk):
            sp = specs[c0: c0 + 4 * a.chunk]
            cb = expand_cache(hc, len(sp))
            H.patch([x[1] for x in sp], [x[2] for x in sp], lambda l: asrc[l])
            lp1 = fwd(cb, [[last]] * len(sp))
            H.off()
            lp2 = fwd(cb, [[ans]] * len(sp))
            ms, nd = metrics(lp1, lp2, ans)
            for b, (name, d, _) in enumerate(sp):
                recs.append({"host": "none", "row": name, "depth": d, **ms[b]})
                dist[f"none|{name}"] = nd[b]
            del cb
        del hc
        tp.append(time.time())

        # Phase 2b: all-position rows: patched host prefix + test turn patched + answer position patched.
        specs = [(rowname(si, "all", d), d, si) for d in depths for si in range(len(src_names))]
        for c0 in range(0, len(specs), a.chunk):
            sp = specs[c0: c0 + a.chunk]
            cb = concat_caches([PP[(si, d)] for _, d, si in sp])
            dep, sidx = [x[1] for x in sp], [x[2] for x in sp]
            H.patch(dep, sidx, lambda l: tsrc[l]); fwd(cb, [ctx] * len(sp))
            H.patch(dep, sidx, lambda l: asrc[l]); lp1 = fwd(cb, [[last]] * len(sp))
            H.off()
            lp2 = fwd(cb, [[ans]] * len(sp))
            ms, nd = metrics(lp1, lp2, ans)
            for b, (name, d, _) in enumerate(sp):
                recs.append({"host": "none", "row": name, "depth": d, **ms[b]})
                dist[f"none|{name}"] = nd[b]
            del cb
        del tsrc, asrc
        tp.append(time.time())

        # Reference: 100 dots, full prefill.
        ids_d, _ = C.tokenize_prompt(tok, C.build_prompt(prompts, fewshot, it, "dots100", 10)[0])
        cd = prefill1(ids_d[:-1])
        lp1 = fwd(cd, [[ids_d[-1]]]); lp2 = fwd(cd, [[ans]])
        ms, nd = metrics(lp1, lp2, ans)
        recs.append({"host": "dots100", "row": "base", "depth": -1, **ms[0]})
        dist["dots100|base"] = nd[0]
        tp.append(time.time())

        for r in recs:
            rows_f.write(json.dumps({"idx": it["idx"], "answer": it["answer"], **r}) + "\n")
        rows_f.flush()
        np.savez_compressed(out / "dist" / f"{it['idx']:04d}.npz", **dist)
        el = time.time() - t0
        ph = " ".join(f"{n}={b - a_:.1f}s" for n, a_, b in zip(["seeds", "ans", "all", "dots"], tp, tp[1:]))
        print(f"item {it['idx']} ({j + 1}/{len(items)}): {el / (j + 1):.1f}s/item [{ph}] rows={len(recs)} "
              f"test_len={len(test_ids)}", flush=True)


if __name__ == "__main__":
    main()
