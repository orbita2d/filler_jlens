"""Seed ensembles over every item: N seeded runs (no patching) plus the no-seed and 100-dot references.

Same prompts and numerics path as seed_patch2.py phase 1: per seed, system + seed + examples prefilled once
(item-independent); per item, all seeds' test turns as batched rows on those caches, then the answer step and the
teacher-forced EOS step. References (no seed, 100 dots) are full B=1 prefills. Single-token answers only.
    python seed_ens.py --model M --repo R --seeds seeds100.json --out O [--items 0-599]
"""
import argparse, copy, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "filler_jlens"))  # pod layout: ~/fj/filler_jlens -> code
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo layout: common.py at the root
import common as C  # noqa: E402
from seed_patch import canonical_number_ids, split_prompt  # noqa: E402  (also registers dots100)
from seed_patch2 import concat_caches  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--repo", required=True)
    ap.add_argument("--seeds", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--items", default=None, help="items.json or lo-hi; default all")
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    prompts, all_items, fewshot = C.load_repo(a.repo)
    if a.items is None:
        sel = list(range(len(all_items)))
    elif a.items.endswith(".json"):
        sel = json.load(open(a.items))["items"]
    else:
        lo, hi = map(int, a.items.split("-")); sel = list(range(lo, hi + 1))
    sel = sel[: a.limit] if a.limit else sel
    seeds = json.load(open(a.seeds))["seeds"][: a.n_seeds]
    N = len(seeds); assert N == a.n_seeds

    tok, model = C.load_model(a.model)
    dev = model.model.embed_tokens.weight.device
    num_ids = canonical_number_ids(tok)
    eos = tok.convert_tokens_to_ids(C.EOS)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    sep = enc("\n\n")

    out = Path(a.out); (out / "dist").mkdir(parents=True, exist_ok=True)
    rows_f = open(out / "rows.jsonl", "a")
    done = {json.loads(l)["idx"] for l in open(out / "rows.jsonl")} if (out / "rows.jsonl").stat().st_size else set()
    json.dump(vars(a) | {"N": N}, open(out / "config.json", "w"), indent=1)

    sys_t, ex_t, _ = split_prompt(C.build_prompt(prompts, fewshot, all_items[sel[0]], "k0", 10)[0])
    sys_ids, ex_ids = enc(sys_t), enc(ex_t)

    @torch.inference_mode()
    def fwd(cache, ids_rows):
        lg = model(input_ids=torch.tensor(ids_rows, device=dev), past_key_values=cache, use_cache=True).logits[:, -1].float()
        return torch.log_softmax(lg, -1)

    @torch.inference_mode()
    def prefill1(ids, cache=None):
        return model.model(input_ids=torch.tensor([ids], device=dev), past_key_values=cache, use_cache=True).past_key_values

    t0 = time.time()
    P = [prefill1(ex_ids, prefill1(sys_ids + sep + s["ids"] + sep)) for s in seeds]
    print(f"setup: {N} seed prefixes in {time.time() - t0:.0f}s", flush=True)

    def metrics(lp1, lp2, ans):
        la, le = lp1[:, ans].cpu(), lp2[:, eos].cpu()
        am = lp1.argmax(-1).cpu()
        return [{"logp_ans": float(la[b]), "logp_eos": float(le[b]), "logp_correct": float(la[b] + le[b]),
                 "argmax": tok.convert_ids_to_tokens(int(am[b]))} for b in range(len(la))], \
            lp1[:, num_ids.to(lp1.device)].cpu().to(torch.float16).numpy()

    t0 = time.time(); k = 0
    for i in sel:
        it = all_items[i]
        if it["idx"] in done:
            continue
        ans = C.answer_ids(tok, it["answer"])
        if len(ans) != 1:
            print(f"skip {i}: multi-token answer {it['answer']}", flush=True); continue
        ans = ans[0]
        text = C.build_prompt(prompts, fewshot, it, "k0", 10)[0]
        s_t, e_t, test_t = split_prompt(text)
        assert s_t == sys_t and e_t == ex_t
        test_ids = enc(test_t)
        ctx, last = test_ids[:-1], test_ids[-1]
        recs, dist = [], {}
        for c0 in range(0, N, a.chunk):
            ss = list(range(c0, min(N, c0 + a.chunk)))
            cb = concat_caches([P[s] for s in ss])
            fwd(cb, [ctx] * len(ss)); lp1 = fwd(cb, [[last]] * len(ss)); lp2 = fwd(cb, [[ans]] * len(ss))
            ms, nd = metrics(lp1, lp2, ans)
            for b, s in enumerate(ss):
                recs.append({"host": str(s), "row": "seed", **ms[b]}); dist[f"{s}|seed"] = nd[b]
            del cb
        for cond, host in (("k0", "none"), ("dots100", "dots100")):
            ids_d, _ = C.tokenize_prompt(tok, C.build_prompt(prompts, fewshot, it, cond, 10)[0])
            cd = prefill1(ids_d[:-1])
            lp1 = fwd(cd, [[ids_d[-1]]]); lp2 = fwd(cd, [[ans]])
            ms, nd = metrics(lp1, lp2, ans)
            recs.append({"host": host, "row": "base", **ms[0]}); dist[f"{host}|base"] = nd[0]
        for r in recs:
            rows_f.write(json.dumps({"idx": it["idx"], "answer": it["answer"], **r}) + "\n")
        rows_f.flush()
        np.savez_compressed(out / "dist" / f"{it['idx']:04d}.npz", **dist)
        k += 1
        if k % 10 == 0 or k == 1:
            print(f"item {it['idx']} ({k} done): {(time.time() - t0) / k:.1f}s/item", flush=True)
    print("ALL_ITEMS_DONE", flush=True)


if __name__ == "__main__":
    main()
