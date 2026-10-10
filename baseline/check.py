"""Step 0 checks before the main run. Prints a report and writes OUT/lens_config.json.

    python check.py --model MODEL_DIR --repo REPO_DIR --lens LENS.pt --out OUT

1. Tokenizer: every 0-999 is one token; answer / intermediate token counts on eval.jsonl.
2. Chat template: our render() vs the tokenizer's own template (thinking off).
3. Prompt lengths per shots x condition; 100 dots add exactly 100 tokens.
4. Causality: appending the answer token leaves earlier logits unchanged.
5. Greedy answers on a few items, zero-shot and 10-shot.
6. Lens: spider check for every readout variant; picks one and writes lens_config.json.
7. Lens: the '48' in 'zif = 48' should be readable at late layers.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: common.py
import common as C

SPIDER = "The number of legs on the animal that spins webs is"


def rank_of(lp, tid):
    return int((lp > lp[tid]).sum().item()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--lens", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-items", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prompts, items, fewshot = C.load_repo(args.repo)
    tok, model = C.load_model(args.model)
    dev = model.model.embed_tokens.weight.device
    fwd = lambda ids: model(input_ids=torch.tensor([ids], device=dev), use_cache=False).logits[0].float()

    print("== 1. tokenizer")
    bad = [n for n in range(1000) if len(C.answer_ids(tok, n)) != 1]
    num_ids, _ = C.number_token_ids(tok)
    print(f"0-999 not single tokens: {bad[:10]} ({len(bad)}); all-digit vocab tokens: {len(num_ids)}")
    for key, get in [("answer", lambda x: x["answer"]), ("y", lambda x: x["queried_value"]),
                     ("c1x", lambda x: x["chain"]["c1x"]), ("c2y", lambda x: x["chain"]["c2y"])]:
        print(f"  {key}: tokens per value {dict(Counter(len(C.answer_ids(tok, get(x))) for x in items))}")
    print(f"  ' 264' -> {tok.tokenize(' 264')}  (digits split from the space)")

    print("== 2. chat template")
    text, messages = C.build_prompt(prompts, fewshot, items[0], "k0", 0)
    theirs = None
    for kw in ({"thinking": False}, {"enable_thinking": False}, {}):
        try:
            theirs = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
            print(f"  apply_chat_template({kw}) worked")
            break
        except Exception as e:  # noqa: BLE001
            print(f"  apply_chat_template({kw}) failed: {str(e)[:120]}")
    if theirs is not None:
        same = theirs == text
        print(f"  identical to render(): {same}")
        if not same:
            print(f"  ours  tail: {text[-120:]!r}\n  theirs tail: {theirs[-120:]!r}\n  ours head: {text[:80]!r}\n  theirs head: {theirs[:80]!r}")

    print("== 3. prompt lengths")
    for shots in (0, 10):
        lens_by_cond = {}
        for cond in C.CONDITIONS:
            lens_by_cond[cond] = np.array([len(C.tokenize_prompt(tok, C.build_prompt(prompts, fewshot, it, cond, shots)[0])[0])
                                           for it in items[:50]])
            v = lens_by_cond[cond]
            print(f"  {shots}-shot {cond}: mean {v.mean():.1f} min {v.min()} max {v.max()}")
        per_turn = (lens_by_cond["dots100"] - lens_by_cond["k0"]) / (shots + 1)
        print(f"  dots100 adds {set(per_turn.tolist())} tokens per turn (expect 100)")
    ids, seg = C.tokenize_prompt(tok, text)
    toks = tok.convert_ids_to_tokens(ids)
    print("  segments of the test turn:", [(t, int(s)) for t, s in zip(toks, seg) if s > 0][:60])

    print("== 4. causality with the answer token appended")
    for it in items[:3]:
        text, _ = C.build_prompt(prompts, fewshot, it, "dots100", 0)
        ids, _ = C.tokenize_prompt(tok, text)
        with torch.inference_mode():
            a = fwd(ids)
            b = fwd(ids + C.answer_ids(tok, it["answer"]))
        print(f"  item {it['idx']}: max |logit diff| over prompt positions {(a - b[:len(ids)]).abs().max().item():.3g}")

    print("== 5. greedy answers")
    for shots in (0, 10):
        for cond in C.CONDITIONS:
            got = []
            for it in items[:args.n_items]:
                text, _ = C.build_prompt(prompts, fewshot, it, cond, shots)
                ids, _ = C.tokenize_prompt(tok, text)
                with torch.inference_mode():
                    g = model.generate(torch.tensor([ids], device=dev), max_new_tokens=8, do_sample=False)
                s = tok.decode(g[0, len(ids):], skip_special_tokens=True)
                got.append((s, it["answer"]))
            acc = np.mean([s.strip() == str(a) for s, a in got])
            print(f"  {shots}-shot {cond}: exact {acc:.2f}; first 5 {got[:5]}")

    if not args.lens:
        return
    print("== 6. lens: spider check")
    cap = C.ResidCapture(model)
    ids = tok.encode(C.BOS + SPIDER, add_special_tokens=False)
    with torch.inference_mode():
        final = torch.log_softmax(fwd(ids)[-1], -1)
    spider, eight = tok.convert_tokens_to_ids("Ġspider"), tok.convert_tokens_to_ids("Ġeight")
    print(f"  model's own next token: {tok.convert_ids_to_tokens(final.argmax().item())!r}; "
          f"rank of ' spider' {rank_of(final, spider)}, ' eight' {rank_of(final, eight)}")
    results = {}
    lens = C.JLens(args.lens, model)
    with torch.inference_mode():  # the readout path itself must reproduce the model's logits
        last = max(cap.out)
        re = lens.final_logits(model.model.hc_head(cap.out[last].to(model.model.hc_head.hc_fn.device))[0])
        print(f"  readout path check: max |final_logits(hc_head(resid[{last}])) - model logits| "
              f"{(re[-1].to(final.device) - fwd(ids)[-1]).abs().max().item():.3g} (expect ~0)")
    print(f"  lens meta: {lens.meta}; layers {lens.layers[0]}..{lens.layers[-1]} ({len(lens.layers)})")
    for how in C.COLLAPSES:
        for tr in (False, True):
            lens.collapse_how, lens.transpose = how, tr
            ranks, kl_last = [], None
            with torch.inference_mode():
                for l in lens.layers:
                    lp = lens.layer_logprobs(cap.out[l][0], l)
                    ranks.append(rank_of(lp[-1], spider))
                kl_last = float((final.exp() * (final - lp[-1].to(final.device))).sum())
            results[(how, tr)] = (ranks, kl_last)
            print(f"  collapse={how:8s} transpose={tr!s:5s} best rank of ' spider' {min(ranks):6d} "
                  f"at layer {lens.layers[int(np.argmin(ranks))]}; layers in top 10: {sum(r <= 10 for r in ranks)}; "
                  f"KL(final || lens@{lens.layers[-1]}) {kl_last:.3f}")
    best = min(results, key=lambda k: (-sum(r <= 10 for r in results[k][0]), min(results[k][0]), results[k][1]))
    ranks = results[best][0]
    print(f"  chosen: collapse={best[0]} transpose={best[1]}; ' spider' rank by layer: {dict(zip(lens.layers, ranks))}")
    if min(ranks) > 10:
        print("  WARNING: ' spider' never reaches the top 10 at any layer. The hook point or readout is "
              "probably wrong; do not run the lens dump until this is fixed.")
    (out / "lens_config.json").write_text(json.dumps({"collapse": best[0], "transpose": best[1],
                                                      "spider_ranks": dict(zip(map(str, lens.layers), ranks))}))
    lens.collapse_how, lens.transpose = best

    print("== 7. lens: reading a defined value")
    it = items[0]
    text, _ = C.build_prompt(prompts, fewshot, it, "k0", 0)
    ids, seg = C.tokenize_prompt(tok, text)
    x_name, x_val = next((n, v) for n, v in it["definitions"] if isinstance(v, int))
    tid = C.answer_ids(tok, x_val)[0]
    pos = [p for p, t in enumerate(ids) if t == tid and seg[p] == C.SEG_DEFS][0]
    cap.clear()
    with torch.inference_mode():
        fwd(ids)
        for l in lens.layers[::4] + [lens.layers[-1]]:
            lp = lens.layer_logprobs(cap.out[l][0], l)
            top = tok.convert_ids_to_tokens(lp[pos].topk(5).indices.tolist())
            print(f"  layer {l:2d} at '{x_name} = {x_val}' (pos {pos}): rank of {x_val} {rank_of(lp[pos], tid)}; top5 {top}")
    cap.remove()


if __name__ == "__main__":
    main()
