"""Black-box rollouts like the example repo: graded with the repo's strict parse_answer, max 10 new
tokens. Per item x condition: one greedy rollout (phase2) and --samples rollouts at T=1 (phase3 style).

Fast path: prefill the prompt once, expand the KV cache to 1 + samples rows, and decode all rows
together (row 0 greedy, the rest sampled at T=1, full vocab). model.generate(num_return_sequences=n)
would prefill the prompt n times. Resumable.

    python rollout.py --model M --repo R --out O --shots 10 --conds k0,dots10,dots100 [--samples 8] [--items 0-599]
"""
import argparse, copy, json, time
from pathlib import Path
import torch
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: common.py
import common as C

C.CONDITIONS.setdefault("dots10", ("dots", 10))
C.CONDITIONS.setdefault("dots25", ("dots", 25))


def expand(obj, n, seen=None):
    """Repeat every batch-1 tensor inside a cache object to batch n, in place."""
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return obj
    seen.add(id(obj))
    if torch.is_tensor(obj):
        return obj.repeat_interleave(n, 0) if obj.dim() > 0 and obj.shape[0] == 1 else obj
    if isinstance(obj, dict):
        for k in obj:
            obj[k] = expand(obj[k], n, seen)
        return obj
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = expand(obj[i], n, seen)
        return obj
    if isinstance(obj, tuple):
        return tuple(expand(v, n, seen) for v in obj)
    if hasattr(obj, "__dict__") and not isinstance(obj, (torch.nn.Module, type)):
        for k, v in vars(obj).items():
            setattr(obj, k, expand(v, n, seen))
    return obj


def rollouts(model, tok, ids, n_samples, max_new, gen):
    dev = model.model.embed_tokens.weight.device
    eos = tok.convert_tokens_to_ids(C.EOS)
    out = model(input_ids=torch.tensor([ids], device=dev), use_cache=True)
    cache, logits = out.past_key_values, out.logits[:, -1].float()
    n = 1 + n_samples
    cache = expand(cache, n)
    logits = logits.repeat(n, 1)
    seqs = torch.zeros((n, 0), dtype=torch.long, device=dev)
    finished = torch.zeros(n, dtype=torch.bool, device=dev)
    for step in range(max_new):
        probs = torch.softmax(logits, -1)
        nxt = torch.cat([logits[:1].argmax(-1), torch.multinomial(probs[1:], 1, generator=gen).squeeze(-1)]) if n > 1 \
            else logits.argmax(-1)
        nxt = torch.where(finished, torch.full_like(nxt, eos), nxt)
        seqs = torch.cat([seqs, nxt[:, None]], 1)
        finished |= nxt == eos
        if finished.all() or step == max_new - 1:
            break
        pos = torch.full((n, 1), len(ids) + step, device=dev)
        out = model(input_ids=nxt[:, None], past_key_values=cache, position_ids=pos, use_cache=True)
        cache, logits = out.past_key_values, out.logits[:, -1].float()
    texts = [tok.decode(s, skip_special_tokens=True) for s in seqs.tolist()]
    return texts[0], texts[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--repo", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=8); ap.add_argument("--shots", type=int, default=10)
    ap.add_argument("--conds", default="k0,dots10,dots100"); ap.add_argument("--items", default="0-599")
    ap.add_argument("--check-generate", type=int, default=0, help="also run model.generate greedy on the first N items and compare")
    a = ap.parse_args()
    conds = a.conds.split(",")
    prompts, items, fewshot = C.load_repo(a.repo)
    lo, hi = map(int, a.items.split("-"))
    items = [it for it in items if lo <= it["idx"] <= hi]
    out = Path(a.out) / f"rollout_{a.shots}shot.jsonl"
    done = {(r["idx"], r["cond"]) for r in map(json.loads, open(out))} if out.exists() else set()
    tok, model = C.load_model(a.model)
    dev = model.model.embed_tokens.weight.device
    gen = torch.Generator(device=dev)
    t0, n = time.time(), 0
    with open(out, "a") as f, torch.inference_mode():
        for j, it in enumerate(items):
            for cond in conds:
                if (it["idx"], cond) in done:
                    continue
                gen.manual_seed(1000 * it["idx"] + conds.index(cond))
                ids, _ = C.tokenize_prompt(tok, C.build_prompt(prompts, fewshot, it, cond, a.shots)[0])
                greedy, samples = rollouts(model, tok, ids, a.samples, 10, gen)
                grade = lambda s: prompts.parse_answer(s) == it["answer"]
                row = {"idx": it["idx"], "cond": cond, "shots": a.shots, "answer": it["answer"], "prompt_tokens": len(ids),
                       "greedy": greedy, "greedy_correct": grade(greedy), "samples": samples,
                       "n_correct": sum(map(grade, samples)), "n_parsed": sum(prompts.parse_answer(s) is not None for s in samples)}
                if j < a.check_generate:
                    g = model.generate(torch.tensor([ids], device=dev), max_new_tokens=10, do_sample=False)
                    row["generate_greedy"] = tok.decode(g[0, len(ids):], skip_special_tokens=True)
                f.write(json.dumps(row) + "\n"); f.flush(); n += 1
            if n and (j % 10 == 0 or j == len(items) - 1):
                print(f"item {it['idx']}: {n} item-conds, {(time.time()-t0)/n:.2f}s each", flush=True)


if __name__ == "__main__":
    main()
