"""One teacher-forced forward pass per (item, condition): P(correct) at T=1 from the logits, and
optionally the Jacobian lens at every layer and prompt position.

    python run.py --model MODEL_DIR --repo REPO_DIR --out OUT --shots 0 --lens LENS.pt
    python run.py --model MODEL_DIR --repo REPO_DIR --out OUT --shots 10          # logits only

Writes OUT/{shots}shot/results.jsonl (one line per item x condition) and, with --lens,
OUT/{shots}shot/{lens_tag}/{cond}/{idx:04d}.npz. Re-running skips finished items.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: common.py
import common as C


def parse_items(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def final_metrics(logits, P, ans, eos_id, num_ids, tok):
    """logits: [P + len(ans), V] for input prompt + answer tokens. Position P-1 predicts the
    first answer token; position P-1+len(ans) predicts end-of-sequence."""
    lp = torch.log_softmax(logits[P - 1:].float(), -1)  # rows: answer pos, then teacher-forced
    targets = list(ans) + [eos_id]
    tok_lp = [lp[j, t].item() for j, t in enumerate(targets)]
    greedy = [lp[j].argmax().item() for j in range(len(targets))]
    first = lp[0]
    p_first = first.exp()
    top_v, top_i = first.topk(20)
    return {
        "logp_answer_tokens": sum(tok_lp[:-1]),
        "logp_eos_after_answer": tok_lp[-1],
        "logp_correct": sum(tok_lp),  # log P(correct) at T=1, canonical tokenization
        "greedy_correct": greedy == targets,
        "argmax_first": tok.convert_ids_to_tokens(greedy[0]),
        "entropy_first": float(-(p_first * first).sum()),
        "number_mass_first": float(p_first[torch.as_tensor(num_ids, device=first.device)].sum()),
        "top20_first": [[tok.convert_ids_to_tokens(i), round(v, 4)] for v, i in zip(top_v.tolist(), top_i.tolist())],
    }, first[torch.as_tensor(num_ids, device=first.device)].cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local DeepSeek-V4-Flash checkpoint dir")
    ap.add_argument("--repo", required=True, help="clone of orvelte/fillertokensminiproject")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--conds", default="k0,dots100")
    ap.add_argument("--items", default="", help="e.g. 0-99 or 0,5,7; default all 600")
    ap.add_argument("--lens", default=None, help="jlens .pt; omit for logits only")
    ap.add_argument("--lens-tag", default="jlens")
    ap.add_argument("--lens-config", default=None, help="lens_config.json from check.py")
    ap.add_argument("--topk", type=int, default=128, help="top-k non-number lens tokens to keep")
    ap.add_argument("--save-resid", action="store_true",
                    help="also save the residual streams: all layers at the answer position, and "
                         "--resid-layers at every prompt position")
    ap.add_argument("--resid-layers", default="10,20,30,41")
    args = ap.parse_args()

    prompts, items, fewshot = C.load_repo(args.repo)
    tok, model = C.load_model(args.model)
    num_ids, num_texts = C.number_token_ids(tok)
    eos_id = tok.convert_tokens_to_ids(C.EOS)
    cap = C.ResidCapture(model)

    lens = None
    if args.lens:
        cfg = json.load(open(args.lens_config)) if args.lens_config else {"collapse": "mean", "transpose": False}
        lens = C.JLens(args.lens, model, collapse_how=cfg["collapse"], transpose=cfg["transpose"])
        num_mask = torch.zeros(model.lm_head.weight.shape[0], dtype=torch.bool, device=lens.device)
        num_mask[torch.as_tensor(num_ids, device=lens.device)] = True
        print(f"lens {args.lens}: layers {lens.layers[0]}..{lens.layers[-1]}, meta {lens.meta}, cfg {cfg}")
    resid_layers = [int(x) for x in args.resid_layers.split(",")]

    root = Path(args.out) / f"{args.shots}shot"
    root.mkdir(parents=True, exist_ok=True)
    (root / "number_tokens.json").write_text(json.dumps({"ids": num_ids.tolist(), "texts": num_texts}))
    res_path = root / "results.jsonl"
    done = set()
    if res_path.exists():
        for line in open(res_path):
            r = json.loads(line)
            done.add((r["idx"], r["cond"], r.get("lens_tag")))
    res_f = open(res_path, "a")

    todo = parse_items(args.items, len(items))
    conds = args.conds.split(",")
    t0 = time.time()
    for n, i in enumerate(todo):
        item = items[i]
        for cond in conds:
            tag = args.lens_tag if lens else None
            if (i, cond, tag) in done:
                continue
            text, _ = C.build_prompt(prompts, fewshot, item, cond, args.shots)
            ids, seg = C.tokenize_prompt(tok, text)
            ans = C.answer_ids(tok, item["answer"])
            P = len(ids)
            inp = torch.tensor([ids + ans], device=model.model.embed_tokens.weight.device)
            cap.clear()
            with torch.inference_mode():
                logits = model(input_ids=inp, use_cache=False).logits[0]
                metrics, num_lp_final = final_metrics(logits, P, ans, eos_id, num_ids, tok)
                rec = {"idx": i, "cond": cond, "shots": args.shots, "lens_tag": tag,
                       "answer": item["answer"], "y": item["queried_value"], "n_prompt_tokens": P,
                       "n_answer_tokens": len(ans), **metrics}

                if lens is not None:
                    d = root / args.lens_tag / cond
                    d.mkdir(parents=True, exist_ok=True)
                    L, S = len(lens.layers), P
                    top_i = np.zeros((L, S, args.topk), np.int32)
                    top_v = np.zeros((L, S, args.topk), np.float16)
                    num_lp = np.zeros((L, S, len(num_ids)), np.float16)
                    nid = torch.as_tensor(num_ids, device=lens.device)
                    for li, l in enumerate(lens.layers):
                        lp = lens.layer_logprobs(cap.out[l][0, :P], l)  # [S, V]
                        ti, tv = C.top_non_number(lp, num_mask, args.topk)
                        top_i[li], top_v[li] = ti.cpu().numpy(), tv.cpu().numpy()
                        num_lp[li] = lp[:, nid].to(torch.float16).cpu().numpy()
                        del lp
                    np.savez(d / f"{i:04d}.npz", input_ids=np.array(ids, np.int32), seg=seg,
                             layers=np.array(lens.layers, np.int16), top_ids=top_i, top_logprobs=top_v,
                             number_logprobs=num_lp, final_number_logprobs=num_lp_final)

                if args.save_resid:
                    d = root / "resid" / cond
                    d.mkdir(parents=True, exist_ok=True)
                    torch.save({"answer_pos": torch.stack([cap.out[l][0, P - 1].cpu() for l in sorted(cap.out)]),
                                "layers": resid_layers,
                                "all_pos": torch.stack([cap.out[l][0, :P].cpu() for l in resid_layers]),
                                "seg": torch.from_numpy(seg)}, d / f"{i:04d}.pt")

            res_f.write(json.dumps(rec) + "\n")
            res_f.flush()
        if n % 20 == 0:
            el = time.time() - t0
            print(f"{n + 1}/{len(todo)} items, {el / 60:.1f} min elapsed", flush=True)
    cap.remove()


if __name__ == "__main__":
    main()
