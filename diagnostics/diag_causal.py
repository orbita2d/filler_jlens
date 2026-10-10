"""Where do prompt-position logits change when a token is appended? Also run-to-run noise."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: common.py
import torch, numpy as np, common as C
prompts, items, fewshot = C.load_repo("/data/repo")
tok, model = C.load_model("/data/v4flash")
print("attn impl", model.config._attn_implementation)
dev = model.model.embed_tokens.weight.device
fwd = lambda ids: model(input_ids=torch.tensor([ids], device=dev), use_cache=False).logits[0].float()
def report(name, a, b):
    d = (a - b).abs().max(-1).values.cpu().numpy()
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    kl = (la.exp() * (la - lb)).sum(-1).cpu().numpy()
    worst = np.argsort(-d)[:8]
    print(f"  {name}: max|dlogit| {d.max():.3g} median {np.median(d):.3g}; argmax agree {agree:.3f}; KL max {kl.max():.3g} mean {kl.mean():.3g}; worst positions {worst.tolist()} (len {len(d)})")
    print(f"    max|dlogit| by position decile: {[round(float(x),2) for x in [d[i*len(d)//10:(i+1)*len(d)//10].max() for i in range(10)]]}")
with torch.inference_mode():
    for it in items[:2]:
        for cond in ("k0", "dots100"):
            ids, _ = C.tokenize_prompt(tok, C.build_prompt(prompts, fewshot, it, cond, 0)[0])
            a = fwd(ids); a2 = fwd(ids)
            print(f"item {it['idx']} {cond} len {len(ids)}")
            report("same input twice", a, a2)
            b = fwd(ids + C.answer_ids(tok, it["answer"]))[:len(ids)]
            report("append answer", a, b)
            c = fwd(ids[:-5])
            report("drop last 5 (compare prefix)", a[:len(ids)-5], c)
