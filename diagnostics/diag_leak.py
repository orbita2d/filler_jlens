"""Same-length test: prompt+[tok A] vs prompt+[tok B]. Kernels see identical shapes, so any difference
at prompt positions can only be information flowing backwards from the appended token."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: common.py
import torch, common as C
prompts, items, fewshot = C.load_repo("/data/repo")
tok, model = C.load_model("/data/v4flash")
dev = model.model.embed_tokens.weight.device
fwd = lambda ids: torch.log_softmax(model(input_ids=torch.tensor([ids], device=dev), use_cache=False).logits[0].float(), -1)
with torch.inference_mode():
    for it in items[:4]:
        for cond in ("k0", "dots100"):
            ids, _ = C.tokenize_prompt(tok, C.build_prompt(prompts, fewshot, it, cond, 0)[0])
            L = len(ids)
            a = fwd(ids + C.answer_ids(tok, it["answer"])[:1])[:L]
            b = fwd(ids + C.answer_ids(tok, 7))[:L]
            c = fwd(ids + tok.encode(" the", add_special_tokens=False)[:1])[:L]
            kl = lambda p, q: (p.exp() * (p - q)).sum(-1)
            for name, q in (("ans vs 7", b), ("ans vs ' the'", c)):
                k = kl(a, q)
                nz = (k.abs() > 0).nonzero().flatten().tolist()
                print(f"item {it['idx']} {cond} L={L} L%4={L%4} {name}: KL at answer pos {k[-1].item():.4g}, max {k.max().item():.4g}, "
                      f"positions differing {len(nz)} first {nz[:5]} last {nz[-5:]}")
