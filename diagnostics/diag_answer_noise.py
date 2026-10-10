"""Length-dependent noise where it matters: log P(answer) at the answer position from the prompt-only
pass vs the prompt+answer pass (same position, different sequence length)."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: common.py
import torch, numpy as np, common as C
prompts, items, fewshot = C.load_repo("/data/repo")
tok, model = C.load_model("/data/v4flash")
dev = model.model.embed_tokens.weight.device
fwd = lambda ids: torch.log_softmax(model(input_ids=torch.tensor([ids], device=dev), use_cache=False).logits[0].float(), -1)
num_ids, _ = C.number_token_ids(tok)
num_ids = torch.tensor(num_ids, device="cpu")
with torch.inference_mode():
    for cond in ("k0", "dots100"):
        rows = []
        for it in items[:30]:
            ids, _ = C.tokenize_prompt(tok, C.build_prompt(prompts, fewshot, it, cond, 0)[0])
            a = C.answer_ids(tok, it["answer"])
            p1 = fwd(ids)[-1].cpu(); p2 = fwd(ids + a)[len(ids) - 1].cpu()
            kl = float((p1.exp() * (p1 - p2)).sum())
            rows.append((float(p1[a[0]]), float(p2[a[0]]), kl, float(p1[num_ids].exp().sum()), float(p2[num_ids].exp().sum())))
        r = np.array(rows)
        d = r[:, 1] - r[:, 0]
        print(f"{cond}: mean P(ans tok) {np.exp(r[:,0]).mean():.3f} vs {np.exp(r[:,1]).mean():.3f}; |dlogP| median {np.median(abs(d)):.3f} max {abs(d).max():.3f}; "
              f"|dP| max {np.abs(np.exp(r[:,1])-np.exp(r[:,0])).max():.3f}; KL median {np.median(r[:,2]):.4f} max {r[:,2].max():.3f}; number mass {r[:,3].mean():.3f}")
