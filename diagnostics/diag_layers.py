"""First layer / sublayer where prefix positions diverge between a prompt and the prompt minus 5 tokens."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: common.py
import torch, common as C
prompts, items, fewshot = C.load_repo("/data/repo")
tok, model = C.load_model("/data/v4flash")
dev = model.model.embed_tokens.weight.device
caps = {}
def hook(name):
    def f(m, i, o):
        o = o[0] if isinstance(o, tuple) else o
        caps.setdefault(name, []).append(o.detach().float().cpu())
    return f
hs = []
for l, layer in enumerate(model.model.layers[:6]):
    hs.append(layer.register_forward_hook(hook(f"L{l}")))
    for n, m in layer.named_children():
        hs.append(m.register_forward_hook(hook(f"L{l}.{n}")))
print([n for n, _ in model.model.layers[0].named_children()])
print("mlp children:", [n for n, _ in model.model.layers[0].mlp.named_children()][:10])
ids, _ = C.tokenize_prompt(tok, C.build_prompt(prompts, fewshot, items[0], "k0", 0)[0])
with torch.inference_mode():
    model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
    model(input_ids=torch.tensor([ids[:-5]], device=dev), use_cache=False)
P = 60  # compare the first 60 positions only: far from the cut, inside sliding window history
for name, (a, b) in caps.items():
    if a.dim() < 2: continue
    a, b = a[0, :P], b[0, :P]
    d = (a - b).abs().max().item(); s = a.abs().max().item()
    print(f"{name:28s} shape {tuple(a.shape)} max|diff| first {P} pos {d:.3g} (scale {s:.3g})")
