import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: common.py
import time, torch, common as C
prompts, items, fewshot = C.load_repo("/data/repo")
tok, model = C.load_model("/data/v4flash")
dev = model.model.embed_tokens.weight.device
ids, _ = C.tokenize_prompt(tok, C.build_prompt(prompts, fewshot, items[0], "dots100", 10)[0])
x = torch.tensor([ids], device=dev)
def t(f, n=3):
    f(); torch.cuda.synchronize(); s = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - s) / n
with torch.inference_mode():
    print("prefill 2k", t(lambda: model(input_ids=x, use_cache=False)))
    print("prefill 2k x8 batch", t(lambda: model(input_ids=x.repeat(8, 1), use_cache=False), 1))
    print("greedy gen", t(lambda: model.generate(x, max_new_tokens=10, do_sample=False)))
    print("sample gen x8", t(lambda: model.generate(x, max_new_tokens=10, do_sample=True, top_k=0, num_return_sequences=8), 1))
    print("sample gen x16", t(lambda: model.generate(x, max_new_tokens=10, do_sample=True, top_k=0, num_return_sequences=16), 1))
