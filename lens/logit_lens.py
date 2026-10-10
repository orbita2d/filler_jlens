"""Logit lens on the full-precision 50-item subset, from the saved residuals (no model needed):
logits_l = lm_head(norm(collapse(h_l))) for every layer 0..42 and every test-turn position.

Two collapses of V4's 4 residual streams: "hchead" (the model's own final HyperHead mix, applied at every
layer; at layer 42 this is exactly the model's output) and "mean". Output mirrors the subset J-lens npz
(float32 number_logprobs over the 1110 number columns, top-128 non-number tokens):
    results/subset/10shot/logitlens_{collapse}/{cond}/{idx:04d}.npz
Column n of number_logprobs is NOT the number n: map through number_tokens.json texts.index(str(n)).
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "results/subset/10shot"
CONDS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["k0", "dots10"]
dev = "cuda"
W = {}
with safe_open(ROOT / "head/shard45.safetensors", "pt") as f:
    for k in ("head.weight", "norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"):
        W[k] = f.get_tensor(k).to(dev)
print({k: (tuple(v.shape), v.dtype) for k, v in W.items()})
EPS = 1e-6  # rms_norm_eps and hc_eps in config.json
head = W["head.weight"].to(torch.bfloat16)
num = json.load(open(SUB / "number_tokens.json"))
num_ids = torch.tensor(num["ids"], device=dev)
num_mask = torch.zeros(head.shape[0], dtype=torch.bool, device=dev); num_mask[num_ids] = True


def collapse(x, how):  # x [S, 4, d] -> [S, d] float32
    x = x.float()
    if how == "mean":
        return x.mean(1)
    flat = x.flatten(1)
    flat = flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + EPS)
    pre = torch.sigmoid(flat @ W["hc_head_fn"].float().T * W["hc_head_scale"].float() + W["hc_head_base"].float()) + EPS
    return (pre.unsqueeze(-1) * x).sum(1).to(torch.bfloat16).float()  # model casts back to bf16


def logprobs(h):  # final RMSNorm (weighted) + lm_head, as DeepseekV4Model / ForCausalLM do
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS)
    h = (W["norm.weight"].float() * h.to(torch.bfloat16)).to(torch.bfloat16)
    return torch.log_softmax((h @ head.T).float(), -1)


for how in ("hchead", "mean"):
    for cond in CONDS:
        out = SUB / f"logitlens_{how}" / cond
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for p in sorted((SUB / "resid" / cond).glob("*.pt")):
            if (out / f"{p.stem}.npz").exists():
                continue
            r = torch.load(p, map_location="cpu", weights_only=False)
            z = np.load(SUB / "jlens" / cond / f"{p.stem}.npz")
            allp = r["all_pos"]  # [43, S, 4, d]
            L, S = allp.shape[0], allp.shape[1]
            top_i = np.zeros((L, S, 128), np.int32); top_v = np.zeros((L, S, 128), np.float32)
            num_lp = np.zeros((L, S, len(num_ids)), np.float32)
            with torch.no_grad():
                for l in range(L):
                    lp = logprobs(collapse(allp[l].to(dev), how))
                    num_lp[l] = lp[:, num_ids].cpu().numpy()
                    v, i = lp.masked_fill(num_mask, float("-inf")).topk(128, -1)
                    top_i[l], top_v[l] = i.cpu().numpy(), v.cpu().numpy()
            np.savez(out / f"{p.stem}.npz", input_ids=z["input_ids"], seg=z["seg"], pos_offset=z["pos_offset"],
                     layers=np.arange(L, dtype=np.int16), top_ids=top_i, top_logprobs=top_v,
                     number_logprobs=num_lp, final_number_logprobs=z["final_number_logprobs"])
            if how == "hchead":  # layer 42 hchead readout must reproduce the model's own answer distribution
                err = np.abs(num_lp[-1, -1] - z["final_number_logprobs"]).max()
                print(f"{how} {cond} {p.stem}: max |L42 readout - model| over numbers at answer pos {err:.3g}", flush=True)
        print(f"{how} {cond} done in {time.time()-t0:.0f}s", flush=True)
