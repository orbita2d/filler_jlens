"""Shared pieces: prompts (from the example repo, unchanged), tokenizer helpers, model loading,
residual capture and the Jacobian-lens readout for DeepSeek V4 Flash."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------------------------
# Prompts. Text comes from the example repo's scripts/prompts.py so it is identical to theirs.
# ---------------------------------------------------------------------------------------------
BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
NO_THINK = "</think>"  # V4 non-thinking mode: assistant turn opens with an empty think block

CONDITIONS = {"k0": ("dots", 0), "dots10": ("dots", 10), "dots100": ("dots", 100)}

# Segment labels stored per prompt position.
SEG_OTHER, SEG_DEFS, SEG_QUESTION, SEG_FILLER, SEG_ANSWER = 0, 1, 2, 3, 4


def load_repo(repo):
    """Import the example repo's prompts.py and read its task files."""
    repo = Path(repo)
    sys.path.insert(0, str(repo / "scripts"))
    import prompts  # noqa: E402  (the repo's own module)
    read = lambda p: [json.loads(l) for l in open(repo / "data" / "tasks" / p)]
    return prompts, read("eval.jsonl"), read("fewshot.jsonl")


def render(messages):
    """DeepSeek V4 chat format, thinking off. check.py compares this with the tokenizer's own
    chat template and prints any difference."""
    s = BOS
    for m in messages:
        if m["role"] == "system":
            s += m["content"]
        elif m["role"] == "user":
            s += USER + m["content"]
        elif m["role"] == "assistant":
            s += ASSISTANT + NO_THINK + m["content"] + EOS
    return s + ASSISTANT + NO_THINK


def build_prompt(prompts, fewshot, item, cond, n_shot):
    kind, k = CONDITIONS[cond]
    messages = prompts.build_messages(fewshot, item, kind=kind, k=k, n_shot=n_shot)
    return render(messages), messages


def tokenize_prompt(tok, text):
    """Token ids plus a segment label for every position of the test item's turn."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    starts = np.array([a for a, _ in enc["offset_mapping"]])
    u = text.rfind(USER) + len(USER)  # start of the test item's user turn
    q = text.index("\nQuestion:", u) + 1
    f = text.index("\n\nFiller:", q) + 2
    a = text.index("\n\nAnswer:", f) + 2
    seg = np.full(len(ids), SEG_OTHER, dtype=np.int8)
    seg[(starts >= u) & (starts < q)] = SEG_DEFS
    seg[(starts >= q) & (starts < f)] = SEG_QUESTION
    seg[(starts >= f) & (starts < a)] = SEG_FILLER
    seg[starts >= a] = SEG_ANSWER
    return ids, seg


def number_token_ids(tok):
    """Every vocab entry whose text is all ASCII digits (0-999 plus forms like '00', '007')."""
    vocab = tok.get_vocab()
    pairs = sorted((int(t) if len(t) < 4 else 10**6, t, i) for t, i in vocab.items()
                   if t.isascii() and t.isdigit())
    ids = np.array([i for _, _, i in pairs], dtype=np.int64)
    texts = [t for _, t, _ in pairs]
    return ids, texts


def answer_ids(tok, value):
    return tok.encode(str(value), add_special_tokens=False)


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------
def load_model(path, max_memory=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    kw = dict(device_map="auto", dtype="auto")
    if max_memory:
        kw["max_memory"] = max_memory
    model = AutoModelForCausalLM.from_pretrained(path, **kw).eval()
    return tok, model


class ResidCapture:
    """Forward hooks on every decoder layer. out[l] is the residual after layer l,
    shape [batch, seq, hc_mult, d_model] (V4 keeps hc_mult parallel streams)."""

    def __init__(self, model):
        self.out = {}
        self.handles = []
        for l, layer in enumerate(model.model.layers):
            self.handles.append(layer.register_forward_hook(self._hook(l)))

    def _hook(self, l):
        def f(_mod, _inp, output):
            self.out[l] = (output[0] if isinstance(output, tuple) else output).detach()
        return f

    def clear(self):
        self.out = {}

    def remove(self):
        for h in self.handles:
            h.remove()


# How to turn the hc_mult streams into the single d_model vector the lens reads. The lens
# release does not say for V4, so check.py tries each one on the spider prompt and picks.
def collapse(model, x, how):
    """x: [seq, hc_mult, d_model] -> [seq, d_model] float32."""
    if how == "mean":
        return x.float().mean(-2)
    if how == "sum":
        return x.float().sum(-2)
    if how == "hchead":  # the model's own final collapse, applied at this layer
        head = model.model.hc_head
        dev = head.hc_fn.device
        return head(x.to(dev).unsqueeze(0))[0].float()
    if how.startswith("stream"):
        return x[:, int(how[6:])].float()
    raise ValueError(how)


COLLAPSES = ["mean", "sum", "hchead", "stream0"]


class JLens:
    """A Jacobian lens saved by jlens: {"J": {layer: [d, d] fp16}, "source_layers", "d_model",
    ...}. Readout: logits = lm_head(final_norm(J_l h_l))."""

    def __init__(self, path, model, collapse_how="mean", transpose=False, device=None):
        raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(raw, dict) or "J" not in raw:
            raise ValueError(f"{path}: not a saved jlens (keys: {list(raw)[:12] if isinstance(raw, dict) else type(raw)})")
        self.meta = {k: (v if not torch.is_tensor(v) else tuple(v.shape)) for k, v in raw.items() if k != "J"}
        self.model = model
        self.norm = model.model.norm
        self.head = model.lm_head
        self.device = device or self.head.weight.device
        self.J = {int(k): v.to(self.device, torch.float16) for k, v in raw["J"].items()}
        self.layers = sorted(self.J)
        self.collapse_how = collapse_how
        self.transpose = transpose

    def readout_logits(self, h, layer):
        """h: [seq, d_model] float32 at the residual after `layer`. Returns float32 logits."""
        J = self.J[layer].float()
        h = h.to(self.device)
        pred = h @ J if self.transpose else h @ J.T
        return self.final_logits(pred)

    def final_logits(self, h):
        """Final norm + unembedding on a collapsed [seq, d_model] vector."""
        w = self.head.weight
        dtype = w.dtype if w.element_size() >= 2 else torch.bfloat16  # fp8 head: feed bf16
        x = self.norm(h.to(self.norm.weight.device, torch.bfloat16))
        return self.head(x.to(w.device, dtype)).float()

    def layer_logprobs(self, streams, layer):
        """streams: [seq, hc_mult, d_model] residual after `layer` -> [seq, vocab] log-probs."""
        h = collapse(self.model, streams, self.collapse_how)
        return torch.log_softmax(self.readout_logits(h, layer), -1)


def top_non_number(logprobs, num_mask, k):
    """Top-k over the vocabulary with number tokens excluded. logprobs [..., V]."""
    masked = logprobs.masked_fill(num_mask, float("-inf"))
    v, i = masked.topk(k, dim=-1)
    return i.to(torch.int32), v.to(torch.float16)


class LogitLens(JLens):
    """Logit lens: logits = lm_head(final_norm(collapse(h_l))), no J. Same interface as JLens; covers every
    decoder layer (0..n_layers-1). With collapse "hchead" the last layer reproduces the model's logits."""

    def __init__(self, model, collapse_how="hchead"):
        self.meta = {"kind": "logit_lens", "collapse": collapse_how}
        self.model = model
        self.norm = model.model.norm
        self.head = model.lm_head
        self.device = self.head.weight.device
        self.layers = list(range(len(model.model.layers)))
        self.collapse_how = collapse_how
        self.transpose = False

    def readout_logits(self, h, layer):
        return self.final_logits(h)
