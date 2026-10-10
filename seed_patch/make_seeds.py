"""WikiText-103 "seeds": --n passages of exactly --length tokens with no digits anywhere, one per paragraph,
taken in file order from the test split. seeds.json = {"seeds": [{"ids", "text", "source"}]}.
    python make_seeds.py --model M --out seeds.json --n 32 --length 128 [--text-file wiki.txt]"""
import argparse, json, re

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=32); ap.add_argument("--length", type=int, default=128)
ap.add_argument("--text-file", help="offline: one paragraph per line instead of the HF hub")
a = ap.parse_args()

if a.text_file:
    lines = open(a.text_file).read().splitlines()
    src = [f"{a.text_file}#{j}" for j in range(len(lines))]
else:
    # test, then validation, then the first train shard: the test split alone has only ~65 usable paragraphs.
    # Order is fixed, so the first 32 seeds are the same as in the 32-seed version.
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    lines, src = [], []
    for f in ["wikitext-103-raw-v1/test-00000-of-00001.parquet", "wikitext-103-raw-v1/validation-00000-of-00001.parquet",
              "wikitext-103-raw-v1/train-00000-of-00002.parquet"]:
        t = pq.read_table(hf_hub_download("Salesforce/wikitext", f, repo_type="dataset")).column("text").to_pylist()
        lines += t; src += [f"Salesforce/wikitext:{f}#{j}" for j in range(len(t))]

tok = AutoTokenizer.from_pretrained(a.model)
seeds = []
for j, t in enumerate(lines):
    t = t.strip()
    if len(t) < 6 * a.length or t.startswith("=") or re.search(r"\d", t):
        continue
    ids = tok(t, add_special_tokens=False)["input_ids"][: a.length]
    text = tok.decode(ids)
    if len(ids) == a.length and not re.search(r"\d", text) and tok(text, add_special_tokens=False)["input_ids"] == ids:
        seeds.append({"ids": ids, "text": text, "source": src[j]})
    if len(seeds) == a.n:
        break
assert len(seeds) == a.n, f"only {len(seeds)} seeds"
json.dump({"seeds": seeds, "length": a.length}, open(a.out, "w"), indent=0)
print(f"{len(seeds)} seeds of {a.length} tokens -> {a.out}")
