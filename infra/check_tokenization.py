# Usage: python3 check_tokenization.py <V4 Flash tokenizer.json> <eval.jsonl>
# Checks which task numbers are single tokens under the DeepSeek V4 Flash tokenizer.
import json, sys
from collections import Counter
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
enc = lambda s: tok.encode(s, add_special_tokens=False)
# 1. coverage of 0..999
missing = [n for n in range(1000) if len(enc(str(n)).ids) != 1]
print("0-999 not single token:", missing[:20], len(missing))
print("1010 ->", enc("1010").tokens, " -31 ->", enc("-31").tokens, " ' 264' ->", enc(" 264").tokens)
# 2. in context
for ctx in ["<｜Assistant｜></think>264<｜end▁of▁sentence｜>", "Answer: 264", "Answer:\n264", "zif = 48\nyaj = three times", "Filler: . . . .\n\nAnswer:"]:
    print(repr(ctx), "->", enc(ctx).tokens)
xs = [json.loads(l) for l in open(sys.argv[2])]
def ntok(v): return len(enc(str(v)).ids)
ans = Counter(ntok(x["answer"]) for x in xs)
y = Counter(ntok(x["queried_value"]) for x in xs)
c1x = Counter(ntok(x["chain"]["c1x"]) for x in xs)
c2y = Counter(ntok(x["chain"]["c2y"]) for x in xs)
print("answer tokens:", ans, " intermediate y:", y, " c1*x:", c1x, " c2*y:", c2y)
print("answers >=1000:", sorted(x["answer"] for x in xs if x["answer"] >= 1000))
print("c2y >= 1000:", sum(x["chain"]["c2y"] >= 1000 for x in xs), " neg values anywhere:", sum(any(v < 0 for v in x["values"].values()) for x in xs))
# distractor/rival values
allv = Counter(ntok(v) for x in xs for v in list(x["values"].values()) + list(x["rivals"].values()))
print("all defined+rival values:", allv)
# 3. filler tokens
d100 = " ".join(["."]*100)
print("'Filler: '+100 dots ->", len(enc("Filler: " + d100).ids), "vs 'Filler:' ->", len(enc("Filler:").ids), set(enc(d100).tokens))
