#!/bin/bash
# Wait for the 10-shot rollouts, pick the 50-item subset, run the full-precision lens + full residuals on it.
set -euo pipefail
cd ~/fj/code
until [ "$(wc -l < /data/out/rollout_10shot.jsonl)" -ge 1800 ]; do sleep 30; done
../.venv/bin/python select_subset.py /data/out/rollout_10shot.jsonl /data/out/subset.json
ITEMS=$(../.venv/bin/python -c "import json; s=json.load(open('/data/out/subset.json')); print(','.join(map(str, sorted(s['uplift'] + s['no_uplift']))))")
echo "items: $ITEMS"
HF_HOME=/data/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ../.venv/bin/python run.py --model /data/v4flash --repo /data/repo \
  --out /data/out/subset --shots 10 --conds k0,dots10,dots100 --items "$ITEMS" \
  --lens /data/jlens/deepseek-v4-flash/jlens/NeelNanda-pile-10k/deepseek-v4-flash_jacobian_lens_cblank.pt \
  --precise --topk 128 --save-resid --resid-layers all
echo SUBSET_DONE
