#!/bin/bash
export HF_HOME=/data/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=../.venv/bin/python; O=/data/out/seed_patch
$P seed_patch.py --model /data/v4flash --repo /data/repo --seeds seeds.json --items items.json --out $O/main --check 5 > $O/main.log 2>&1
$P analyze.py $O/main items.json > /dev/null 2>> $O/main.log
echo MAIN_DONE >> $O/main.log
