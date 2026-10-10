#!/bin/bash
export HF_HOME=/data/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=../.venv/bin/python; O=/data/out/seed_patch2
mkdir -p $O
$P seed_patch2.py --model /data/v4flash --repo /data/repo --seeds seeds100.json --items items.json --out $O/$1 ${@:2} > $O/$1.log 2>&1
$P analyze2.py $O/$1 items.json > /dev/null 2>> $O/$1.log
echo RUN_DONE >> $O/$1.log
