#!/bin/bash
export HF_HOME=/data/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=../.venv/bin/python; M=/data/v4flash; R=/data/repo; O=/data/out/seed_patch
$P seed_patch.py --model $M --repo $R --seeds seeds.json --items items.json --out $O/pilot --limit 5 --check 5 > $O/pilot.log 2>&1
$P analyze.py $O/pilot items.json > /dev/null 2>> $O/pilot.log
$P seed_patch.py --model $M --repo $R --seeds seeds.json --items items.json --out $O/pilot_nocache --no-prefix-cache --limit 5 --check 5 > $O/pilot_nocache.log 2>&1
$P analyze.py $O/pilot_nocache items.json > /dev/null 2>> $O/pilot_nocache.log
echo PILOTS_DONE >> $O/pilot_nocache.log
