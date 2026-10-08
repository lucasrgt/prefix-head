#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-freqsplit || systemctl --user is-active --quiet frugal-pending; do sleep 30; done
export MATMUL_NUM_THREADS=8
python3 - <<'PY'
import json
lines=open('results/alpha.jsonl').read().splitlines()
open('results/alpha.jsonl','w').write('\n'.join(l for l in lines if json.loads(l).get('seed')!=3)+'\n')
PY
sosc="--model mingru --core sosc --n 51 --h 128 --layers 2 --seq 64 --batch 16 --stateful 64"
for g in grp-first-91 grp-sem-91; do
  ./target/release/alpha --alpha $g:bpe8192 --epochs 3 --seed 3 > /dev/null 2> results/seed3-mlp-$g.log
  ./target/release/alpha $sosc --alpha $g:bpe8192 --epochs 3 --seed 3 > /dev/null 2> results/seed3-sosc-$g.log
done
