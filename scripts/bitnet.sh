#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RETRO_NO_ADAPT=1
tag=bitnet2b; src=data/models/bitnet-bf16
[ -f data/retro/$tag-test.json ] || .venv-gpu/bin/python tools/gpu/hidden_hf.py $src data/norm/test.txt data/retro/$tag-test 1024 1 --head data/retro/$tag.W 2> results/retro-$tag-test.log || { echo "bitnet test extraction failed" >> results/retro-failures.txt; exit 1; }
[ -f data/retro/$tag-train.json ] || .venv-gpu/bin/python tools/gpu/hidden_hf.py $src data/norm/train.txt data/retro/$tag-train 1024 3 2> results/retro-$tag-train.log || { echo "bitnet train extraction failed" >> results/retro-failures.txt; exit 1; }
for seed in 1 2 3; do
  RETRO_SEED=$seed .venv-gpu/bin/python tools/gpu/retro.py $tag prefix sem > /dev/null 2>> results/retro-gpu-$tag.log || echo "bitnet retro $seed failed" >> results/retro-failures.txt
  suffix=""; [ $seed -ne 1 ] && suffix="-s$seed"
  .venv-gpu/bin/python tools/gpu/kl.py $tag prefix$suffix sem$suffix > /dev/null 2>> results/retro-kl.log
done
.venv-gpu/bin/python tools/gpu/retro.py $tag freq random id hyb > /dev/null 2>> results/retro-gpu-$tag.log
.venv-gpu/bin/python tools/gpu/kl.py $tag freq random id hyb > /dev/null 2>> results/retro-kl.log
