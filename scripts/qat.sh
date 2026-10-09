#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
py() { uv run -q --with numpy --with pyyaml --with tqdm --with sentencepiece env PYTHONPATH=.tools/src/llama.cpp/gguf-py python3 "$@"; }
tag=gemma270mqat; f=data/models/llm/gemma-3-270m-it-qat-Q4_0.gguf
[ -f data/retro/$tag.W ] || py tools/hidden/head.py $f data/retro/$tag.W > results/retro-$tag-head.log 2>&1
[ -f data/retro/$tag-test.json ] || .tools/hidden $f data/norm/test.txt data/retro/$tag-test 1024 1 8 2> results/retro-$tag-test.log
[ -f data/retro/$tag-train.json ] || .tools/hidden $f data/norm/train.txt data/retro/$tag-train 1024 3 8 2> results/retro-$tag-train.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RETRO_NO_ADAPT=1
.venv-gpu/bin/python tools/gpu/retro.py $tag prefix prefix@128 > /dev/null 2>> results/retro-gpu-$tag.log || echo "qat retro failed" >> results/retro-failures.txt
.venv-gpu/bin/python tools/gpu/kl.py $tag prefix prefix-128 > /dev/null 2>> results/retro-kl.log
sleep 60
for model in "gemma-3-270m-it-qat-Q4_0 gemma270mqat" "gemma-3-270m-it-qat-Q8_0 gemma270mqat" "gemma-3-270m-it-Q8_0 gemma270m"; do
  set -- $model
  for rep in 1 2 3; do
    ./target/release/gem bench --model data/models/llm/$1.gguf --tag $2 --tokens 512 --threads 8 > /dev/null 2>> results/gem-bench.log
  done
done
