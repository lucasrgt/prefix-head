#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
py() { uv run -q --with numpy --with pyyaml --with tqdm --with sentencepiece env PYTHONPATH=.tools/src/llama.cpp/gguf-py python3 "$@"; }
while pgrep -x llama-server > /dev/null; do sleep 20; done
tag=qwencoder; f=data/models/llm/qwen2.5-coder-0.5b-instruct-q4_0.gguf
[ -f data/retro/$tag.W ] || py tools/hidden/head.py $f data/retro/$tag.W > results/retro-$tag-head.log 2>&1
[ -f data/retro/$tag-test.json ] || .tools/hidden $f data/code/test.txt data/retro/$tag-test 1024 1 8 2> results/retro-$tag-test.log
[ -f data/retro/$tag-train.json ] || .tools/hidden $f data/code/train.txt data/retro/$tag-train 1024 2 8 2> results/retro-$tag-train.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RETRO_NO_ADAPT=1
.venv-gpu/bin/python tools/gpu/retro.py $tag prefix hyb prefix@128 freq sem random > /dev/null 2>> results/retro-gpu-$tag.log || echo "qwencoder retro failed" >> results/retro-failures.txt
.venv-gpu/bin/python tools/gpu/kl.py $tag prefix hyb prefix-128 freq sem random > /dev/null 2>> results/retro-kl.log
