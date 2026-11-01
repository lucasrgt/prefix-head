#!/usr/bin/env bash
# Hidden-state extraction for the code-domain KL test (Python stdlib corpus in data/code/{train,test}.txt).
# Tags: qcoder = Qwen2.5-Coder-0.5B-Instruct, qwen06b-code = Qwen3-0.6B (same tags + "-code" keep the Portuguese dumps intact).
set -uo pipefail
cd "$(dirname "$0")/.."
py() { uv run -q --with numpy --with pyyaml --with tqdm --with sentencepiece env PYTHONPATH=.tools/src/llama.cpp/gguf-py python3 "$@"; }
for spec in "${@:-qcoder:qwen2.5-coder-0.5b-instruct-q8_0}"; do
  tag=${spec%%:*}; f=data/models/llm/${spec#*:}.gguf
  [ -f data/retro/$tag.W ] || py tools/hidden/head.py $f data/retro/$tag.W > results/code-$tag-head.log 2>&1
  [ -f data/retro/$tag-test.json ]  || .tools/hidden $f data/code/test.txt  data/retro/$tag-test  1024 1 12 2> results/code-$tag-test.log
  [ -f data/retro/$tag-train.json ] || .tools/hidden $f data/code/train.txt data/retro/$tag-train 1024 3 12 2> results/code-$tag-train.log
done
echo EXTRACT_DONE "$@"
