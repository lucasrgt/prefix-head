#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
py() { uv run -q --with numpy --with pyyaml --with tqdm --with sentencepiece env PYTHONPATH=.tools/src/llama.cpp/gguf-py python3 "$@"; }
f=data/models/llm/gpt-oss-20b-MXFP4.gguf
[ -f $f ] || curl -sL -o $f https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf
while systemctl --user is-active --quiet frugal-extract; do sleep 60; done
tag=gptoss20b
[ -f data/retro/$tag.W ] || py tools/hidden/head.py $f data/retro/$tag.W > results/retro-$tag-head.log 2>&1
[ -f data/retro/$tag-test.json ] || .tools/hidden $f data/norm/test.txt data/retro/$tag-test 1024 1 8 2> results/retro-$tag-test.log
[ -f data/retro/$tag-train.json ] || .tools/hidden $f data/norm/train.txt data/retro/$tag-train 1024 6 8 2> results/retro-$tag-train.log
