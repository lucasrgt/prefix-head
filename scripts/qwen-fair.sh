#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
m=data/models/llm/qwen2.5-coder-0.5b-instruct-q4_0.gguf
h() { python3 tools/eval/humaneval.py $m "$@" > /dev/null 2>> results/humaneval.log; }
h --tag fair-llamacpp-ctx1024 --ctx 1024
h --tag fair-llamacpp-ctx1024-nommap --ctx 1024 --no-mmap
h --tag fair-gem-full --gem --tag qwencoder --full
h --tag fair-gem-prefix64 --gem --tag qwencoder --head2 qwenchat-prefix-self --classes 64
h --bench mbpp --tag mbpp-llamacpp --ctx 1024
h --bench mbpp --tag mbpp-gem-full --gem --tag qwencoder --full
for k in 8 32 64; do h --bench mbpp --tag mbpp-gem-prefix$k --gem --tag qwencoder --head2 qwenchat-prefix-self --classes $k; done
