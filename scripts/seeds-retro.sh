#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RETRO_NO_ADAPT=1
for seed in 2 3; do
  for tag in gemma270m qwen06b gemma1b gemma4e2b gptoss20b; do
    case $tag in gemma*) groups="prefix sem prefix@128 g4@128";; *) groups="prefix sem";; esac
    RETRO_SEED=$seed .venv-gpu/bin/python tools/gpu/retro.py $tag $groups > /dev/null 2>> results/retro-gpu-seeds.log || echo "seeds $tag $seed failed" >> results/retro-failures.txt
    kl=""; for g in $groups; do kl="$kl ${g/@/-}-s$seed"; done
    .venv-gpu/bin/python tools/gpu/kl.py $tag $kl > /dev/null 2>> results/retro-kl.log || echo "kl $tag $seed failed" >> results/retro-failures.txt
  done
done
