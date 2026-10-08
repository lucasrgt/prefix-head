#!/usr/bin/env bash
# Scores data/norm/test.txt with a GGUF model in llama.cpp (prefill, 8 threads) and reports bits per
# char and marginal package energy per char. Usage: tools/llm_baseline.sh <model.gguf> <tokens_per_char>
set -euo pipefail
cd "$(dirname "$0")/.."
M=$1; TPC=$2; CTX=1024
BIN=.tools/src/llama.cpp/build/bin
E=/sys/class/powercap/intel-rapl:0/energy_uj
e0=$(cat $E); sleep 3; e1=$(cat $E)
idle=$(python3 -c "print(($e1-$e0)/1e6/3)")
t0=$(date +%s.%N); e2=$(cat $E)
out=$($BIN/llama-perplexity -m "$M" -f data/norm/test.txt -c $CTX -b $CTX -t 8 2>&1)
e3=$(cat $E); t1=$(date +%s.%N)
ppl=$(echo "$out" | grep -oP 'Final estimate: PPL = \K[0-9.]+')
chunks=$(echo "$out" | grep -oP 'calculating perplexity over \K[0-9]+')
python3 - "$M" "$ppl" "$chunks" "$TPC" "$CTX" "$idle" "$e2" "$e3" "$t0" "$t1" <<'PY'
import sys, math
m, ppl, chunks, tpc, ctx, idle, e2, e3, t0, t1 = sys.argv[1:]
ppl, chunks, tpc, ctx, idle = float(ppl), int(chunks), float(tpc), int(ctx), float(idle)
secs = float(t1) - float(t0)
wrap = int(open("/sys/class/powercap/intel-rapl:0/max_energy_range_uj").read())
d = int(e3) - int(e2)
joules = (d if d >= 0 else d + wrap) / 1e6
chars = chunks * ctx / tpc
marginal = joules - idle * secs
print(f"{m.split('/')[-1]}: ppl {ppl:.3f} -> {math.log2(ppl) * tpc:.4f} bpc | {chunks} chunks x {ctx} tok, {secs:.1f} s, "
      f"{joules:.0f} J total, idle {idle:.1f} W, marginal {marginal / chars * 1e6:.1f} uJ/char, {secs / chars * 1e6:.2f} us/char")
PY
