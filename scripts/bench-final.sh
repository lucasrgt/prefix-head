#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user list-units 'frugal-*' --no-legend --state=active | grep -v bench-final | grep -q .; do sleep 60; done
sleep 120
for isa in avx512 avx2; do
  for threads in 8 1; do
    for rep in 1 2 3; do
      GEM_ISA=$isa ./target/release/gem bench --tokens 512 --threads $threads > /dev/null 2>> results/gem-bench.log
    done
  done
done
for rep in 1 2 3; do
  ./target/release/gem bench --tokens 512 --threads 8 --head2 gemma270m-prefix-128-self > /dev/null 2>> results/gem-bench.log
done
