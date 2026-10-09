#!/usr/bin/env bash
# The 14 public-domain Portuguese novels used for every main result (data/raw).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
for id in 18220 21406 23145 26988 31347 31971 40409 42942 54829 55682 55752 67535 67740 69187; do
  f=data/raw/$id.txt
  [ -s "$f" ] && continue
  curl -s --http1.1 --retry 3 --max-time 120 -L -o "$f.part" "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" && mv "$f.part" "$f" || rm -f "$f.part"
  sleep 1.5
done
echo "done: $(ls data/raw | wc -l) files"
