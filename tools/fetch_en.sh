#!/usr/bin/env bash
# 14 public-domain 19th-century English novels for the cross-language replication.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/en/raw
for id in 1342 1400 1260 768 2701 84 345 98 158 145 174 161 766 1661; do
  f=data/en/raw/$id.txt
  [ -s "$f" ] && continue
  curl -s --http1.1 --retry 3 --max-time 120 -L -o "$f.part" "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" && mv "$f.part" "$f" || rm -f "$f.part"
  sleep 1.5
done
ls data/en/raw | wc -l; du -sh data/en/raw
