#!/usr/bin/env bash
# Downloads the extra Portuguese Gutenberg books listed in data/extra/ids.json, politely, once.
set -uo pipefail
cd "$(dirname "$0")/.."
for id in $(python3 -c "import json; print(' '.join(map(str, json.load(open('data/extra/ids.json')))))"); do
  f=data/extra/raw/$id.txt
  [ -s "$f" ] && continue
  curl -s --http1.1 --retry 3 --max-time 120 -L -o "$f.part" "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" && mv "$f.part" "$f" || rm -f "$f.part"
  sleep 1.5
done
echo "done: $(ls data/extra/raw | wc -l) files, $(du -sh data/extra/raw | cut -f1)"
