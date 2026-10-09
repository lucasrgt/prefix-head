"""Associative-recall corpus: each line lists N key:value pairs (3 random lowercase letters each),
then ' ; ', then every key again in random order with its value. Values in the query half are
predictable only by recalling the first half. N is drawn from {2, 4, 8, 16, 32} per line.
Usage: python3 tools/mqar/gen.py <out_dir> <files> <chars_per_file>"""
import random
import string
import sys
from pathlib import Path

out, files, size = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
rnd = random.Random(1)
word = lambda: "".join(rnd.choice(string.ascii_lowercase) for _ in range(3))
for f in range(files):
    lines, total = [], 0
    while total < size:
        n = rnd.choice([2, 4, 8, 16, 32])
        keys = set()
        while len(keys) < n:
            keys.add(word())
        pairs = [(k, word()) for k in keys]
        q = pairs[:]
        rnd.shuffle(q)
        line = " ".join(f"{k}:{v}" for k, v in pairs) + " ; " + " ".join(f"{k}:{v}" for k, v in q) + "\n"
        lines.append(line)
        total += len(line)
    (out / f"{f:02d}.txt").write_text("".join(lines))
