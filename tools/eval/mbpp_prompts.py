"""Writes ChatML prompts built from MBPP problem statements, for collecting the model's own
generations (hidden states at the positions where the head is actually used). HumanEval is never
touched. Two instruction styles alternate: a plain request and a HumanEval-style signature to complete.
Usage: python3 tools/eval/mbpp_prompts.py <out_dir> <first> <last>"""
import json
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from humaneval import CHATML  # noqa: E402

out, first, last = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in (root / "data/mbpp/mbpp.jsonl").read_text().splitlines()][first:last]
for i, r in enumerate(rows):
    sig = re.search(r"^def .*:\s*$", r["code"].replace("\r", ""), re.M)
    if i % 2 == 0 or not sig:
        ask = r["text"].strip() + " Reply with the full function in a single Python code block."
    else:
        ask = "Complete the following Python function. Reply with the full function in a single Python code block.\n\n```python\n" + sig.group(0).rstrip() + f'\n    """ {r["text"].strip()}\n    """\n```'
    (out / f"{i:04d}.txt").write_text(CHATML.format(ask))
