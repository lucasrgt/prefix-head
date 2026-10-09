"""Speculative-decoding acceptance of a small drafter against a larger target on the same text.

For each test position, with p the target's next-token distribution and q the drafter's:
  alpha  = sum_t min(p_t, q_t)            (acceptance probability of a sampled draft token)
  greedy = [argmax q == argmax p]         (acceptance of a greedy draft token)
Drafter heads: the full head; two-level heads (exact factorization, Section 2 of the paper) for each
saved class predictor; and the top-k screening used by Gemma 4's drafters (score classes, keep the
top 32, softmax over their tokens only) on the same orderings.
Usage: .venv-gpu/bin/python tools/gpu/spec.py <draft tag> <target tag> <head> [<head> ...]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import retro  # noqa: E402

root, dev = retro.root, retro.dev
d = root / "data/retro"


def load(tag):
    meta = json.loads((d / f"{tag}-test.json").read_text())
    h = torch.from_numpy(np.fromfile(d / f"{tag}-test.h", dtype=np.float16).reshape(-1, meta["d"])).to(dev)
    w = torch.from_numpy(np.fromfile(d / f"{tag}.W", dtype=np.float16).reshape(meta["vocab"], meta["d"])).to(dev)
    capf = d / f"{tag}.softcap"
    return h, w, float(capf.read_text()) if capf.exists() else None


def softcap(x, cap):
    return x if cap is None else cap * torch.tanh(x / cap)


def head(tag, name, v, dim):
    base = d / f"{tag}-{name}-self"
    order = torch.from_numpy(np.fromfile(base.with_suffix(".order"), dtype=np.uint32).astype(np.int64)).to(dev)
    b = torch.from_numpy(np.fromfile(base.with_suffix(".b"), dtype=np.float32)).to(dev)
    k = len(b)
    a = torch.from_numpy(np.fromfile(base.with_suffix(".A"), dtype=np.float32).reshape(k, dim)).to(dev)
    m = math.ceil(v / k)
    cls = torch.empty(v, dtype=torch.long, device=dev)
    cls[order] = torch.arange(len(order), device=dev) // m
    return a, b, cls, k, m


def main():
    draft, target = sys.argv[1], sys.argv[2]
    names = sys.argv[3:]
    hd, wd, capd = load(draft)
    ht, wt, capt = load(target)
    v, dim = wd.shape
    heads = {n: head(draft, n, v, dim) for n in names}
    variants = ["full"] + [f"{n}:exact" for n in names] + [f"{n}:top32" for n in names if heads[n][3] >= 512]
    stats = {k: [0.0, 0.0] for k in variants}
    n = len(hd)
    for s in range(0, n, 128):
        with torch.no_grad():
            lt = softcap((ht[s : s + 128] @ wt.T).float(), capt)
            p = torch.softmax(lt, 1)
            best = p.argmax(1)
            ld = softcap((hd[s : s + 128] @ wd.T).float(), capd)
            qs = {"full": torch.softmax(ld, 1)}
            for name, (a, b, cls, k, m) in heads.items():
                lc = torch.log_softmax(hd[s : s + 128].float() @ a.T + b, 1)
                within = ld - torch.zeros(len(ld), k, device=dev).scatter_reduce(1, cls.expand(len(ld), -1), ld, "amax", include_self=False).gather(1, cls.expand(len(ld), -1))
                ex = within.exp()
                denom = torch.zeros(len(ld), k, device=dev).index_add_(1, cls, ex)
                qs[f"{name}:exact"] = (lc.gather(1, cls.expand(len(ld), -1)).exp()) * ex / denom.gather(1, cls.expand(len(ld), -1))
                if f"{name}:top32" in stats:
                    keep = torch.zeros(len(ld), k, dtype=torch.bool, device=dev).scatter_(1, lc.topk(32, 1).indices, True)
                    masked = ld.masked_fill(~keep.gather(1, cls.expand(len(ld), -1)), float("-inf"))
                    qs[f"{name}:top32"] = torch.softmax(masked, 1)
            for key, q in qs.items():
                stats[key][0] += torch.minimum(p, q).sum().item()
                stats[key][1] += (q.argmax(1) == best).sum().item()
    for key, (alpha, greedy) in stats.items():
        line = {"draft": draft, "target": target, "head": key, "alpha": round(alpha / n, 4), "greedy": round(greedy / n, 4), "positions": n}
        print(json.dumps(line), flush=True)
        with open(root / "results/spec.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
