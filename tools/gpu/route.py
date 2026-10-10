"""Offline routing study for the two-level head on a dump of the model's own greedy generations
(`gem complete --dump`): each row is a hidden state and the token the full head chose there.

Reports class recall (is the chosen token's class among the predictor's top k), split by structural
tokens, and for several decoding policies the agreement with the full head and the average number of
output rows read per token:

  top-k       best (class, token) pair among the k most likely classes, by log P(c) + log P(t | c)
  screen-k    the k most likely classes are evaluated and the highest raw logit wins (the rows are the
              model's own, so logits compare across classes); agrees whenever the class is among the k.
              Optionally the fence and control tokens are always evaluated as well.
  adaptive    2 classes when the class margin log P(c1) − log P(c2) is below δ, else 1
  fallback    the full head when P(c1) is below τ (or the margin below δ), else top-1

Usage: .venv-gpu/bin/python tools/gpu/route.py <dump tag> <head name> [<tag the head was trained under>]
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


def main():
    tag, name = sys.argv[1], sys.argv[2]
    meta = json.loads((d / f"{tag}-test.json").read_text())
    v, dim = meta["vocab"], meta["d"]
    h = torch.from_numpy(np.fromfile(d / f"{tag}-test.h", dtype=np.float16).reshape(-1, dim)).to(dev)
    y = torch.from_numpy(np.fromfile(d / f"{tag}-test.y", dtype=np.uint32).astype(np.int64)).to(dev)
    w = torch.from_numpy(np.fromfile(d / f"{tag}.W", dtype=np.float16).reshape(v, dim)).to(dev)
    base = d / f"{sys.argv[3] if len(sys.argv) > 3 else tag}-{name}-self"
    order = np.fromfile(base.with_suffix(".order"), dtype=np.uint32).astype(np.int64)
    b = torch.from_numpy(np.fromfile(base.with_suffix(".b"), dtype=np.float32)).to(dev)
    k = len(b)
    a = torch.from_numpy(np.fromfile(base.with_suffix(".A"), dtype=np.float32).reshape(k, dim)).to(dev)
    m = math.ceil(v / k)
    g = retro.Groups(order, v, m, k)
    pcs = retro.pieces(d / f"{tag}-test.vocab")
    kinds = {
        "fence": torch.tensor([b"```" in p for p in pcs], device=dev),
        "newline": torch.tensor([b"\n" in p for p in pcs], device=dev),
        "end": torch.tensor([p.startswith(b"<|") for p in pcs], device=dev),
    }
    n = len(y)
    rank_of, conf, margin, best_tok, best_score = [], [], [], [], []
    for s in range(0, n, 256):
        hb = h[s : s + 256]
        lc = torch.log_softmax(hb.float() @ a.T + b, 1)
        lg = (hb @ w.T).float()
        blocks = lg[:, g.pad.clamp(min=0)].masked_fill(~g.valid, float("-inf")).view(-1, k, m)
        within = torch.log_softmax(blocks, 2)
        top_tok = within.max(2)
        score = lc + top_tok.values
        srt = lc.sort(1, descending=True)
        cy = g.cls[y[s : s + 256]]
        rank_of.append((srt.indices == cy[:, None]).float().argmax(1))
        conf.append(srt.values[:, 0].exp())
        margin.append(srt.values[:, 0] - srt.values[:, 1])
        cand_score = score.gather(1, srt.indices[:, :4])
        cand_tok = g.pad.view(k, m)[srt.indices[:, :4], top_tok.indices.gather(1, srt.indices[:, :4])]
        best_tok.append(cand_tok)
        best_score.append(cand_score)
    rank_of, conf, margin = torch.cat(rank_of), torch.cat(conf), torch.cat(margin)
    best_tok, best_score = torch.cat(best_tok), torch.cat(best_score)

    def pick(kk):
        return best_tok.gather(1, best_score[:, :kk].argmax(1, keepdim=True))[:, 0]

    out = {"dump": tag, "head": name, "classes": k, "block": m, "rows": n, "recall": {}, "policies": []}
    for label, mask in [("all", torch.ones(n, dtype=torch.bool, device=dev))] + [(kk, kinds[kk][y]) for kk in kinds]:
        cnt = int(mask.sum())
        out["recall"][label] = {"tokens": cnt, **{f"top{kk}": round((rank_of[mask] < kk).float().mean().item(), 4) if cnt else None for kk in (1, 2, 4)}}

    def policy(label, tok, rows_read):
        agree = (tok == y).float().mean().item()
        out["policies"].append({"policy": label, "agreement": round(agree, 4), "rows_read": round(float(rows_read), 1), "head_cut": round(v / float(rows_read), 1)})

    for kk in (1, 2, 4):
        policy(f"top-{kk}", pick(kk), k + kk * m)
    special = kinds["fence"] | kinds["end"]
    extra = int(special.sum())
    for kk in (1, 2, 4, 8, 16, 32, 64):
        hit = rank_of < kk
        policy(f"screen-{kk} (raw logits)", torch.where(hit, y, -torch.ones_like(y)), k + kk * m)
        policy(f"screen-{kk} + {extra} always-on rows", torch.where(hit | special[y], y, -torch.ones_like(y)), k + kk * m + extra)
    for delta in (0.5, 1.0, 2.0, 4.0):
        two = margin < delta
        policy(f"adaptive δ={delta}", torch.where(two, pick(2), pick(1)), k + m * (1 + two.float().mean().item()))
    for tau in (0.5, 0.7, 0.9, 0.97, 0.99):
        fb = conf < tau
        policy(f"fallback τ={tau}", torch.where(fb, y, pick(1)), k + m + v * fb.float().mean().item())
    for delta in (1.0, 2.0, 4.0):
        fb = margin < delta
        policy(f"fallback δ={delta}", torch.where(fb, y, pick(1)), k + m + v * fb.float().mean().item())
    for delta, tau in ((2.0, 0.5), (4.0, 0.7)):
        fb = conf < tau
        two = (margin < delta) & ~fb
        policy(f"adaptive δ={delta} + fallback τ={tau}", torch.where(fb, y, torch.where(two, pick(2), pick(1))), k + m * (1 + two.float().mean().item()) + v * fb.float().mean().item())
    print(json.dumps(out, indent=1))
    with open(root / "results/route.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
