"""Exact greedy decoding with fewer output-head reads, two graphics-style schemes, simulated on the
test dump of a frozen model.

  bvh     bounding spheres per block (centre = mean row, radius = farthest row); a block whose bound
          h·c + |h|·r cannot beat the best logit found so far is skipped. Blocks are visited in
          bound order, so the visited ones are those whose bound reaches the true maximum.
  mipmap  a low-precision copy of every row (absmax per 32 weights) gives each logit with a known
          error bound |h|·|w − ŵ|; only rows whose interval reaches the best lower bound are
          recomputed at full precision.

Usage: .venv-gpu/bin/python tools/gpu/cull.py <tag> [rows]
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


def quantize(w, bits):
    blocks = w.view(w.shape[0], -1, 32)
    q = 2 ** (bits - 1) - 1
    scale = blocks.abs().amax(2, keepdim=True) / q
    return ((blocks / scale.clamp(min=1e-12)).round().clamp(-q - 1, q) * scale).view_as(w)


def main():
    tag = sys.argv[1]
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    d = root / "data/retro"
    meta = json.loads((d / f"{tag}-test.json").read_text())
    v, dim = meta["vocab"], meta["d"]
    w = torch.from_numpy(np.fromfile(d / f"{tag}.W", dtype=np.float16).reshape(v, dim)).to(dev).float()
    h = torch.from_numpy(np.fromfile(d / f"{tag}-test.h", dtype=np.float16, count=rows * dim).reshape(rows, dim)).to(dev).float()
    pcs = retro.pieces(d / f"{tag}-test.vocab")
    out = []

    def emit(scheme, variant, read_rows, bytes_read, note):
        line = {"model": tag, "scheme": scheme, "variant": variant, "rows_read": round(read_rows, 1), "rows_cut": round(v / read_rows, 1), "bytes_read": round(bytes_read), "bytes_cut_vs_f16": round(v * dim * 2 / bytes_read, 1), "note": note}
        print(json.dumps(line), flush=True)
        out.append(line)

    for m in [128, 512]:
        orders = {
            "prefix": np.array(sorted(range(v), key=lambda t: (pcs[t], t))),
            "random": np.random.default_rng(7).permutation(v),
            "semantic": retro.semantic(w.half(), m, 7),
        }
        for name, order in orders.items():
            g = retro.Groups(order, v, m)
            blk = g.blocks(w)
            valid = g.valid.view(g.k, m)
            centre = blk.sum(1) / valid.sum(1, keepdim=True)
            radius = ((blk - centre[:, None]).norm(dim=2) * valid).amax(1)
            visited = 0.0
            for s in range(0, rows, 512):
                hb = h[s : s + 512]
                best = (hb @ w.T).amax(1, keepdim=True)
                bound = hb @ centre.T + hb.norm(dim=1, keepdim=True) * radius
                visited += (bound >= best).sum().item()
            per = visited / rows
            emit("bvh", f"{name}@{m}", g.k + per * m, (g.k * (dim + 1)) * 2 + per * m * dim * 2, f"{per:.1f} of {g.k} blocks visited")
    for bits in [2, 3, 4]:
        wq = quantize(w, bits)
        err = (w - wq).norm(dim=1)
        cand = 0.0
        for s in range(0, rows, 512):
            hb = h[s : s + 512]
            approx = hb @ wq.T
            slack = hb.norm(dim=1, keepdim=True) * err
            lower = (approx - slack).amax(1, keepdim=True)
            cand += (approx + slack >= lower).sum().item()
        per = cand / rows
        emit("mipmap", f"int{bits}", v + per, v * dim * bits / 8 + v * dim / 32 * 2 + per * dim * 2, f"{per:.1f} rows refined")
    with open(root / "results/cull.jsonl", "a") as f:
        for line in out:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
