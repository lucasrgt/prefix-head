"""Fidelity of a two-level head to the model's own full head, independent of calibration.

For each saved class predictor (data/retro/<tag>-<group>-self.{order,A,b}) reports, on the test dump,
KL(P_full || P_2lvl) in bits/char. P(token | class) is exact, so the KL is the class-level KL.
Usage: .venv-gpu/bin/python tools/gpu/kl.py <tag> [<group> ...]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parents[2]
dev = "cuda"


def main():
    tag = sys.argv[1]
    d = root / "data/retro"
    meta = json.loads((d / f"{tag}-test.json").read_text())
    v, dim, chars = meta["vocab"], meta["d"], meta["chars"]
    cap_file = d / f"{tag}.softcap"
    cap = float(cap_file.read_text()) if cap_file.exists() else None
    h = torch.from_numpy(np.fromfile(d / f"{tag}-test.h", dtype=np.float16).reshape(-1, dim)).to(dev)
    w = torch.from_numpy(np.fromfile(d / f"{tag}.W", dtype=np.float16).reshape(v, dim)).to(dev)
    groups = sys.argv[2:] or sorted(p.name[len(tag) + 1 : -len("-self.A")] for p in d.glob(f"{tag}-*-self.A"))
    for group in groups:
        base = d / f"{tag}-{group}-self"
        order = torch.from_numpy(np.fromfile(base.with_suffix(".order"), dtype=np.uint32).astype(np.int64)).to(dev)
        b = torch.from_numpy(np.fromfile(base.with_suffix(".b"), dtype=np.float32)).to(dev)
        k = len(b)
        a = torch.from_numpy(np.fromfile(base.with_suffix(".A"), dtype=np.float32).reshape(k, dim)).to(dev)
        m = math.ceil(v / k)
        pad = torch.full((k * m,), -1, dtype=torch.long, device=dev)
        pad[: len(order)] = order
        valid = pad >= 0
        kl = 0.0
        for s in range(0, len(h), 256):
            lg = (h[s : s + 256] @ w.T).float()
            if cap:
                lg = cap * torch.tanh(lg / cap)
            g = lg[:, pad.clamp(min=0)].masked_fill(~valid, float("-inf")).view(-1, k, m)
            logp = torch.logsumexp(g, 2) - torch.logsumexp(lg, 1, keepdim=True)
            logq = torch.log_softmax(h[s : s + 256].float() @ a.T + b, 1)
            kl += (logp.exp() * (logp - logq)).sum().item()
        line = {"model": tag, "group": group, "classes": k, "kl_bpc": round(kl / math.log(2) / chars, 4)}
        print(json.dumps(line), flush=True)
        with open(root / "results/retro-kl.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
