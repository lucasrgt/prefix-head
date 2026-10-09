"""Variations on prefix-ordered classes, all measured as KL(full || factored) in bits/char on held-out
text, with class predictors trained to reproduce the frozen model's own class probabilities.

  sweep     prefix blocks of size m (cost/fidelity curve), random blocks as reference
  levels    3-level prefix tree: super-class, class, token (P = P(s|h) P(c|s,h) P(t|c,h))
  refine    start from prefix blocks; move tokens to the class the trained predictor already favours
            in their contexts (capacity-limited), retrain, repeat
  hybrid    word-start tokens ordered by semantic k-means or frequency, continuations by prefix;
            prefix ignoring the leading space; suffix order

Usage: .venv-gpu/bin/python tools/gpu/explore.py <tag> <experiment> [...]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import retro  # noqa: E402

root = retro.root
dev = retro.dev
LN2 = math.log(2)


def kl_two_level(h, w, g, a, b, chunk=256):
    kl = 0.0
    for s in range(0, len(h), chunk):
        lg = retro.cap((h[s : s + chunk] @ w.T).float())
        logp = g.class_lse(lg) - torch.logsumexp(lg, 1, keepdim=True)
        logq = torch.log_softmax(h[s : s + chunk].float() @ a.T + b, 1)
        kl += (logp.exp() * (logp - logq)).sum().item()
    return kl / LN2


class Run:
    def __init__(self, tag):
        self.tag = tag
        d = root / "data/retro"
        cap = d / f"{tag}.softcap"
        retro.CAP = float(cap.read_text()) if cap.exists() else None
        self.htr, self.ytr, _ = retro.load(d / f"{tag}-train", on_gpu=False)
        self.hte, _, meta = retro.load(d / f"{tag}-test")
        self.v, self.d, self.chars = meta["vocab"], meta["d"], meta["chars"]
        self.w = torch.from_numpy(np.fromfile(d / f"{tag}.W", dtype=np.float16).reshape(self.v, self.d)).to(dev)
        self.pcs = retro.pieces(d / f"{tag}-test.vocab")
        self.sqrt = math.ceil(math.sqrt(self.v))

    def emit(self, exp, variant, m, classes, kl_bits, cost):
        line = {"model": self.tag, "exp": exp, "variant": variant, "m": m, "classes": classes, "kl_bpc": round(kl_bits / self.chars, 4), "head_rows_read": cost, "head_cut": round(self.v / cost, 1)}
        print(json.dumps(line), flush=True)
        with open(root / "results/explore.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")

    def two_level(self, order, m):
        g = retro.Groups(order, self.v, m)
        a, b = retro.train_self(self.htr, self.w, g, 2.0, 1e-3)
        return g, a, b, kl_two_level(self.hte, self.w, g, a, b)

    def prefix_order(self, key=None):
        key = key or (lambda t: self.pcs[t])
        return np.array(sorted(range(self.v), key=lambda t: (key(t), t)))

    def sweep(self):
        for m in [64, 128, 256, self.sqrt, 1024, 2048, 4096]:
            for name, order in [("prefix", self.prefix_order()), ("random", np.random.default_rng(7).permutation(self.v))]:
                if name == "random" and m not in (128, self.sqrt, 2048):
                    continue
                g, _, _, kl = self.two_level(order, m)
                self.emit("sweep", name, m, g.k, kl, g.k + m)

    def levels(self):
        for s, c in self.level_shapes():
            for name, order in [("prefix", self.prefix_order()), ("random", np.random.default_rng(7).permutation(self.v))]:
                kl, cost = self.three_level(order, s, c)
                self.emit("levels", name, self.v // (s * c) + (self.v % (s * c) > 0), s * c, kl, cost)

    def level_shapes(self):
        side = round(self.v ** (1 / 3))
        return [(side, side), (side // 2, side * 2)]

    def three_level(self, order, ns, nc):
        m = math.ceil(self.v / (ns * nc))
        g = retro.Groups(order, self.v, m, ns * nc)
        k = g.k
        sup = torch.arange(k, device=dev) // nc
        cent = torch.nan_to_num(retro.centroid(self.w, g))
        a1 = torch.stack([cent[sup == s].mean(0) for s in range(ns)]).clone().requires_grad_()
        b1 = torch.zeros(ns, device=dev, requires_grad=True)
        a2 = cent.clone().requires_grad_()
        b2 = torch.zeros(k, device=dev, requires_grad=True)
        opt = torch.optim.Adam([a1, b1, a2, b2], lr=1e-3)
        batch, epochs = 512, 2.0
        steps = int(epochs * len(self.htr) / batch)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / steps)
        perm = torch.randperm(len(self.htr), generator=torch.Generator().manual_seed(1))
        for step in range(steps):
            o = (step * batch) % len(self.htr)
            if o + batch > len(self.htr):
                continue
            hb = self.htr[perm[o : o + batch]].to(dev)
            with torch.no_grad():
                pc = torch.softmax(g.class_lse(retro.cap((hb @ self.w.T).float())), 1)
                ps = pc.view(-1, ns, nc).sum(2)
                pcs = pc.view(-1, ns, nc) / ps.clamp(min=1e-30)[:, :, None]
            hf = hb.float()
            l1 = torch.log_softmax(hf @ a1.T + b1, 1)
            l2 = torch.log_softmax((hf @ a2.T + b2).view(-1, ns, nc), 2)
            loss = -(ps * l1).sum(1).mean() - (ps[:, :, None] * pcs * l2).sum((1, 2)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        kl = 0.0
        with torch.no_grad():
            for s in range(0, len(self.hte), 256):
                hb = self.hte[s : s + 256]
                lg = retro.cap((hb @ self.w.T).float())
                logp = g.class_lse(lg) - torch.logsumexp(lg, 1, keepdim=True)
                hf = hb.float()
                logq = (torch.log_softmax(hf @ a1.T + b1, 1)[:, :, None] + torch.log_softmax((hf @ a2.T + b2).view(-1, ns, nc), 2)).view(-1, k)
                kl += torch.where(logp > -1e30, logp.exp() * (logp - logq), 0.0).sum().item()
        return kl / LN2, ns + nc + m

    def refine(self, rounds=3):
        m = self.sqrt
        order = self.prefix_order()
        for r in range(rounds + 1):
            g, a, b, kl = self.two_level(order, m)
            self.emit("refine", f"round{r}", m, g.k, kl, g.k + m)
            if r == rounds:
                break
            order = self.reassign(g, a, b, m)

    def reassign(self, g, a, b, m):
        k = g.k
        score = torch.zeros(self.v, k, device=dev)
        with torch.no_grad():
            for s in range(0, len(self.htr), 512):
                hb = self.htr[s : s + 512].to(dev)
                p = torch.softmax(retro.cap((hb @ self.w.T).float()), 1)
                logq = torch.log_softmax(hb.float() @ a.T + b, 1)
                score += p.T @ logq
        top = 16
        val, cand = score.topk(top, dim=1)
        val, cand = val.cpu().numpy(), cand.cpu().numpy()
        current = g.cls.cpu().numpy()
        order_pairs = np.argsort(-val.ravel(), kind="stable")
        cls = np.full(self.v, -1, dtype=np.int64)
        fill = np.zeros(k, dtype=np.int64)
        tok = np.repeat(np.arange(self.v), top)
        flat = cand.ravel()
        for i in order_pairs:
            t, c = tok[i], flat[i]
            if cls[t] < 0 and fill[c] < m:
                cls[t] = c
                fill[c] += 1
        for t in np.nonzero(cls < 0)[0]:
            c = current[t] if fill[current[t]] < m else int(np.argmin(fill))
            cls[t] = c
            fill[c] += 1
        moved = int((cls != current).sum())
        print(f"  reassign: {moved} tokens moved", file=sys.stderr, flush=True)
        within = {c: [] for c in range(k)}
        pos = {t: i for i, t in enumerate(self.prefix_order())}
        for t in range(self.v):
            within[cls[t]].append(t)
        return np.array([t for c in range(k) for t in sorted(within[c], key=lambda t: pos[t])])

    def hybrid(self):
        m = self.sqrt
        start = lambda t: self.pcs[t][:1] in (b" ", b"\n") or self.pcs[t].startswith("▁".encode())
        starts = [t for t in range(self.v) if start(t)]
        conts = [t for t in range(self.v) if not start(t)]
        prefix = self.prefix_order()
        rank = {t: i for i, t in enumerate(prefix)}
        conts_prefix = sorted(conts, key=lambda t: rank[t])
        sem = retro.semantic(self.w, m, 7)
        srank = {t: i for i, t in enumerate(sem)}
        freq = np.bincount(self.ytr.numpy(), minlength=self.v)
        variants = {
            "starts-semantic+conts-prefix": sorted(starts, key=lambda t: srank[t]) + conts_prefix,
            "starts-frequency+conts-prefix": sorted(starts, key=lambda t: (-freq[t], t)) + conts_prefix,
            "prefix-ignoring-space": self.prefix_order(lambda t: self.pcs[t].replace("▁".encode(), b"").lstrip(b" ")),
            "suffix": self.prefix_order(lambda t: self.pcs[t][::-1]),
        }
        print(f"  {len(starts)} word-start tokens, {len(conts)} continuations", file=sys.stderr, flush=True)
        for name, order in variants.items():
            g, _, _, kl = self.two_level(np.array(order), m)
            self.emit("hybrid", name, m, g.k, kl, g.k + m)


def main():
    run = Run(sys.argv[1])
    for exp in sys.argv[2:]:
        getattr(run, exp)()


if __name__ == "__main__":
    main()
