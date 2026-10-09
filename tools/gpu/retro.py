"""Two-level head retrofit of a frozen LLM, on the GPU, with the domain-adaptation confound removed.

Reads the dumps written by .tools/hidden (final-norm hidden states + next token) and tools/hidden/head.py
(output rows). For each grouping it reports, in bits/char of the test text:

  full          original head, untouched
  self          class predictor trained to reproduce the model's own class probabilities on the train
                text (no labels, no other model); P(token | class) = original rows restricted to the class
                -> the pure cost of factoring, no adaptation
  adapt-full    original full head fine-tuned on the train text (true next tokens)
  adapt-2lvl    class predictor + per-class blocks fine-tuned on the train text, same epochs and lr
                -> equal-budget adaptation comparison

Usage: .venv-gpu/bin/python tools/gpu/retro.py <tag> [groupings...]
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parents[2]
dev = "cuda"
LN2 = math.log(2)
CAP = None
BATCH = 256
SEED = 1


def cap(x):
    return x if CAP is None else CAP * torch.tanh(x / CAP)


def load(prefix, on_gpu=True):
    meta = json.loads(prefix.with_suffix(".json").read_text())
    d = meta["d"]
    h = torch.from_numpy(np.fromfile(prefix.with_suffix(".h"), dtype=np.float16).reshape(-1, d))
    y = torch.from_numpy(np.fromfile(prefix.with_suffix(".y"), dtype=np.uint32).astype(np.int64))
    return (h.to(dev), y.to(dev), meta) if on_gpu else (h.pin_memory(), y, meta)


def pieces(path):
    b = path.read_bytes()
    out, i = [], 0
    while i < len(b):
        n = b[i]
        out.append(b[i + 1 : i + 1 + n])
        i += 1 + n
    return out


def semantic(w, m, seed):
    v, d = w.shape
    k = math.ceil(v / m)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = w
    cent = x[torch.randperm(v, generator=g)[:k].to(dev)].float()
    for _ in range(15):
        assign = torch.empty(v, dtype=torch.long, device=dev)
        for s in range(0, v, 16384):
            dist = (cent * cent).sum(1)[None] - 2 * x[s : s + 16384].float() @ cent.T
            assign[s : s + 16384] = dist.argmin(1)
        sums = torch.zeros(k, d, device=dev)
        for s in range(0, v, 16384):
            sums.index_add_(0, assign[s : s + 16384], x[s : s + 16384].float())
        n = torch.bincount(assign, minlength=k).float()
        empty = n == 0
        cent = torch.where(empty[:, None], x[torch.randint(0, v, (k,), generator=g).to(dev)].float(), sums / n.clamp(min=1)[:, None])
    top = min(16, k)
    dists, cands = [], []
    for s in range(0, v, 16384):
        dist = (cent * cent).sum(1)[None] - 2 * x[s : s + 16384].float() @ cent.T
        dv, ci = dist.topk(top, dim=1, largest=False)
        dists.append(dv.cpu())
        cands.append(ci.cpu())
    dist, cand = torch.cat(dists).numpy(), torch.cat(cands).numpy()
    tok = np.repeat(np.arange(v), top)
    order = np.argsort(dist.ravel(), kind="stable")
    cls = np.full(v, -1, dtype=np.int64)
    fill = np.zeros(k, dtype=np.int64)
    flat_c = cand.ravel()
    for i in order:
        t, c = tok[i], flat_c[i]
        if cls[t] < 0 and fill[c] < m:
            cls[t] = c
            fill[c] += 1
    free = [c for c in range(k) for _ in range(m - fill[c])]
    for t in np.nonzero(cls < 0)[0]:
        cls[t] = free.pop()
    return np.argsort(cls, kind="stable")


def ordering(kind, w, pcs, train_y, m, seed):
    v = w.shape[0]
    if kind == "prefix":
        return np.array(sorted(range(v), key=lambda t: (pcs[t], t)))
    if kind == "freq":
        n = np.bincount(train_y.cpu().numpy(), minlength=v)
        return np.lexsort((np.arange(v), -n))
    if kind == "id":
        return np.arange(v)
    if kind == "random":
        return np.random.default_rng(seed).permutation(v)
    if kind == "sem":
        return semantic(w, m, seed)
    if kind.startswith("cipher"):
        perm = np.random.default_rng(int(kind[6:] or 1)).permutation(256).astype(np.uint8)
        return np.array(sorted(range(v), key=lambda t: (bytes(perm[list(pcs[t])]), t)))
    if kind == "hyb":
        n = np.bincount(train_y.cpu().numpy(), minlength=v)
        start = [t for t in range(v) if pcs[t][:1] in (b" ", b"\n") or pcs[t].startswith("▁".encode())]
        rest = sorted(set(range(v)) - set(start), key=lambda t: (pcs[t], t))
        return np.array(sorted(start, key=lambda t: (-n[t], t)) + rest)
    if kind == "g4":
        return np.fromfile(root / "data/models/g4/e4b-assistant.order", dtype=np.uint32).astype(np.int64)
    raise SystemExit("grouping: prefix|freq|id|random|sem|hyb|g4, optionally @<block size>")


class Groups:
    def __init__(self, order, v, m, k=None):
        self.k = k or math.ceil(v / m)
        self.m = m
        pad = np.full(self.k * m, -1, dtype=np.int64)
        pad[: len(order)] = order
        self.pad = torch.from_numpy(pad).to(dev)
        self.valid = self.pad >= 0
        self.cls = torch.empty(v, dtype=torch.long, device=dev)
        self.pos = torch.empty(v, dtype=torch.long, device=dev)
        idx = torch.arange(len(order), device=dev)
        o = torch.from_numpy(np.asarray(order, dtype=np.int64)).to(dev)
        self.cls[o] = idx // m
        self.pos[o] = idx % m

    def class_lse(self, logits):
        g = logits[:, self.pad.clamp(min=0)].masked_fill(~self.valid, float("-inf"))
        return torch.logsumexp(g.view(-1, self.k, self.m), 2)

    def blocks(self, w):
        b = w[self.pad.clamp(min=0)] * self.valid[:, None]
        return b.view(self.k, self.m, -1)


def full_pass(h, y, w, g, chunk=512):
    """Bits of the full head, plus the exact within-class bits under grouping g."""
    full = within = 0.0
    for s in range(0, len(y), chunk):
        lg = cap((h[s : s + chunk] @ w.T).float())
        yy = y[s : s + chunk]
        tgt = lg.gather(1, yy[:, None])[:, 0]
        full += (torch.logsumexp(lg, 1) - tgt).sum().item()
        cl = g.class_lse(lg)
        within += (cl.gather(1, g.cls[yy][:, None])[:, 0] - tgt).sum().item()
    return full / LN2, within / LN2


def class_bits(h, y, a, b, g, chunk=4096):
    tot = 0.0
    for s in range(0, len(y), chunk):
        lc = (h[s : s + chunk].float() @ a.T + b)
        tot += (torch.logsumexp(lc, 1) - lc.gather(1, g.cls[y[s : s + chunk]][:, None])[:, 0]).sum().item()
    return tot / LN2


def centroid(w, g):
    out = torch.empty(g.k, w.shape[1], device=dev)
    idx, val = g.pad.view(g.k, g.m), g.valid.view(g.k, g.m)
    for s in range(0, g.k, 128):
        blk = w[idx[s : s + 128].clamp(min=0)].float() * val[s : s + 128, :, None]
        out[s : s + 128] = blk.sum(1) / val[s : s + 128].sum(1, keepdim=True)
    return out


def train_self(hx, w, g, epochs, lr, batch=512):
    a = centroid(w, g).clone().requires_grad_()
    b = torch.zeros(g.k, device=dev, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=lr)
    steps = int(epochs * len(hx) / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / steps)
    gen = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(hx), generator=gen)
    for step in range(steps):
        i = perm[(step * batch) % len(hx) : (step * batch) % len(hx) + batch]
        if len(i) < batch:
            perm = torch.randperm(len(hx), generator=gen)
            continue
        hb = hx[i].to(dev, non_blocking=True)
        with torch.no_grad():
            target = torch.softmax(g.class_lse(cap((hb @ w.T).float())), 1)
        lc = hb.float() @ a.T + b
        loss = -(target * torch.log_softmax(lc, 1)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return a.detach(), b.detach()


def train_adapt_full(hx, yx, w, epochs, lr, batch=None):
    batch = batch or BATCH
    wt = w.float().clone().requires_grad_()
    opt = torch.optim.Adam([wt], lr=lr)
    steps = int(epochs * len(hx) / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / steps)
    gen = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(hx), generator=gen)
    for step in range(steps):
        o = (step * batch) % len(hx)
        if o + batch > len(hx):
            perm = torch.randperm(len(hx), generator=gen)
            continue
        i = perm[o : o + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = hx[i].to(dev, non_blocking=True) @ wt.T
        loss = torch.nn.functional.cross_entropy(cap(lg.float()), yx[i].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return wt.detach().half()


def train_adapt_2lvl(hx, yx, w, g, epochs, lr, lr_class, batch=None):
    batch = batch or BATCH
    a = centroid(w, g).clone().requires_grad_()
    b = torch.zeros(g.k, device=dev, requires_grad=True)
    blk = g.blocks(w.float()).clone().requires_grad_()
    mask = g.valid.view(g.k, g.m)
    opt = torch.optim.Adam([{"params": [a, b], "lr": lr_class}, {"params": [blk], "lr": lr}])
    steps = int(epochs * len(hx) / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / steps)
    gen = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(hx), generator=gen)
    for step in range(steps):
        o = (step * batch) % len(hx)
        if o + batch > len(hx):
            perm = torch.randperm(len(hx), generator=gen)
            continue
        i = perm[o : o + batch]
        hb, yb = hx[i].to(dev, non_blocking=True).float(), yx[i].to(dev)
        c, p = g.cls[yb], g.pos[yb]
        lc = hb @ a.T + b
        inner = cap(torch.bmm(blk[c], hb[:, :, None])[:, :, 0]).masked_fill(~mask[c], float("-inf"))
        loss = torch.nn.functional.cross_entropy(lc, c) + torch.nn.functional.cross_entropy(inner, p)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return a.detach(), b.detach(), blk.detach()


def eval_2lvl(h, y, a, b, blk, g, chunk=512):
    mask = g.valid.view(g.k, g.m)
    cls = wit = 0.0
    for s in range(0, len(y), chunk):
        hb, yb = h[s : s + chunk].float(), y[s : s + chunk]
        c, p = g.cls[yb], g.pos[yb]
        lc = hb @ a.T + b
        cls += (torch.logsumexp(lc, 1) - lc.gather(1, c[:, None])[:, 0]).sum().item()
        inner = cap(torch.bmm(blk[c], hb[:, :, None])[:, :, 0]).masked_fill(~mask[c], float("-inf"))
        wit += (torch.logsumexp(inner, 1) - inner.gather(1, p[:, None])[:, 0]).sum().item()
    return cls / LN2, wit / LN2


def main():
    tag = sys.argv[1]
    kinds = sys.argv[2:] or ["prefix", "freq", "id", "random", "sem"]
    d = root / "data/retro"
    global CAP, BATCH, SEED
    if (d / f"{tag}.softcap").exists():
        CAP = float((d / f"{tag}.softcap").read_text())
    BATCH = int(os.environ.get("RETRO_BATCH", "256"))
    seed = SEED = int(os.environ.get("RETRO_SEED", "1"))
    suffix = "" if seed == 1 else f"-s{seed}"
    htr, ytr, _ = load(d / f"{tag}-train", on_gpu=False)
    hte, yte, meta = load(d / f"{tag}-test")
    v, dim, chars = meta["vocab"], meta["d"], meta["chars"]
    w = torch.from_numpy(np.fromfile(d / f"{tag}.W", dtype=np.float16).reshape(v, dim)).to(dev)
    pcs = pieces(d / f"{tag}-test.vocab")
    m = math.ceil(math.sqrt(v))
    epochs, lr_self, lr_adapt = 2.0, 1e-3, 1e-4

    t0 = time.time()
    cache = d / f"{tag}-adapt-full.txt"
    adapt = os.environ.get("RETRO_NO_ADAPT") != "1"
    if not adapt:
        adapt_full = float("nan")
    elif cache.exists():
        adapt_full = float(cache.read_text())
    else:
        wa = train_adapt_full(htr, ytr, w, epochs, lr_adapt)
        adapt_full = full_pass(hte, yte, wa, Groups(np.arange(v), v, m))[0] / chars
        cache.write_text(f"{adapt_full}\n")
        del wa
        torch.cuda.empty_cache()
    print(f"{tag}: adapt-full {adapt_full:.4f} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)

    for spec in kinds:
        t0 = time.time()
        kind, _, size = spec.partition("@")
        m = int(size) if size else math.ceil(math.sqrt(v))
        g = Groups(ordering(kind, w, pcs, ytr, m, 6 + seed), v, m)
        full, within = full_pass(hte, yte, w, g)
        a, b = train_self(htr, w, g, epochs, lr_self)
        self_class = class_bits(hte, yte, a, b, g)
        out = d / f"{tag}-{spec.replace('@', '-')}{suffix}-self"
        order = g.pad[g.valid].cpu().numpy().astype(np.uint32)
        order.tofile(out.with_suffix(".order"))
        a.cpu().numpy().astype(np.float32).tofile(out.with_suffix(".A"))
        b.cpu().numpy().astype(np.float32).tofile(out.with_suffix(".b"))
        if adapt:
            a2, b2, blk = train_adapt_2lvl(htr, ytr, w, g, epochs, lr_adapt, lr_self)
            ad_class, ad_within = eval_2lvl(hte, yte, a2, b2, blk, g)
            del a2, b2, blk
        else:
            ad_class = ad_within = float("nan")
        r = {
            "model": tag, "group": spec, "seed": seed, "vocab": v, "d": dim, "classes": g.k, "m": m, "epochs": epochs,
            "train_rows": len(ytr), "test_rows": len(yte), "test_chars": chars,
            "full_bpc": round(full / chars, 4),
            "self_bpc": round((self_class + within) / chars, 4),
            "self_class_bpc": round(self_class / chars, 4),
            "oracle_class_bpc": round((full - within) / chars, 4),
            "within_bpc": round(within / chars, 4),
            "adapt_full_bpc": round(adapt_full, 4),
            "adapt_2lvl_bpc": round((ad_class + ad_within) / chars, 4),
            "adapt_2lvl_class_bpc": round(ad_class / chars, 4),
            "adapt_2lvl_within_bpc": round(ad_within / chars, 4),
            "head_macs_full": v * dim, "head_macs_2lvl": (g.k + m) * dim,
        }
        line = json.dumps(r)
        print(line, flush=True)
        with open(root / "results/retro-gpu.jsonl", "a") as f:
            f.write(line + "\n")
        print(f"  {kind} done in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
