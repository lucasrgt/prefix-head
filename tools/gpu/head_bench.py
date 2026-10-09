"""Times a full softmax head vs a two-level head (class, then token inside the class block) on the GPU,
fp16, under CUDA graphs so launch overhead does not decide the result. "score" = log-prob of a known
next token (training, evaluation, prefill); "sample" = draw the next token (generation).
Usage: .venv-gpu/bin/python tools/gpu/head_bench.py"""
import math
import torch

dev = "cuda"
torch.manual_seed(0)


def gumbel_argmax(logits):
    u = torch.rand_like(logits, dtype=torch.float32).clamp_(1e-20, 1.0)
    return (logits.float() - torch.log(-torch.log(u))).argmax(-1)


def timed(fn, iters=200):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(10):
        g.replay()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000


print(f"{'vocab':>7} {'d':>5} {'batch':>6} {'mode':>6} {'full us':>9} {'2-level us':>11} {'speedup':>8} {'full logits MB':>15} {'2-level MB':>11}")
for v, d, name in [(32768, 512, "ours-32k"), (128256, 2048, "llama3.2-1b"), (151936, 1024, "qwen3-0.6b"), (262144, 640, "gemma3-270m")]:
    m = math.ceil(math.sqrt(v))
    c = math.ceil(v / m)
    w = torch.randn(c * m, d, device=dev, dtype=torch.half) / math.sqrt(d)
    blocks = w.view(c, m, d)
    a = torch.randn(c, d, device=dev, dtype=torch.half) / math.sqrt(d)
    for batch in [1, 8, 64, 512, 4096]:
        if batch * v * 4 * 3 > 2.5 * 2**30:
            continue
        h = torch.randn(batch, d, device=dev, dtype=torch.half)
        y = torch.randint(0, v, (batch,), device=dev)
        out = torch.empty(batch, dtype=torch.long, device=dev)
        lp = torch.empty(batch, dtype=torch.float32, device=dev)

        def full_sample():
            out.copy_(gumbel_argmax(h @ w.T))

        def two_sample():
            cls = gumbel_argmax(h @ a.T)
            inner = torch.bmm(blocks[cls], h.unsqueeze(-1)).squeeze(-1)
            out.copy_(cls * m + gumbel_argmax(inner))

        def full_score():
            lg = (h @ w.T).float()
            lp.copy_(lg.gather(1, y[:, None])[:, 0] - torch.logsumexp(lg, 1))

        def two_score():
            cls, pos = y // m, y % m
            lc = (h @ a.T).float()
            inner = torch.bmm(blocks[cls], h.unsqueeze(-1)).squeeze(-1).float()
            lp.copy_(lc.gather(1, cls[:, None])[:, 0] - torch.logsumexp(lc, 1) + inner.gather(1, pos[:, None])[:, 0] - torch.logsumexp(inner, 1))

        for mode, f, t in [("score", full_score, two_score), ("sample", full_sample, two_sample)]:
            tf, tt = timed(f), timed(t)
            print(f"{v:>7} {d:>5} {batch:>6} {mode:>6} {tf:>9.1f} {tt:>11.1f} {tf / tt:>7.1f}x {batch * v * 2 / 2**20:>15.1f} {batch * (c + m) * 2 / 2**20:>11.2f}  {name}", flush=True)
        del h
    del w, blocks, a
    torch.cuda.empty_cache()
