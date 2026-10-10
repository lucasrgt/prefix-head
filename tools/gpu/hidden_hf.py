"""Extracts final-norm hidden states and output rows from a Hugging Face causal LM on the GPU, in the
same dump format as .tools/hidden (llama.cpp) and tools/hidden/head.py, so tools/gpu/retro.py and
kl.py can use it. Used for BitNet b1.58 2B4T, which mainline llama.cpp cannot convert.

Usage: .venv-gpu/bin/python tools/gpu/hidden_hf.py <hf dir> <text> <out prefix> <ctx> <every> [--head <W file>]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main():
    src, text_path, out, ctx, every = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    head_out = sys.argv[sys.argv.index("--head") + 1] if "--head" in sys.argv else None
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16).cuda().eval()
    v = model.config.vocab_size
    dec = byte_decoder()
    pieces = []
    for t in range(v):
        s = tok.convert_ids_to_tokens(t) or ""
        pieces.append(bytes(dec[c] for c in s) if all(c in dec for c in s) else s.encode())
    ids = tok(Path(text_path).read_text(), add_special_tokens=False)["input_ids"]
    body = ctx - 1
    windows = (len(ids) - 1) // body
    bos = tok.bos_token_id
    fh, fy = open(out.with_suffix(".h"), "wb"), open(out.with_suffix(".y"), "wb")
    rows = chars = 0
    with torch.no_grad():
        for w in range(0, windows, every):
            s = w * body
            x = torch.tensor([[bos] + ids[s : s + body]], device="cuda")
            hidden = model.model(input_ids=x).last_hidden_state[0, 1:]
            fh.write(hidden.half().cpu().numpy().tobytes())
            y = np.array(ids[s + 1 : s + body + 1], dtype=np.uint32)
            fy.write(y.tobytes())
            chars += sum(sum(1 for b in pieces[t] if b & 0xC0 != 0x80) for t in y)
            rows += len(y)
            print(f"\rwindow {w + 1}/{windows}", end="", file=sys.stderr, flush=True)
    fh.close()
    fy.close()
    with open(out.with_suffix(".vocab"), "wb") as f:
        for p in pieces:
            p = p[:255]
            f.write(bytes([len(p)]) + p)
    d = model.config.hidden_size
    out.with_suffix(".json").write_text(json.dumps({"model": src, "d": d, "vocab": v, "rows": rows, "chars": chars, "ctx": ctx, "every": every, "tokens": len(ids)}) + "\n")
    if head_out:
        model.get_output_embeddings().weight.detach().half().cpu().numpy().tofile(head_out)
    print(f"\n{rows} rows, {chars} chars, d {d}, vocab {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
