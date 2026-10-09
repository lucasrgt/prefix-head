import sys
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

r = GGUFReader(sys.argv[1])
for k, f in r.fields.items():
    if "softcap" in k or "logit" in k or "scale" in k:
        print(k, f.parts[-1])
t = {t.name: t for t in r.tensors}
w = t.get("output.weight") or t["token_embd.weight"]
m = dequantize(w.data, w.tensor_type).astype(np.float16)
print(w.name, m.shape)
m.tofile(sys.argv[2])
