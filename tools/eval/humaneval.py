"""HumanEval pass@1 for a GGUF chat model served by llama.cpp, plus tokens/s and peak RAM.

Each problem is sent once at temperature 0. The reply's Python code block is used as the solution
(or prompt + reply when the reply is only a function body) and run against the benchmark's tests
inside bubblewrap: no network, read-only filesystem, a throwaway /tmp, 10 s limit.

Usage: python3 tools/eval/humaneval.py <model.gguf> [--threads 8] [--limit N] [--tag name] [--gem <gem complete flags>]
"""
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[2]
SERVER = root / ".tools/src/llama.cpp/build/bin/llama-server"
PORT = 8791


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def chat(prompt, max_tokens=512):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=300))
    return out["choices"][0]["message"]["content"], out.get("timings", {}), out.get("usage", {})


def solution(problem, reply):
    blocks = re.findall(r"```(?:python)?\n(.*?)```", reply, re.S)
    code = max(blocks, key=len) if blocks else reply
    if f"def {problem['entry_point']}" in code:
        return code
    return problem["prompt"] + code


def passes(problem, code):
    program = code + "\n\n" + problem["test"] + f"\n\ncheck({problem['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "s.py").write_text(program)
        cmd = ["bwrap", "--ro-bind", "/", "/", "--tmpfs", "/tmp", "--bind", d, d, "--chdir", d, "--unshare-net", "--die-with-parent", "python3", "s.py"]
        try:
            return subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0
        except subprocess.TimeoutExpired:
            return False


def rss_mb(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM"):
            return int(line.split()[1]) / 1024
    return 0.0


CHATML = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


def instruction(p):
    return "Complete the following Python function. Reply with the full function in a single Python code block.\n\n```python\n" + p["prompt"] + "```"


def run_gem(model, problems, tag, threads):
    """Our runtime: `gem complete` over a folder of ChatML prompts; extra flags follow `--gem`."""
    extra = sys.argv[sys.argv.index("--gem") + 1 :]
    with tempfile.TemporaryDirectory() as d:
        for i, p in enumerate(problems):
            (Path(d) / f"{i:04d}.txt").write_text(CHATML.format(instruction(p)))
        cmd = [str(root / "target/release/gem"), "complete", "--model", str(model), "--prompts", d, "--threads", threads] + extra
        stats = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
        ok = sum(passes(p, solution(p, (Path(d) / f"{i:04d}.out").read_text(errors="replace"))) for i, p in enumerate(problems))
    line = {"model": tag, "file": model.name, "problems": len(problems), "pass_at_1": round(ok / len(problems), 4), "tok_per_s": stats["tok_per_s"], "peak_rss_mb": stats["peak_rss_mb"], "threads": int(threads), "runtime": "gem", "head": stats["head"], "classes_tried": stats["classes_tried"], "ctx": stats["ctx"]}
    print(json.dumps(line))
    with open(root / "results/humaneval.jsonl", "a") as f:
        f.write(json.dumps(line) + "\n")


def main():
    model = Path(sys.argv[1])
    threads = arg("--threads", "8")
    limit = int(arg("--limit", "164"))
    tag = arg("--tag", model.stem)
    problems = [json.loads(l) for l in (root / "data/humaneval/HumanEval.jsonl").read_text().splitlines()][:limit]
    if "--gem" in sys.argv:
        return run_gem(model, problems, tag, threads)
    server = subprocess.Popen([str(SERVER), "-m", str(model), "-t", threads, "-c", "4096", "--port", str(PORT), "-np", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        ok = toks = secs = 0
        for i, p in enumerate(problems):
            reply, timings, usage = chat(instruction(p))
            toks += timings.get("predicted_n", usage.get("completion_tokens", 0))
            secs += timings.get("predicted_ms", 0) / 1000
            ok += passes(p, solution(p, reply))
            print(f"\r{i + 1}/{len(problems)} pass {ok}", end="", file=sys.stderr, flush=True)
        line = {"model": tag, "file": model.name, "problems": len(problems), "pass_at_1": round(ok / len(problems), 4), "tok_per_s": round(toks / secs, 1) if secs else None, "peak_rss_mb": round(rss_mb(server.pid)), "threads": int(threads), "runtime": "llama.cpp"}
        print("\n" + json.dumps(line))
        with open(root / "results/humaneval.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
