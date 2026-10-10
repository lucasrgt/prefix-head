"""HumanEval pass@1 for a GGUF chat model served by llama.cpp, plus tokens/s and peak RAM.

Each problem is sent once at temperature 0. The reply's Python code block is used as the solution
(or prompt + reply when the reply is only a function body) and run against the benchmark's tests
inside bubblewrap: no network, read-only filesystem, a throwaway /tmp, 10 s limit.

Usage: python3 tools/eval/humaneval.py <model.gguf> [--bench humaneval|mbpp] [--threads 8] [--limit N] [--tag name]
       [--ctx 4096] [--no-mmap] [--gem <gem complete flags>]

`--bench mbpp` uses MBPP problems 874-973 (never used to train the class predictor).
"""
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[2]
SERVER = root / ".tools/src/llama.cpp/build/bin/llama-server"
PORT = 8791


def arg(name, default):
    own = sys.argv[: sys.argv.index("--gem")] if "--gem" in sys.argv else sys.argv
    return own[own.index(name) + 1] if name in own else default


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


def lenient(problem, reply):
    """Like `solution`, but when the reply has no fenced block it keeps the code-looking part: from
    the first def/import/class line, dedented, minus trailing lines that do not parse."""
    if re.search(r"```(?:python)?\n.*?```", reply, re.S):
        return solution(problem, reply)
    lines = reply.splitlines()
    start = next((i for i, l in enumerate(lines) if re.match(r"\s*(def |import |from |class |@)", l)), None)
    if start is None:
        return solution(problem, reply)
    body = textwrap.dedent("\n".join(lines[start:])).splitlines()
    while body:
        try:
            ast.parse("\n".join(body))
            break
        except SyntaxError:
            body.pop()
    code = "\n".join(body)
    return code if f"def {problem['entry_point']}" in code else problem["prompt"] + code


def passes(problem, code):
    program = code + "\n\n" + problem["tail"]
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "s.py").write_text(program)
        cmd = ["bwrap", "--ro-bind", "/", "/", "--tmpfs", "/tmp", "--bind", d, d, "--chdir", d, "--unshare-net", "--die-with-parent", "python3", "s.py"]
        try:
            return subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0
        except subprocess.TimeoutExpired:
            return False


def status_mb(pid, key):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(key):
            return round(int(line.split()[1]) / 1024)
    return 0


def load(bench, limit):
    """Problems as dicts with `task_id`, `ask` (the user message), `prompt` (prefix used when the
    reply is only a function body), `entry_point` and `tail` (appended to the solution and run)."""
    if bench == "mbpp":
        rows = [json.loads(l) for l in (root / "data/mbpp/mbpp.jsonl").read_text().splitlines()][874:974][:limit]
        out = []
        for r in rows:
            tests = "\n".join(r["test_list"])
            name = re.search(r"assert\s+\(*\s*(\w+)\(", r["test_list"][0]).group(1)
            ask = r["text"].strip() + " Your code should pass these tests:\n\n" + tests + "\n\nReply with the full function in a single Python code block."
            out.append({"task_id": f"mbpp/{r['task_id']}", "ask": ask, "prompt": "", "entry_point": name, "tail": r.get("test_setup_code", "") + "\n" + tests + "\n"})
        return out
    rows = [json.loads(l) for l in (root / "data/humaneval/HumanEval.jsonl").read_text().splitlines()][:limit]
    for p in rows:
        p["ask"] = "Complete the following Python function. Reply with the full function in a single Python code block.\n\n```python\n" + p["prompt"] + "```"
        p["tail"] = p["test"] + f"\n\ncheck({p['entry_point']})\n"
    return rows


CHATML = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


def run_gem(model, problems, tag, threads, bench):
    """Our runtime: `gem complete` over a folder of ChatML prompts; extra flags follow `--gem`."""
    extra = sys.argv[sys.argv.index("--gem") + 1 :]
    with tempfile.TemporaryDirectory() as d:
        for i, p in enumerate(problems):
            (Path(d) / f"{i:04d}.txt").write_text(CHATML.format(p["ask"]))
        cmd = [str(root / "target/release/gem"), "complete", "--model", str(model), "--prompts", d, "--threads", threads] + extra
        stats = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
        replies = [(Path(d) / f"{i:04d}.out").read_text(errors="replace") for i in range(len(problems))]
    keep = root / "results/humaneval-out"
    keep.mkdir(exist_ok=True)
    (keep / f"{tag}.jsonl").write_text("".join(json.dumps({"task": p["task_id"], "reply": r}) + "\n" for p, r in zip(problems, replies)))
    ok = sum(passes(p, solution(p, r)) for p, r in zip(problems, replies))
    ok_lenient = sum(passes(p, lenient(p, r)) for p, r in zip(problems, replies))
    fenced = sum("```" in r for r in replies)
    line = {"model": tag, "bench": bench, "file": model.name, "problems": len(problems), "pass_at_1": round(ok / len(problems), 4), "pass_lenient": round(ok_lenient / len(problems), 4), "fenced_replies": fenced, "tok_per_s": stats["tok_per_s"], "peak_rss_mb": stats["peak_rss_mb"], "rss_anon_mb": stats.get("rss_anon_mb"), "rss_file_mb": stats.get("rss_file_mb"), "threads": int(threads), "runtime": "gem", "head": stats["head"], "classes_tried": stats["classes_tried"], "fallback": stats.get("fallback", 0), "fallback_share": stats.get("fallback_share", 0), "ctx": stats["ctx"]}
    print(json.dumps(line))
    with open(root / "results/humaneval.jsonl", "a") as f:
        f.write(json.dumps(line) + "\n")


def main():
    model = Path(sys.argv[1])
    threads = arg("--threads", "8")
    limit = int(arg("--limit", "164"))
    tag = arg("--tag", model.stem)
    bench = arg("--bench", "humaneval")
    ctx = arg("--ctx", "4096")
    own = sys.argv[: sys.argv.index("--gem")] if "--gem" in sys.argv else sys.argv
    problems = load(bench, limit)
    if "--gem" in sys.argv:
        return run_gem(model, problems, tag, threads, bench)
    server = subprocess.Popen([str(SERVER), "-m", str(model), "-t", threads, "-c", ctx, "--port", str(PORT), "-np", "1"] + (["--no-mmap"] if "--no-mmap" in own else []) + os.environ.get("LLAMA_EXTRA", "").split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        ok = toks = secs = 0
        for i, p in enumerate(problems):
            reply, timings, usage = chat(p["ask"])
            toks += timings.get("predicted_n", usage.get("completion_tokens", 0))
            secs += timings.get("predicted_ms", 0) / 1000
            ok += passes(p, solution(p, reply))
            print(f"\r{i + 1}/{len(problems)} pass {ok}", end="", file=sys.stderr, flush=True)
        line = {"model": tag, "bench": bench, "file": model.name, "problems": len(problems), "pass_at_1": round(ok / len(problems), 4), "tok_per_s": round(toks / secs, 1) if secs else None, "peak_rss_mb": status_mb(server.pid, "VmHWM"), "rss_anon_mb": status_mb(server.pid, "RssAnon"), "rss_file_mb": status_mb(server.pid, "RssFile"), "threads": int(threads), "ctx": int(ctx), "mmap": "--no-mmap" not in own, "server_args": os.environ.get("LLAMA_EXTRA", ""), "runtime": "llama.cpp"}
        print("\n" + json.dumps(line))
        with open(root / "results/humaneval.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
