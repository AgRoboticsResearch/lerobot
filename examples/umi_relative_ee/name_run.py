#!/usr/bin/env python
"""Generate a unique run-folder name for the open-loop eval pipeline.

Prints a folder basename:  <slug>_<YYYYmmdd-HHMMSS>
(to be placed under outputs/research_report/).

The slug is preferably LLM-suggested; the datetime always guarantees uniqueness
so repeated runs never clobber each other. Resolution order:
  1. $RUN_NAME env        -> used as-is (cleaned to kebab-case)
  2. $LLM_NAME_CMD env    -> shell command that reads the run context on stdin and prints a slug
  3. `claude -p` CLI      -> best-effort LLM suggestion (timeout-bounded, ~8s)
  4. fallback             -> "open_loop_val_compare"

Usage:
  python name_run.py [optional context words]
  RUN_NAME="my-fixed-name" python name_run.py
"""
import os, re, shutil, subprocess, sys, datetime

FALLBACK = "open_loop_val_compare"


def _clean(s: str) -> str:
    s = (s or "").strip().strip("`'\"")
    s = re.split(r"[\r\n]", s)[0]              # first line only
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s[:48] or FALLBACK


def _context() -> str:
    ds = os.environ.get("DATASET_ROOT", "")
    ds = os.path.basename(ds.rstrip("/")) if ds else "a strawberry-picking validation set"
    return (f"open-loop decoded-metric comparison of UMI relative-EE robot policies "
            f"(ACT, SmolVLA, pi0.5) on dataset '{ds}'")


def _llm_slug(context: str):
    cmd = os.environ.get("LLM_NAME_CMD")
    if cmd:
        try:
            r = subprocess.run(cmd, shell=True, input=context, capture_output=True,
                               text=True, timeout=40)
            if r.returncode == 0 and r.stdout.strip():
                return _clean(r.stdout)
        except Exception:
            return None
    claude = shutil.which("claude")
    if claude:
        prompt = (f"Reply with ONLY a short 2-4 word kebab-case folder slug (no prose, no quotes) "
                  f"describing this ML run: {context}.")
        try:
            r = subprocess.run([claude, "-p", "--output-format", "text", prompt],
                               capture_output=True, text=True, timeout=40)
        except Exception:
            return None
        if r.returncode == 0 and (r.stdout or "").strip():
            return _clean(r.stdout)
    return None


def main():
    ctx = " ".join(sys.argv[1:]) or _context()
    if os.environ.get("RUN_NAME"):
        slug = _clean(os.environ["RUN_NAME"])
    else:
        slug = _llm_slug(ctx) or FALLBACK
    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"{slug}_{dt}")


if __name__ == "__main__":
    main()
