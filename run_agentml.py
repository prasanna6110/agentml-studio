"""One-click launcher: open this file in VS Code and press ▷ Run.

Steps (each is skipped if already done):
  1. create a virtual environment in .venv (re-using packages you already have installed)
  2. install AgentML Studio + dev tools
  3. run the test-suite (offline, mock LLM)
  4. run the end-to-end demo (AutoML crew + agentic RAG)
  5. start the web UI on http://127.0.0.1:8000 and open it in your browser

Everything is also written to run_log.txt.
Tests and demo run only the first time (delete .venv\\agentml_checked to re-run them).
Pass --serve-only to skip steps 3-4.
"""

import os
import subprocess
import sys
import threading
import time
import webbrowser

for _stream in (sys.stdout, sys.stderr):  # Windows consoles (cp1252) can't print box-drawing chars
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(ROOT, ".venv")
PY = os.path.join(VENV, "Scripts" if os.name == "nt" else "bin", "python.exe" if os.name == "nt" else "python")
LOG = open(os.path.join(ROOT, "run_log.txt"), "w", encoding="utf-8")


def log(msg=""):
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


def run(title, args, env=None, check=True):
    log(f"\n===== {title} =====")
    log("$ " + " ".join(args))
    start = time.time()
    proc = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", **(env or {})})
    for line in proc.stdout:
        log(line.rstrip())
    code = proc.wait()
    log(f"----- {title}: {'OK' if code == 0 else 'FAILED (exit ' + str(code) + ')'} in {time.time() - start:.1f}s")
    if check and code != 0:
        log("\nStopped because a step failed. Send run_log.txt to get help.")
        sys.exit(code)
    return code


def main():
    log(f"AgentML Studio launcher — Python {sys.version.split()[0]} at {sys.executable}")
    if sys.version_info < (3, 10):  # noqa: UP036
        log("Python 3.10+ is required.")
        sys.exit(1)
    if not os.path.exists(PY):
        run("1/5 Create virtual environment", [sys.executable, "-m", "venv", "--system-site-packages", VENV])
    else:
        log("\n===== 1/5 Virtual environment already exists =====")
    run("2/5 Install AgentML Studio", [PY, "-m", "pip", "install", "--disable-pip-version-check", "-e", ".[dev]"])
    has_torch = subprocess.run([PY, "-c", "import torch"], capture_output=True).returncode == 0
    log("PyTorch found — deep_resnet model enabled" if has_torch
        else "PyTorch not installed — deep model falls back to sklearn MLP (optional: pip install torch)")

    marker = os.path.join(VENV, "agentml_checked")
    if "--serve-only" in sys.argv or os.path.exists(marker):
        log("\n===== 3-4/5 Tests & demo already passed once — skipping (delete .venv\\agentml_checked to re-run) =====")
    else:
        ok = run("3/5 Run tests", [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                 env={"AGENTML_PROVIDER": "mock"}, check=False) == 0
        run("4/5 Demo: AutoML crew + agentic RAG", [PY, "-m", "agentml", "demo"], check=False)
        if ok:
            open(marker, "w").close()

    log("\n===== 5/5 Web UI → http://127.0.0.1:8000  (stop with the ■ button / Ctrl+C) =====")
    threading.Timer(4, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    run("Web server", [PY, "-m", "agentml", "serve", "--port", "8000"], check=False)


if __name__ == "__main__":
    main()
