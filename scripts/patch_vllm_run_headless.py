"""Add a run_headless alias for vLLM versions that only expose run_server.

Some verl versions import ``run_headless`` from ``vllm.entrypoints.cli.serve``,
while vLLM 0.8.x exposes the compatible entrypoint as ``run_server``. Run this
inside the training conda environment after installing vLLM.
"""

from __future__ import annotations

from pathlib import Path

import vllm


MARKER = "# GUI-G2 verl compatibility: run_headless alias"


def main() -> None:
    serve_py = Path(vllm.__file__).parent / "entrypoints" / "cli" / "serve.py"
    text = serve_py.read_text(encoding="utf-8")
    if "def run_headless" in text or MARKER in text:
        print(f"run_headless compatibility already present: {serve_py}")
        return
    if "def run_server" not in text:
        raise RuntimeError(f"Could not find run_server in {serve_py}")

    serve_py.write_text(
        text.rstrip()
        + "\n\n"
        + MARKER
        + "\n"
        + "run_headless = run_server\n",
        encoding="utf-8",
    )
    print(f"Patched run_headless alias into: {serve_py}")


if __name__ == "__main__":
    main()
