#!/usr/bin/env python
"""Orchestrates the Stage 1 hardening validation run, in order:

  1. reproducibility_experiment.py  -- baseline vs. seeded, N trials each
  2. recovery_regression_check.py   -- real recovery path against a real failure
  3. final Stage 1 pipeline re-run  -- cli.py extract-topics, using the seed
                                        the experiment recommends (if any)

Run strictly sequentially, never in parallel: only one process talks to the
local Ollama server at a time. This machine has ~2GB free RAM once
Docling+Ollama are both loaded, and concurrent calls risk OOM/hangs
(observed directly in earlier sessions).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPRO_SUMMARY = Path("output/reproducibility_experiment/summary.json")


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--textbook", default="Satyanarayan Biochemistry.pdf")
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    py = sys.executable

    print("=== Step 1/3: reproducibility experiment ===", flush=True)
    rc = run([py, "scripts/reproducibility_experiment.py", "--textbook", args.textbook, "--trials", str(args.trials)])
    if rc != 0:
        print(f"Reproducibility experiment exited {rc} -- continuing anyway, final run will use production default.")

    seed_to_use = None
    if REPRO_SUMMARY.exists():
        summary = json.loads(REPRO_SUMMARY.read_text(encoding="utf-8"))
        if summary.get("seeded_identical") and not summary.get("baseline_identical"):
            seed_to_use = summary.get("seed_tested")
            print(f"\nReproducibility experiment recommends seed={seed_to_use} -- using it for the final run.")
        else:
            print(
                "\nReproducibility experiment does not support relying on a fixed seed alone -- "
                "final run uses the production default (no seed), leaning on validation as the safety net."
            )

    print("\n=== Step 2/3: recovery regression check ===", flush=True)
    rc = run([py, "scripts/recovery_regression_check.py", "--textbook", args.textbook])
    print(f"Recovery regression check exited {rc} (0=pass, 2=did not resolve, 3=resolved but wrong name).")

    print("\n=== Step 3/3: final Stage 1 pipeline re-run ===", flush=True)
    final_cmd = [py, "cli.py", "extract-topics", "--textbook", args.textbook]
    if seed_to_use is not None:
        final_cmd += ["--seed", str(seed_to_use)]
    rc = run(final_cmd)
    print(f"Final pipeline re-run exited {rc}.")

    print("\n=== Stage 1 hardening pipeline complete ===", flush=True)


if __name__ == "__main__":
    main()
