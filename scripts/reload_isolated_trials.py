#!/usr/bin/env python
"""Runs N trials of extract_topics against the same cached evidence, with a
FRESH `ollama stop <model>` unload before every single trial.

Follows up on a finding from reproducibility_experiment.py: without an
unload between calls, the model appears to drift into a persistent
degraded state after the first call in a session (repeatable, deterministic
truncated-JSON failures on every subsequent call, until reload). That
confound makes the original baseline/seeded comparison unreliable for
isolating what `seed` itself does. This script removes the confound by
reloading before every trial in both groups, so the only thing that varies
between "fresh_no_seed" and "fresh_seeded" runs is the seed option itself.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exampapersorter.config import DEFAULT_CONFIG
from exampapersorter.database import Database
from exampapersorter.llm_client import LLMCallFailed
from exampapersorter.pdf_utils import compute_file_hash
from exampapersorter.schemas import TopicExtractionResult
from exampapersorter.topic_extraction.extract import extract_topics

OUT_DIR = Path("output/reproducibility_experiment")


def unload_model(model_name: str) -> None:
    subprocess.run(["ollama", "stop", model_name], capture_output=True)


def topic_fingerprint(result):
    return [
        (t.id, t.name, t.level, t.parent_id, t.declared_page_number, t.resolution_status, t.name_evidence_source)
        for t in result.topics
    ]


def run_trials(label, config, evidence, n, model_name):
    results = []
    for i in range(1, n + 1):
        out_path = OUT_DIR / f"{label}_trial{i}.json"
        if out_path.exists():
            print(f"[{label}] trial {i}/{n} already recorded -- reusing", flush=True)
            results.append(TopicExtractionResult.model_validate_json(out_path.read_text(encoding="utf-8")))
            continue

        print(f"[{label}] unloading {model_name} before trial {i}/{n}...", flush=True)
        unload_model(model_name)

        print(f"[{label}] trial {i}/{n} starting (fresh load)...", flush=True)
        t0 = time.time()
        try:
            result = extract_topics(evidence, config)
        except LLMCallFailed as exc:
            elapsed = time.time() - t0
            print(f"[{label}] trial {i}/{n} FAILED after {elapsed:.1f}s: {exc}", flush=True)
            (OUT_DIR / f"{label}_trial{i}.FAILED.txt").write_text(str(exc), encoding="utf-8")
            results.append(None)
            continue
        elapsed = time.time() - t0
        print(f"[{label}] trial {i}/{n} done in {elapsed:.1f}s -- {len(result.topics)} topics", flush=True)
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--textbook", type=Path, default=Path("Satyanarayan Biochemistry.pdf"))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    file_hash = compute_file_hash(args.textbook)
    db = Database(DEFAULT_CONFIG.database_path)
    evidence = db.get_cached_evidence(file_hash, 1, 25)
    db.close()
    if evidence is None:
        print("No cached evidence for pages 1-25.")
        sys.exit(1)

    # Trial 1 of "fresh_no_seed" is already covered by the original
    # baseline_no_seed_trial1.json (a fresh-loaded, no-seed call at the
    # very start of this session), and post_reload_test.json is a second,
    # independently fresh-loaded no-seed call -- both already saved under
    # the names below so run_trials picks them up as already-done.
    (OUT_DIR / "fresh_no_seed_trial1.json").write_bytes((OUT_DIR / "baseline_no_seed_trial1.json").read_bytes())
    (OUT_DIR / "fresh_no_seed_trial2.json").write_bytes((OUT_DIR / "post_reload_test.json").read_bytes())

    print("=== Group: fresh_no_seed (reload before every trial) ===", flush=True)
    fresh_no_seed = run_trials("fresh_no_seed", DEFAULT_CONFIG, evidence, args.trials, DEFAULT_CONFIG.ollama_model)

    print("\n=== Group: fresh_seeded (reload before every trial) ===", flush=True)
    seeded_config = replace(DEFAULT_CONFIG, llm_seed=args.seed)
    fresh_seeded = run_trials(f"fresh_seeded_{args.seed}", seeded_config, evidence, args.trials, DEFAULT_CONFIG.ollama_model)

    def summarize(label, results):
        valid = [r for r in results if r is not None]
        n_failed = sum(1 for r in results if r is None)
        fps = [topic_fingerprint(r) for r in valid]
        identical = len(valid) > 0 and n_failed == 0 and all(fp == fps[0] for fp in fps)
        print(f"\n{label}: {len(results)} trials, {n_failed} failed, identical={identical}", flush=True)
        return {"trials": len(results), "failed": n_failed, "identical": identical, "topic_counts": [len(r.topics) if r else None for r in results]}

    summary = {
        "fresh_no_seed": summarize("fresh_no_seed", fresh_no_seed),
        "fresh_seeded": summarize(f"fresh_seeded_{args.seed}", fresh_seeded),
        "seed_tested": args.seed,
    }
    (OUT_DIR / "reload_isolated_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved reload_isolated_summary.json under {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
