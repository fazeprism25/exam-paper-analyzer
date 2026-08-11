#!/usr/bin/env python
"""Controlled reproducibility experiment for Stage 1 topic extraction.

Investigates whether identical evidence + prompt + qwen3:8b at
temperature=0 reliably produces the same topic list across repeated calls,
and whether fixing Ollama's `seed` option changes that. Motivating
observation: on the exact same evidence, three earlier runs of this same
call produced materially different results (clean names, then garbled
letter-spaced names, then numeric-only placeholders) despite temperature=0
in all three -- i.e. temperature alone did not make the call deterministic.

This calls the real extract_topics() against the textbook's real, already
cached page 1-25 evidence (no Docling recompute, no mocking) -- so results
reflect actual model behavior, not a simulation of it. Only `llm_seed`
varies between the two groups; every other config value is held fixed.

Usage: py scripts/reproducibility_experiment.py [--trials N] [--textbook PATH]
"""
from __future__ import annotations

import argparse
import json
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

DEFAULT_TEXTBOOK = Path("Satyanarayan Biochemistry.pdf")
OUT_DIR = Path("output/reproducibility_experiment")


def topic_fingerprint(result):
    return [
        (t.id, t.name, t.level, t.parent_id, t.declared_page_number, t.resolution_status, t.name_evidence_source)
        for t in result.topics
    ]


def run_group(label, config, evidence, n):
    """Returns a list of length n; an entry is None if that trial exhausted
    its retries (LLMCallFailed) rather than producing a result. A crashing
    trial does NOT abort the rest of the experiment -- an outright call
    failure (vs. a divergent-but-valid result) is itself a reproducibility
    data point worth recording, not something to hide by letting the whole
    script die.

    Resumable: if a trial's output file already exists on disk (e.g. this
    experiment was interrupted and restarted), that trial is loaded instead
    of re-run, so a restart doesn't throw away already-completed (and
    expensive) trials.
    """
    results = []
    for i in range(1, n + 1):
        out_path = OUT_DIR / f"{label}_trial{i}.json"
        if out_path.exists():
            print(f"[{label}] trial {i}/{n} already recorded at {out_path} -- reusing", flush=True)
            results.append(TopicExtractionResult.model_validate_json(out_path.read_text(encoding="utf-8")))
            continue

        print(f"[{label}] trial {i}/{n} starting...", flush=True)
        t0 = time.time()
        try:
            result = extract_topics(evidence, config)
        except LLMCallFailed as exc:
            elapsed = time.time() - t0
            print(f"[{label}] trial {i}/{n} FAILED after {elapsed:.1f}s -- exhausted retries: {exc}", flush=True)
            (OUT_DIR / f"{label}_trial{i}.FAILED.txt").write_text(str(exc), encoding="utf-8")
            results.append(None)
            continue
        elapsed = time.time() - t0
        print(f"[{label}] trial {i}/{n} done in {elapsed:.1f}s -- {len(result.topics)} topics", flush=True)
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        results.append(result)
    return results


def compare_group(label, results):
    n_failed = sum(1 for r in results if r is None)
    valid = [(i, r) for i, r in enumerate(results) if r is not None]
    counts = [len(r.topics) if r is not None else None for r in results]

    print(f"\n=== {label} summary ===", flush=True)
    print(f"topic counts across trials (None = call exhausted retries and failed): {counts}", flush=True)
    print(f"trials that failed outright: {n_failed}/{len(results)}", flush=True)

    if len(valid) < 2:
        print("fewer than 2 successful trials -- cannot compare for identity", flush=True)
        return False, [], n_failed

    fingerprints = [(i, topic_fingerprint(r)) for i, r in valid]
    baseline_idx, baseline_fp = fingerprints[0]
    all_identical = n_failed == 0 and all(fp == baseline_fp for _, fp in fingerprints)
    print(f"all successful trials byte-for-byte identical: {all_identical}", flush=True)

    diffs = []
    if not all_identical:
        for i, fp in fingerprints[1:]:
            if fp != baseline_fp:
                names_a = [f[1] for f in baseline_fp]
                names_b = [f[1] for f in fp]
                differing = [(a, b) for a, b in zip(names_a, names_b) if a != b]
                count_mismatch = len(names_a) != len(names_b)
                print(
                    f"  trial{baseline_idx + 1} vs trial{i + 1}: {len(differing)} differing names "
                    f"(out of {min(len(names_a), len(names_b))} compared); count_mismatch={count_mismatch}",
                    flush=True,
                )
                for a, b in differing[:10]:
                    print(f"    {a!r} != {b!r}", flush=True)
                diffs.append({"trial": i + 1, "differing_name_count": len(differing), "count_mismatch": count_mismatch})
    return all_identical, diffs, n_failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--textbook", type=Path, default=DEFAULT_TEXTBOOK)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    file_hash = compute_file_hash(args.textbook)
    db = Database(DEFAULT_CONFIG.database_path)
    evidence = db.get_cached_evidence(file_hash, 1, 25)
    db.close()
    if evidence is None:
        print("No cached evidence for pages 1-25 -- run the main pipeline once first so Docling extraction is cached.")
        sys.exit(1)

    print(f"Using cached evidence: {len(evidence.blocks)} blocks, pages {evidence.start_page}-{evidence.end_page}", flush=True)
    print(f"Trials per group: {args.trials}", flush=True)

    baseline_config = DEFAULT_CONFIG  # llm_seed=None: current production default
    seeded_config = replace(DEFAULT_CONFIG, llm_seed=args.seed)

    baseline_results = run_group("baseline_no_seed", baseline_config, evidence, args.trials)
    seeded_results = run_group(f"seeded_{args.seed}", seeded_config, evidence, args.trials)

    baseline_identical, baseline_diffs, baseline_failed = compare_group("baseline_no_seed", baseline_results)
    seeded_identical, seeded_diffs, seeded_failed = compare_group(f"seeded_{args.seed}", seeded_results)

    print("\n=== FINAL VERDICT ===", flush=True)
    print(f"baseline (no seed): identical={baseline_identical}, outright failures={baseline_failed}/{args.trials}", flush=True)
    print(f"seed={args.seed}: identical={seeded_identical}, outright failures={seeded_failed}/{args.trials}", flush=True)
    if seeded_identical and not baseline_identical and seeded_failed == 0:
        print("Seed measurably improves reproducibility -- recommend setting llm_seed by default.", flush=True)
    elif not seeded_identical or seeded_failed > 0:
        print(
            "Seed did NOT fully eliminate variance/failures -- residual nondeterminism likely comes from "
            "floating-point non-associativity in batched/multi-threaded inference (or genuine borderline "
            "output-length cases hitting the num_predict/num_ctx cap), not sampling RNG alone. "
            "Validation/recovery must remain the safety net, not a fixed seed alone.",
            flush=True,
        )
    else:
        print("Both groups were already identical -- inconclusive on seed's specific effect this run.", flush=True)

    summary = {
        "trials_per_group": args.trials,
        "seed_tested": args.seed,
        "baseline_topic_counts": [len(r.topics) if r is not None else None for r in baseline_results],
        "baseline_identical": baseline_identical,
        "baseline_failed_trials": baseline_failed,
        "baseline_diffs": baseline_diffs,
        "seeded_topic_counts": [len(r.topics) if r is not None else None for r in seeded_results],
        "seeded_identical": seeded_identical,
        "seeded_failed_trials": seeded_failed,
        "seeded_diffs": seeded_diffs,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved trial outputs + summary.json under {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
