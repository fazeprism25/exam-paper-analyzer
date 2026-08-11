#!/usr/bin/env python
"""Exercises the real name-recovery pipeline against a real failure case.

The recovery code (topic_extraction/name_recovery.py) has so far only been
covered by unit tests with mocked Docling/LLM calls: the one successful full
pipeline run happened to resolve the master TOC cleanly on the first try, so
recover_names() was never actually triggered end-to-end against real data.

We don't have the raw output of the earlier bad run saved (it was
overwritten before this need was identified), so this constructs an
equivalent real failure instead of replaying that specific one: take a topic
from the last known-good, validated extraction (output/topics.json) that IS
resolved and DOES have a declared_page_number, strip it back down to the
"position known, name unknown" placeholder state extract.py itself uses for
resolution_status="name_unresolved", and hand it to the real recover_names()
-- real Docling extraction of the small candidate page window, real
header/footer self-check, real LLM read of that page. If it recovers the
name the topic already had, that's a real, verifiable demonstration of the
full path: bad name -> declared page -> located page -> confirmed page ->
LLM read -> validated against evidence.

Which topic is used is chosen programmatically (index into the filtered
candidate list), not hand-picked by chapter name or page number, so this
isn't a textbook-specific rule.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exampapersorter.config import DEFAULT_CONFIG
from exampapersorter.database import Database
from exampapersorter.pdf_utils import compute_file_hash
from exampapersorter.schemas import Topic, TopicExtractionResult
from exampapersorter.topic_extraction.name_recovery import recover_names

DEFAULT_TEXTBOOK = Path("Satyanarayan Biochemistry.pdf")
DEFAULT_TOPICS_JSON = Path("output/topics.json")
OUT_PATH = Path("output/recovery_regression_check.json")


def pick_regression_candidate(topics: list[Topic]) -> Topic:
    candidates = [t for t in topics if t.resolution_status == "resolved" and t.declared_page_number is not None]
    if not candidates:
        raise SystemExit("No resolved topic with a declared_page_number found to use as a regression candidate")
    return candidates[len(candidates) // 2]  # a generic, non-hand-picked selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--textbook", type=Path, default=DEFAULT_TEXTBOOK)
    parser.add_argument("--topics-json", type=Path, default=DEFAULT_TOPICS_JSON)
    args = parser.parse_args()

    raw = json.loads(args.topics_json.read_text(encoding="utf-8"))
    topics = [Topic(**t) for t in raw["topics"]]

    original = pick_regression_candidate(topics)
    print(f"Regression candidate: id={original.id!r} name={original.name!r} declared_page_number={original.declared_page_number}")

    broken = original.model_copy(
        update={"name": "UNKNOWN", "resolution_status": "name_unresolved", "name_evidence_source": None, "name_source_pages": []}
    )
    print(f"Simulated failure state: name={broken.name!r} resolution_status={broken.resolution_status!r}")

    file_hash = compute_file_hash(args.textbook)
    db = Database(DEFAULT_CONFIG.database_path)
    toc_window_evidence = db.get_cached_evidence(file_hash, 1, 25)
    if toc_window_evidence is None:
        print("No cached TOC-window evidence for pages 1-25 -- run the main pipeline once first.")
        sys.exit(1)

    print("Running real recover_names() -- real Docling extraction of the candidate page window + real LLM call...", flush=True)
    result = recover_names([broken], toc_window_evidence, args.textbook, file_hash, db, DEFAULT_CONFIG)
    db.close()

    recovered = result[0]
    name_match = recovered.name.strip().lower() == original.name.strip().lower()

    print("\n=== Recovery regression result ===")
    print(f"original name (ground truth): {original.name!r}")
    print(f"recovered name:                {recovered.name!r}")
    print(f"resolution_status:             {recovered.resolution_status!r}")
    print(f"name_evidence_source:          {recovered.name_evidence_source!r}")
    print(f"name_source_pages:             {recovered.name_source_pages!r}")
    print(f"exact case-insensitive match:  {name_match}")

    report = {
        "textbook": str(args.textbook),
        "candidate_topic_id": original.id,
        "original_name": original.name,
        "declared_page_number": original.declared_page_number,
        "simulated_failure_name": broken.name,
        "recovered_name": recovered.name,
        "resolution_status": recovered.resolution_status,
        "name_evidence_source": recovered.name_evidence_source,
        "name_source_pages": recovered.name_source_pages,
        "exact_match": name_match,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report to {OUT_PATH}")

    if recovered.resolution_status != "resolved":
        print("\nRECOVERY DID NOT RESOLVE THE TOPIC -- see resolution_status above.")
        sys.exit(2)
    if not name_match:
        print("\nRECOVERED NAME DOES NOT MATCH GROUND TRUTH -- inspect manually, do not treat as a clean pass.")
        sys.exit(3)

    print("\nPASS: real recovery pipeline independently re-derived the correct name from the real PDF page.")


if __name__ == "__main__":
    main()
