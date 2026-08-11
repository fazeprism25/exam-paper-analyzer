#!/usr/bin/env python
"""One-off: does a fresh model reload reset the deterministic-truncation
streak observed after unloading qwen3:8b with `ollama stop`?"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exampapersorter.config import DEFAULT_CONFIG
from exampapersorter.database import Database
from exampapersorter.llm_client import LLMCallFailed
from exampapersorter.pdf_utils import compute_file_hash
from exampapersorter.topic_extraction.extract import extract_topics

textbook = Path("Satyanarayan Biochemistry.pdf")
file_hash = compute_file_hash(textbook)
db = Database(DEFAULT_CONFIG.database_path)
evidence = db.get_cached_evidence(file_hash, 1, 25)
db.close()

t0 = time.time()
try:
    result = extract_topics(evidence, DEFAULT_CONFIG)
    print(f"SUCCESS after {time.time() - t0:.1f}s -- {len(result.topics)} topics", flush=True)
    Path("output/reproducibility_experiment/post_reload_test.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
except LLMCallFailed as exc:
    print(f"FAILED after {time.time() - t0:.1f}s -- {exc}", flush=True)
