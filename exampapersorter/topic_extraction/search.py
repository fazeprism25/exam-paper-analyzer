"""Adaptive page-range search for a table of contents / index.

Search pages in fixed-size windows starting from page 1. For each window:
  1. Get evidence (from the SQLite cache if we've already extracted this
     exact range for this file, otherwise run Docling and cache it).
  2. Ask the LLM whether this window is a TOC/index (this call is also
     cached, keyed by model, so switching models forces fresh verdicts
     but re-running the same model doesn't repeat work).
  3. Stop at the first window classified as table_of_contents/index with
     confidence >= threshold.

If a window's evidence contains MULTIPLE TOC-like blocks (a book can have
both a master contents list and a per-section mini-contents, as our real
test textbook does), that's resolved later, during topic extraction --
the extraction call sees the whole window's evidence and has to pick the
most complete listing, not the search loop.

Bounded by maximum_textbook_search_pages so a run can't silently crawl an
entire large book. If nothing is found before that bound, we report that
plainly (SearchResult.found=False) -- Stage 1 must not proceed to invent
topics from a TOC that was never confirmed.
"""
from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

from exampapersorter.config import Config
from exampapersorter.database import Database
from exampapersorter.docling_processing import extract_page_range_evidence
from exampapersorter.llm_client import LLMCallFailed, get_last_call_metadata
from exampapersorter.schemas import PageRangeEvidence, TOCDetectionResult
from exampapersorter.topic_extraction.toc_detection import detect_toc

logger = logging.getLogger(__name__)


class SearchAttempt(BaseModel):
    start_page: int
    end_page: int
    conversion_status: str
    document_structure: str | None = None
    confidence: float | None = None
    note: str | None = None


class SearchResult(BaseModel):
    found: bool
    evidence: PageRangeEvidence | None = None
    verdict: TOCDetectionResult | None = None
    attempts: list[SearchAttempt] = []


def find_table_of_contents(
    pdf_path: Path,
    file_hash: str,
    total_pages: int,
    db: Database,
    config: Config,
) -> SearchResult:
    max_page = min(config.maximum_textbook_search_pages, total_pages)
    attempts: list[SearchAttempt] = []

    start = 1
    window_size = config.initial_textbook_search_range
    while start <= max_page:
        end = min(start + window_size - 1, max_page)
        logger.info("Searching pages %d-%d for table of contents / index", start, end)

        evidence = db.get_cached_evidence(file_hash, start, end)
        if evidence is None:
            evidence = extract_page_range_evidence(pdf_path, start, end)
            db.save_evidence(file_hash, evidence)
        else:
            logger.debug("Using cached evidence for pages %d-%d", start, end)

        if evidence.conversion_status not in ("success", "partial_success"):
            logger.warning(
                "Skipping pages %d-%d: conversion status=%s (%s)",
                start, end, evidence.conversion_status, evidence.error_message,
            )
            attempts.append(
                SearchAttempt(
                    start_page=start, end_page=end, conversion_status=evidence.conversion_status,
                    note=evidence.error_message,
                )
            )
            start = end + 1
            window_size = config.textbook_search_step
            continue

        if not evidence.blocks:
            logger.info("No extractable content on pages %d-%d, skipping", start, end)
            attempts.append(
                SearchAttempt(start_page=start, end_page=end, conversion_status=evidence.conversion_status, note="no content extracted")
            )
            start = end + 1
            window_size = config.textbook_search_step
            continue

        verdict = db.get_toc_verdict(file_hash, start, end, config.model_identifier)
        if verdict is None:
            try:
                verdict = detect_toc(evidence, config)
            except LLMCallFailed as exc:
                if exc.quota_exhausted:
                    raise
                logger.error("TOC detection LLM call failed for pages %d-%d: %s", start, end, exc)
                attempts.append(
                    SearchAttempt(
                        start_page=start, end_page=end, conversion_status=evidence.conversion_status,
                        note=f"LLM call failed: {exc}",
                    )
                )
                start = end + 1
                window_size = config.textbook_search_step
                continue
            last_meta = get_last_call_metadata()
            db.log_toc_verdict(
                file_hash, start, end, config.model_identifier, verdict,
                call_metadata=last_meta.to_dict() if last_meta else None,
            )
        else:
            logger.debug("Using cached TOC verdict for pages %d-%d", start, end)

        attempts.append(
            SearchAttempt(
                start_page=start, end_page=end, conversion_status=evidence.conversion_status,
                document_structure=verdict.document_structure, confidence=verdict.confidence,
            )
        )

        if verdict.document_structure in ("table_of_contents", "index") and verdict.confidence >= config.toc_detection_confidence_threshold:
            logger.info(
                "Found %s on pages %d-%d (confidence=%.2f)",
                verdict.document_structure, start, end, verdict.confidence,
            )
            return SearchResult(found=True, evidence=evidence, verdict=verdict, attempts=attempts)

        start = end + 1
        window_size = config.textbook_search_step

    logger.warning("No table of contents/index found within first %d pages", max_page)
    return SearchResult(found=False, attempts=attempts)
