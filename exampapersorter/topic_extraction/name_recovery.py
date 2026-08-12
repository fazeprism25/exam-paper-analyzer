"""Recovers names for topics that came out of initial extraction with
resolution_status="name_unresolved" (a number/position is known, but no
legible name was found in the confirmed TOC evidence).

This does NOT assume any particular corroborating page exists (no "check
page 11" style rules). Instead it uses a near-universal textbook
convention: a chapter's own opening page almost always carries its title.
Given a topic's declared page number (from the table of contents), the
book-page -> PDF-page mapping (page_numbering.py) locates that page, a
deterministic header/footer self-check confirms we actually landed on it,
and only THEN does the LLM get asked to read whatever title is there --
never asked to guess on unconfirmed pages.

Every recovered name must be traceable to a specific confirmed page. If
the mapping can't be built, or a candidate page can't be confirmed, or
the confirmed page doesn't contain a legible title, the topic stays
unresolved/unidentified rather than being given an invented name.
"""
from __future__ import annotations

import logging
from pathlib import Path

from exampapersorter.config import Config
from exampapersorter.database import Database
from exampapersorter.docling_processing import extract_page_range_evidence
from exampapersorter.llm_client import LLMCallFailed, call_structured
from exampapersorter.schemas import ChapterNameRecoveryResult, PageRangeEvidence, Topic
from exampapersorter.topic_extraction.evidence_format import filter_high_signal_blocks, render_evidence
from exampapersorter.topic_extraction.page_numbering import (
    PageNumberMapping,
    build_page_number_mapping,
    confirm_expected_page,
)

logger = logging.getLogger(__name__)

CANDIDATE_WINDOW_BEFORE = 1
CANDIDATE_WINDOW_AFTER = 2
BATCH_SIZE = 8

SYSTEM_PROMPT = """You are confirming chapter/section titles on specific textbook pages.

Each candidate below tells you: the chapter/topic's expected number or identifier (from the \
textbook's own table of contents), and the actual evidence extracted from the specific page \
already confirmed to be where that chapter/section begins.

Rules:
- Only report a name that is genuinely, legibly present in the evidence for that candidate. Never \
invent a plausible-sounding title, even if the expected number matches.
- If the evidence for a candidate does not clearly show a chapter/section title matching the \
expected number (e.g. it's mid-paragraph content, or no heading is present), set status to \
"not_found" and leave name null.
- If more than one plausible, conflicting title appears for the same candidate, set status to \
"ambiguous" and briefly say why in `reason`.
- Otherwise, if exactly one clear title is present, set status to "resolved" and provide it.

Respond with exactly one resolution per candidate given, in the same order, using each candidate's \
topic_id."""


def _candidate_prompt_block(topic: Topic, evidence: PageRangeEvidence) -> str:
    filtered = filter_high_signal_blocks(evidence.blocks)
    return (
        f"--- Candidate topic_id={topic.id}, expected identifier={topic.name!r} ---\n"
        f"{render_evidence(filtered)}\n"
    )


def recover_names(
    topics: list[Topic],
    toc_window_evidence: PageRangeEvidence,
    pdf_path: Path,
    file_hash: str,
    db: Database,
    config: Config,
) -> list[Topic]:
    unresolved = [t for t in topics if t.resolution_status == "name_unresolved"]
    if not unresolved:
        return topics

    logger.info("Attempting name recovery for %d unresolved topics", len(unresolved))

    mapping = build_page_number_mapping(toc_window_evidence.blocks)
    if mapping is None:
        logger.warning(
            "Could not establish a book-page -> PDF-page mapping from the TOC window's "
            "headers/footers; %d topics will remain unidentified", len(unresolved)
        )
        updated_by_id = {t.id: _mark_unidentified(t) for t in unresolved}
        return [updated_by_id.get(t.id, t) for t in topics]

    logger.info("Page number mapping: pdf_page = book_page + %d (from %d samples)", mapping.offset, mapping.sample_size)

    candidates: dict[str, tuple[Topic, PageRangeEvidence]] = {}
    updated_by_id: dict[str, Topic] = {}

    for topic in unresolved:
        located = _locate_and_confirm_page(topic, mapping, pdf_path, file_hash, db)
        if located is None:
            updated_by_id[topic.id] = _mark_unidentified(topic)
            continue
        candidates[topic.id] = located

    topic_ids = list(candidates.keys())
    for i in range(0, len(topic_ids), BATCH_SIZE):
        batch_ids = topic_ids[i : i + BATCH_SIZE]
        batch = [candidates[tid] for tid in batch_ids]
        resolved = _resolve_batch(batch, config)
        updated_by_id.update(resolved)

    return [updated_by_id.get(t.id, t) for t in topics]


def _locate_and_confirm_page(
    topic: Topic, mapping: PageNumberMapping, pdf_path: Path, file_hash: str, db: Database
) -> tuple[Topic, PageRangeEvidence] | None:
    if topic.declared_page_number is None:
        logger.debug("Topic %s has no declared_page_number, cannot locate a candidate page", topic.id)
        return None

    book_page = topic.declared_page_number
    target_pdf_page = mapping.book_page_to_pdf_page(book_page)
    start = max(1, target_pdf_page - CANDIDATE_WINDOW_BEFORE)
    end = target_pdf_page + CANDIDATE_WINDOW_AFTER

    evidence = db.get_cached_evidence(file_hash, start, end)
    if evidence is None:
        evidence = extract_page_range_evidence(pdf_path, start, end)
        db.save_evidence(file_hash, evidence)

    if evidence.conversion_status not in ("success", "partial_success") or not evidence.blocks:
        logger.debug("Topic %s: candidate window pages %d-%d had no usable evidence", topic.id, start, end)
        return None

    confirmed_page = confirm_expected_page(evidence, book_page)
    if confirmed_page is None:
        logger.debug(
            "Topic %s: expected book page %d not confirmed anywhere in fetched PDF pages %d-%d "
            "(offset-based guess may be wrong for this region); leaving unidentified",
            topic.id, book_page, start, end,
        )
        return None

    logger.debug("Topic %s: confirmed book page %d = PDF page %d", topic.id, book_page, confirmed_page)
    return topic, evidence


def _mark_unidentified(topic: Topic) -> Topic:
    return topic.model_copy(update={"resolution_status": "unidentified"})


def _resolve_batch(
    batch: list[tuple[Topic, PageRangeEvidence]], config: Config
) -> dict[str, Topic]:
    prompt_blocks = "\n".join(_candidate_prompt_block(topic, evidence) for topic, evidence in batch)
    user_prompt = f"{len(batch)} candidates:\n\n{prompt_blocks}"

    try:
        result: ChapterNameRecoveryResult = call_structured(config, SYSTEM_PROMPT, user_prompt, ChapterNameRecoveryResult)
    except LLMCallFailed as exc:
        if exc.quota_exhausted:
            raise
        logger.error("Name recovery batch call failed: %s", exc)
        return {topic.id: _mark_unidentified(topic) for topic, _ in batch}

    topics_by_id = {topic.id: topic for topic, _ in batch}
    evidence_by_id = {topic.id: evidence for topic, evidence in batch}
    updated: dict[str, Topic] = {}

    resolutions_by_id = {r.topic_id: r for r in result.resolutions}
    for topic_id, topic in topics_by_id.items():
        resolution = resolutions_by_id.get(topic_id)
        if resolution is None:
            logger.warning("No resolution returned for topic %s in batch; leaving name_unresolved", topic_id)
            continue

        if resolution.status == "resolved" and resolution.name and resolution.name.strip():
            confirmed_page = confirm_expected_page(evidence_by_id[topic_id], topic.declared_page_number)
            updated[topic_id] = topic.model_copy(
                update={
                    "name": resolution.name.strip(),
                    "resolution_status": "resolved",
                    "name_evidence_source": "chapter_opening_page",
                    "name_source_pages": [confirmed_page] if confirmed_page else [],
                }
            )
        elif resolution.status == "ambiguous":
            updated[topic_id] = topic.model_copy(update={"resolution_status": "ambiguous"})
        else:
            # "not_found": page was confirmed correct, but no legible title was on it.
            updated[topic_id] = topic.model_copy(update={"resolution_status": "name_unresolved"})

    return updated
