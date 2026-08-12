"""Topic Authority: the single convergence point between "I have a
textbook" and "I already have an index/syllabus/topic list" user journeys.

Both journeys end up producing the exact same thing Stage 3/5/6 already
consume -- a (file_hash, list[Topic]) pair persisted via
Database.save_topic_extraction_run / read back via
Database.get_latest_topic_extraction. Neither of those DB methods (nor
anything in topic_classification/, frequency_analysis/, or final_analysis/)
cares whether the file_hash they were keyed on came from an actual textbook
PDF -- textbook_files/topic_extraction_runs are just (file_hash -> topics)
bookkeeping, never re-validated against a real textbook file elsewhere in
the pipeline. That's what makes this module possible as a thin layer
rather than a second parallel pipeline: it only has to produce a
TopicExtractionResult and persist it the same way Stage 1 does, then every
later stage already works unmodified.

Two source kinds are supported:
  - Textbook PDF: reuses Stage 1's own TOC-search + extraction machinery
    verbatim (topic_extraction/search.py, topic_extraction/extract.py) --
    this module adds only a cache-check in front of it, so re-running
    against an already-processed textbook doesn't repeat LLM calls.
  - Index/syllabus file (.pdf/.txt/.md/.json): a PDF index is run through
    the SAME extract_topics() LLM call Stage 1 uses (skipping the TOC
    *search* phase, since the whole file is already known to be the
    index/contents/syllabus). A .txt/.md/.json index is parsed fully
    deterministically (topic_extraction/index_parsing.py) -- no LLM call,
    no API key needed for that path at all.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from exampapersorter.config import Config
from exampapersorter.database import Database
from exampapersorter.docling_processing import extract_page_range_evidence
from exampapersorter.llm_client import LLMCallFailed
from exampapersorter.output_writer import write_stage1_outputs
from exampapersorter.pdf_utils import compute_file_hash, get_page_count
from exampapersorter.schemas import Topic, TopicExtractionResult, ValidationReport
from exampapersorter.topic_extraction.extract import extract_topics
from exampapersorter.topic_extraction.hierarchy import derive_parent_ids
from exampapersorter.topic_extraction.index_parsing import (
    IndexParsingError,
    parse_topics_from_json,
    parse_topics_from_outline_text,
)
from exampapersorter.topic_extraction.name_recovery import recover_names
from exampapersorter.topic_extraction.search import find_table_of_contents
from exampapersorter.validation import reconcile_resolution_status, validate_topics

logger = logging.getLogger(__name__)

# Recorded in topic_extraction_runs.model_used for deterministically-parsed
# index sources -- never a real LLM model, so it's kept visually distinct
# from any config.model_identifier value (which is always "<provider>..."
# shaped) and stable across runs/pool changes.
DETERMINISTIC_PARSER_LABEL = "deterministic-index-parser-v1"

_INDEX_SUPPORTED_SUFFIXES = (".pdf", ".txt", ".md", ".markdown", ".json")


class TopicAuthorityError(RuntimeError):
    """Raised when a topic source (textbook or index/syllabus) cannot be
    turned into a usable topic hierarchy -- caller is expected to report
    this and exit cleanly, mirroring FrequencyAnalysisPrerequisiteError's
    role for later stages.

    quota_exhausted=True carries LLMCallFailed.quota_exhausted through, for
    the one case (an account-level OpenRouter quota/credit exhaustion) that
    isn't an ordinary "can't build a topic list" failure -- callers use it
    to pause the whole analysis (resumable) rather than reporting a hard
    failure."""

    def __init__(self, message: str, *, quota_exhausted: bool = False):
        super().__init__(message)
        self.quota_exhausted = quota_exhausted


@dataclass
class TopicAuthorityResult:
    file_hash: str
    source_path: str
    source_type: str  # "textbook" | "index_pdf" | "index_outline" | "index_json"
    topics: list[Topic]
    validation: ValidationReport | None
    from_cache: bool


def resolve_textbook_topic_authority(textbook_path: Path, config: Config, db: Database) -> TopicAuthorityResult:
    """Textbook mode. Reuses Stage 1's find_table_of_contents/extract_topics
    verbatim -- the only addition here is checking for an already-persisted
    topic list first, so re-running `analyze` against the same textbook
    never repeats Stage 1's LLM calls."""
    if not textbook_path.exists():
        raise TopicAuthorityError(f"Textbook not found: {textbook_path}")

    total_pages = get_page_count(textbook_path)
    file_hash = compute_file_hash(textbook_path)
    db.register_textbook(file_hash, str(textbook_path), total_pages)

    cached = db.get_latest_topic_extraction(file_hash)
    if cached is not None:
        logger.info(
            "Reusing already-extracted topic list for %s (%d topics, hash=%s)",
            textbook_path.name, len(cached.topics), file_hash[:12],
        )
        return TopicAuthorityResult(
            file_hash=file_hash, source_path=str(textbook_path), source_type="textbook",
            topics=cached.topics, validation=None, from_cache=True,
        )

    search_cap = min(config.maximum_textbook_search_pages, total_pages)
    search_result = find_table_of_contents(textbook_path, file_hash, total_pages, db, config)
    if not search_result.found:
        raise TopicAuthorityError(
            f"No table of contents/index found in {textbook_path.name} within the first {search_cap} pages. "
            "If you have a separate index, syllabus, or topic list, supply that instead via --index."
        )

    try:
        topics = extract_topics(search_result.evidence, config)
    except LLMCallFailed as exc:
        db.save_topic_extraction_run(
            file_hash, config.model_identifier, search_result.evidence.start_page, search_result.evidence.end_page,
            status="failed", topics=None, validation=None,
        )
        raise TopicAuthorityError(
            f"Topic extraction from {textbook_path.name} failed: {exc}", quota_exhausted=exc.quota_exhausted,
        ) from exc

    topics = TopicExtractionResult(topics=reconcile_resolution_status(topics.topics))
    unresolved_count = sum(1 for t in topics.topics if t.resolution_status == "name_unresolved")
    if unresolved_count:
        recovered = recover_names(topics.topics, search_result.evidence, textbook_path, file_hash, db, config)
        topics = TopicExtractionResult(topics=recovered)
    topics = TopicExtractionResult(topics=derive_parent_ids(topics.topics))

    valid_page_range = (search_result.evidence.start_page, search_result.evidence.end_page)
    validation = validate_topics(topics, config, valid_page_range=valid_page_range)
    status = "success" if validation.passed else "needs_review"
    db.save_topic_extraction_run(
        file_hash, config.model_identifier, search_result.evidence.start_page, search_result.evidence.end_page,
        status=status, topics=topics, validation=validation,
    )
    write_stage1_outputs(
        config.output_directory, textbook_path, topics, validation,
        search_result.evidence, search_result.verdict, search_result.attempts, config.model_identifier,
    )

    return TopicAuthorityResult(
        file_hash=file_hash, source_path=str(textbook_path), source_type="textbook",
        topics=topics.topics, validation=validation, from_cache=False,
    )


def resolve_index_topic_authority(index_path: Path, config: Config, db: Database) -> TopicAuthorityResult:
    """Index/syllabus mode. Dispatches on file extension: .pdf reuses Stage
    1's extract_topics() LLM call directly (no TOC search phase -- the
    whole file is already known to be the index); .txt/.md/.json are parsed
    fully deterministically (topic_extraction/index_parsing.py), no LLM
    call and no API key required for those two."""
    if not index_path.exists():
        raise TopicAuthorityError(f"Index/syllabus file not found: {index_path}")
    if not index_path.is_file():
        raise TopicAuthorityError(f"Index/syllabus path is not a file: {index_path}")

    suffix = index_path.suffix.lower()
    if suffix not in _INDEX_SUPPORTED_SUFFIXES:
        raise TopicAuthorityError(
            f"Unsupported index/syllabus file type '{suffix or index_path.name}'. "
            f"Supported formats: {', '.join(_INDEX_SUPPORTED_SUFFIXES)}"
        )

    file_hash = compute_file_hash(index_path)
    cached = db.get_latest_topic_extraction(file_hash)
    if cached is not None:
        logger.info(
            "Reusing already-loaded topic list for %s (%d topics, hash=%s)",
            index_path.name, len(cached.topics), file_hash[:12],
        )
        return TopicAuthorityResult(
            file_hash=file_hash, source_path=str(index_path), source_type="index",
            topics=cached.topics, validation=None, from_cache=True,
        )

    if suffix == ".pdf":
        return _resolve_index_pdf(index_path, file_hash, config, db)
    if suffix == ".json":
        return _resolve_index_json(index_path, file_hash, config, db)
    return _resolve_index_outline(index_path, file_hash, config, db)


def _resolve_index_pdf(index_path: Path, file_hash: str, config: Config, db: Database) -> TopicAuthorityResult:
    total_pages = get_page_count(index_path)
    if total_pages > config.maximum_textbook_search_pages:
        raise TopicAuthorityError(
            f"{index_path.name} has {total_pages} pages -- too large to treat as an index/syllabus "
            f"(limit {config.maximum_textbook_search_pages} pages). If this is a full textbook, use --textbook instead."
        )
    db.register_textbook(file_hash, str(index_path), total_pages)

    evidence = db.get_cached_evidence(file_hash, 1, total_pages)
    if evidence is None:
        evidence = extract_page_range_evidence(index_path, 1, total_pages)
        db.save_evidence(file_hash, evidence)
    if evidence.conversion_status not in ("success", "partial_success") or not evidence.blocks:
        raise TopicAuthorityError(
            f"Could not extract content from {index_path.name}: {evidence.error_message or 'no content found'}"
        )

    try:
        extracted = extract_topics(evidence, config)
    except LLMCallFailed as exc:
        raise TopicAuthorityError(
            f"Topic extraction from {index_path.name} failed: {exc}", quota_exhausted=exc.quota_exhausted,
        ) from exc

    extracted = TopicExtractionResult(topics=reconcile_resolution_status(extracted.topics))
    extracted = TopicExtractionResult(topics=derive_parent_ids(extracted.topics))
    validation = validate_topics(extracted, config, valid_page_range=(1, total_pages))
    status = "success" if validation.passed else "needs_review"
    db.save_topic_extraction_run(
        file_hash, config.model_identifier, 1, total_pages, status=status, topics=extracted, validation=validation,
    )

    return TopicAuthorityResult(
        file_hash=file_hash, source_path=str(index_path), source_type="index_pdf",
        topics=extracted.topics, validation=validation, from_cache=False,
    )


def _resolve_index_outline(index_path: Path, file_hash: str, config: Config, db: Database) -> TopicAuthorityResult:
    text = index_path.read_text(encoding="utf-8", errors="replace")
    try:
        topics = parse_topics_from_outline_text(text)
    except IndexParsingError as exc:
        raise TopicAuthorityError(f"Could not parse {index_path.name} as a topic outline: {exc}") from exc
    return _persist_deterministic_topics(index_path, file_hash, topics, config, db, source_type="index_outline")


def _resolve_index_json(index_path: Path, file_hash: str, config: Config, db: Database) -> TopicAuthorityResult:
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TopicAuthorityError(f"Could not read {index_path.name} as JSON: {exc}") from exc
    try:
        topics = parse_topics_from_json(data)
    except IndexParsingError as exc:
        raise TopicAuthorityError(f"Could not parse {index_path.name} as a topic list: {exc}") from exc
    return _persist_deterministic_topics(index_path, file_hash, topics, config, db, source_type="index_json")


def _persist_deterministic_topics(
    index_path: Path, file_hash: str, topics: list[Topic], config: Config, db: Database, source_type: str
) -> TopicAuthorityResult:
    db.register_textbook(file_hash, str(index_path), 0)
    result = TopicExtractionResult(topics=topics)
    validation = validate_topics(result, config)
    status = "success" if validation.passed else "needs_review"
    db.save_topic_extraction_run(
        file_hash, DETERMINISTIC_PARSER_LABEL, 0, 0, status=status, topics=result, validation=validation,
    )
    return TopicAuthorityResult(
        file_hash=file_hash, source_path=str(index_path), source_type=source_type,
        topics=result.topics, validation=validation, from_cache=False,
    )
