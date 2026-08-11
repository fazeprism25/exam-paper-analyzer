"""Orchestrates Stage 2 end to end for one question-paper PDF:

    Docling extraction (whole document, cached)
        -> paper boundary detection (cached, deterministically resolved)
        -> per paper: structure extraction (metadata + sections, cached)
        -> per paper: question extraction (cached)
        -> persistence (resumable at paper granularity)

Resumability mirrors Stage 1's pattern (page_range_evidence / toc_search_log
/ topic_extraction_runs): every LLM call's raw verdict is cached keyed by
(file_hash, page range, model), and a paper already saved with
status="success" is never reprocessed. IDs are assigned deterministically
by our own code (paper1, paper2, ... in page order; sections/questions
numbered within their paper) rather than invented by the LLM, so they stay
stable across a resumed run as long as the cached boundary verdict doesn't
change.
"""
from __future__ import annotations

import logging
from pathlib import Path

from exampapersorter.config import Config
from exampapersorter.database import Database
from exampapersorter.docling_processing import extract_page_range_evidence
from exampapersorter.llm_client import LLMCallFailed, get_last_call_metadata, unload_current_model
from exampapersorter.question_extraction.evidence_selection import extract_leading_number, slice_evidence
from exampapersorter.question_extraction.mcq_grouping import group_mcq_options
from exampapersorter.question_extraction.metadata import check_cross_field_consistency, merge_metadata
from exampapersorter.question_extraction.paper_boundaries import detect_paper_boundaries, resolve_paper_spans
from exampapersorter.question_extraction.paper_extraction import extract_paper_structure, extract_questions_for_paper
from exampapersorter.question_extraction.reconciliation import reconcile_extracted_questions
from exampapersorter.schemas import (
    MetadataFieldValue,
    Paper,
    PaperMetadata,
    Question,
    QuestionPaperFileResult,
    Section,
)

logger = logging.getLogger(__name__)


def _call_metadata() -> dict | None:
    """Snapshot of which model actually answered the call_structured() that
    just returned, for the cache-row auditing columns (see database.py's
    call_metadata_json columns) -- read immediately after each fresh
    call_structured()-derived call, never after a cache hit."""
    meta = get_last_call_metadata()
    return meta.to_dict() if meta else None


def _empty_metadata() -> PaperMetadata:
    empty = MetadataFieldValue()
    return PaperMetadata(exam_name=empty, institution=empty, subject=empty, date=empty, year=empty, paper_identifier=empty)


def _best(field: MetadataFieldValue) -> str | None:
    """Best-available reading for denormalizing onto each Question record --
    prefers the document's own value, falls back to the filename's. This is
    a convenience for later grouping/display only; the full, provenance-
    separated pair still lives on Paper.metadata and is never collapsed
    there."""
    return field.document_value or field.filename_value


def process_question_paper_file(
    pdf_path: Path, file_hash: str, total_pages: int, config: Config, db: Database
) -> QuestionPaperFileResult:
    db.register_question_paper_file(file_hash, str(pdf_path), total_pages)

    if db.get_question_paper_file_status(file_hash) == "success":
        logger.info(
            "File %s already fully processed (hash=%s); loading persisted result instead of reprocessing",
            pdf_path.name, file_hash[:12],
        )
        papers = db.get_papers_for_file(file_hash)
        return QuestionPaperFileResult(
            file_path=str(pdf_path), file_hash=file_hash, total_pages=total_pages,
            model_used=config.model_identifier, papers=papers, boundary_issues=[], status="success",
        )

    evidence = db.get_cached_evidence(file_hash, 1, total_pages)
    if evidence is None:
        logger.info("Running Docling on %s (pages 1-%d)", pdf_path.name, total_pages)
        evidence = extract_page_range_evidence(pdf_path, 1, total_pages)
        db.save_evidence(file_hash, evidence)

    if evidence.conversion_status not in ("success", "partial_success") or not evidence.blocks:
        db.update_question_paper_file_status(file_hash, "failed", config.model_identifier, evidence.error_message)
        return QuestionPaperFileResult(
            file_path=str(pdf_path), file_hash=file_hash, total_pages=total_pages,
            model_used=config.model_identifier, papers=[], boundary_issues=[], status="failed",
        )

    boundary_result = db.get_paper_boundary_verdict(file_hash, config.model_identifier)
    if boundary_result is None:
        try:
            boundary_result = detect_paper_boundaries(evidence, config)
        except LLMCallFailed as exc:
            logger.error("Paper boundary detection failed for %s: %s", pdf_path.name, exc)
            db.update_question_paper_file_status(file_hash, "failed", config.model_identifier, str(exc))
            return QuestionPaperFileResult(
                file_path=str(pdf_path), file_hash=file_hash, total_pages=total_pages,
                model_used=config.model_identifier, papers=[], boundary_issues=[], status="failed",
            )
        db.save_paper_boundary_verdict(file_hash, config.model_identifier, boundary_result, call_metadata=_call_metadata())
    else:
        logger.debug("Using cached paper boundary verdict for %s", pdf_path.name)

    spans, boundary_issues = resolve_paper_spans(boundary_result.papers, total_pages)
    logger.info("%s: %d paper(s) detected across %d pages", pdf_path.name, len(spans), total_pages)

    papers: list[Paper] = []
    any_failed = False
    for idx, span in enumerate(spans, start=1):
        paper_id = f"{file_hash[:12]}_paper{idx}"

        existing = db.get_paper(paper_id)
        if existing is not None and existing.status == "success":
            logger.info("Paper %s already processed successfully; skipping", paper_id)
            papers.append(existing)
            continue

        if config.llm_reload_policy == "reload_per_paper":
            logger.info("Reload policy reload_per_paper: unloading %s before paper %s", config.ollama_model, paper_id)
            unload_current_model(config)

        paper_evidence = slice_evidence(evidence, span.start_page, span.end_page)

        structure = db.get_paper_structure_verdict(file_hash, span.start_page, span.end_page, config.model_identifier)
        if structure is None:
            try:
                structure = extract_paper_structure(paper_evidence, config)
            except LLMCallFailed as exc:
                logger.error("Structure extraction failed for %s: %s", paper_id, exc)
                any_failed = True
                paper = Paper(
                    paper_id=paper_id, file_hash=file_hash, start_page=span.start_page, end_page=span.end_page,
                    metadata=_empty_metadata(), sections=[], boundary_confidence=span.confidence,
                    boundary_evidence=span.evidence, status="failed",
                    notes=[f"structure extraction failed: {exc}"],
                )
                db.save_paper(paper)
                papers.append(paper)
                continue
            db.save_paper_structure_verdict(
                file_hash, span.start_page, span.end_page, config.model_identifier, structure,
                call_metadata=_call_metadata(),
            )
        else:
            logger.debug("Using cached paper structure verdict for %s", paper_id)

        metadata = merge_metadata(structure, pdf_path.name)
        consistency_notes = check_cross_field_consistency(metadata)
        sections = [
            Section(
                section_id=f"{paper_id}_s{i + 1}", paper_id=paper_id, raw_label=s.raw_label,
                section_type=s.section_type, start_page=s.start_page, end_page=s.end_page, confidence=s.confidence,
            )
            for i, s in enumerate(structure.sections)
        ]

        q_result = db.get_question_extraction_verdict(file_hash, span.start_page, span.end_page, config.model_identifier)
        extraction_notes: list[str] = []
        if q_result is None:
            try:
                q_result, extraction_notes = extract_questions_for_paper(paper_evidence, structure.sections, config)
            except LLMCallFailed as exc:
                logger.error("Question extraction failed for %s: %s", paper_id, exc)
                any_failed = True
                paper = Paper(
                    paper_id=paper_id, file_hash=file_hash, start_page=span.start_page, end_page=span.end_page,
                    metadata=metadata, sections=sections, boundary_confidence=span.confidence,
                    boundary_evidence=span.evidence, status="partial",
                    notes=[f"question extraction failed: {exc}"],
                )
                db.save_paper(paper)
                papers.append(paper)
                continue
            db.save_question_extraction_verdict(
                file_hash, span.start_page, span.end_page, config.model_identifier, q_result,
                call_metadata=_call_metadata(),
            )
        else:
            logger.debug("Using cached question extraction verdict for %s", paper_id)

        # Applied here (not inside extract_questions) so it covers EVERY
        # source of q_result uniformly -- a fresh whole-paper call, the
        # truncation-driven split-fallback's merged chunks, AND a cached
        # verdict from a run predating this reconciliation logic. Cheap and
        # idempotent (only fills gaps, never overwrites a populated field),
        # so re-applying it to an already-good result is a harmless no-op.
        reconciled_questions = reconcile_extracted_questions(group_mcq_options(paper_evidence.blocks), q_result.questions)
        q_result = q_result.model_copy(update={"questions": reconciled_questions})

        exam_name = _best(metadata.exam_name)
        institution = _best(metadata.institution)
        subject = _best(metadata.subject)
        paper_year = _best(metadata.year)
        paper_date = _best(metadata.date)

        questions: list[Question] = []
        for i, q in enumerate(q_result.questions):
            section_id = None
            if q.section_index is not None and 0 <= q.section_index < len(sections):
                section_id = sections[q.section_index].section_id

            question_number = q.question_number
            question_text = q.question_text
            if question_number is None:
                backfilled_number, remainder = extract_leading_number(question_text)
                if backfilled_number is not None:
                    question_number = backfilled_number
                    question_text = remainder

            questions.append(
                Question(
                    question_id=f"{paper_id}_q{i + 1}", paper_id=paper_id, section_id=section_id,
                    source_filename=pdf_path.name, source_file_hash=file_hash,
                    exam_name=exam_name, institution=institution, subject=subject,
                    paper_year=paper_year, paper_date=paper_date,
                    question_number=question_number, question_text=question_text, options=q.options,
                    original_text=q.original_text, correction_applied=q.correction_applied,
                    question_type=q.question_type, source_pages=q.source_pages,
                    extraction_confidence=q.extraction_confidence, ambiguous=q.ambiguous,
                    ambiguity_note=q.ambiguity_note,
                )
            )

        paper = Paper(
            paper_id=paper_id, file_hash=file_hash, start_page=span.start_page, end_page=span.end_page,
            metadata=metadata, sections=sections, boundary_confidence=span.confidence,
            boundary_evidence=span.evidence, status="success", notes=extraction_notes + consistency_notes,
        )
        db.save_paper(paper)
        db.save_questions(questions)
        papers.append(paper)
        logger.info("Paper %s: %d section(s), %d question(s) extracted", paper_id, len(sections), len(questions))

    file_status = "failed" if not papers else ("partial" if any_failed else "success")
    db.update_question_paper_file_status(file_hash, file_status, config.model_identifier, None)
    return QuestionPaperFileResult(
        file_path=str(pdf_path), file_hash=file_hash, total_pages=total_pages, model_used=config.model_identifier,
        papers=papers, boundary_issues=boundary_issues, status=file_status,
    )
