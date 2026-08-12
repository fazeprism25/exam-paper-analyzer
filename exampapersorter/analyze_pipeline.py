"""Shared Stage 2/3 orchestration helpers used by both the `analyze` CLI
command (cli.py) and the desktop GUI (exampapersorter/app.py), so the two
entry points invoke the exact same per-file processing/caching path
(process_question_paper_file, classify_paper) rather than diverging.

Moved out of cli.py (rather than imported from it) so the GUI does not
depend on the top-level `cli` script module, which is only reliably
importable when the repository root happens to be on sys.path.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from exampapersorter.config import Config
from exampapersorter.database import Database
from exampapersorter.output_writer import write_stage2_outputs
from exampapersorter.pdf_utils import compute_file_hash, get_page_count
from exampapersorter.question_extraction.pipeline import process_question_paper_file
from exampapersorter.topic_classification.pipeline import classify_paper

logger = logging.getLogger(__name__)


def compute_job_id(topic_source_type: str, topic_source_path: Path, question_papers_dir: Path) -> str:
    """Deterministic identity for one `analyze` run, used as
    analysis_jobs.job_id (see database.py) -- keyed on the RESOLVED
    (absolute) paths so the same textbook/folder combo always maps to the
    same job regardless of the working directory it was launched from,
    letting a re-run (whether a genuine resume or just `analyze` invoked
    again) upsert the same row instead of creating a duplicate."""
    key = f"{topic_source_type}:{Path(topic_source_path).resolve()}:{Path(question_papers_dir).resolve()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def compute_job_progress(pdf_paths: list[Path], db: Database) -> tuple[int, int]:
    """(completed, total) exam papers for a GUI progress readout ("12 of 20
    papers already processed"). Always recomputed live from
    question_paper_files (never stored/cached anywhere) so it can never
    drift out of sync with the actual persisted Stage 2 state -- a file
    counts as done only once process_question_paper_file has marked it
    status="success" (see question_extraction/pipeline.py)."""
    completed = sum(1 for p in pdf_paths if db.get_question_paper_file_status(compute_file_hash(p)) == "success")
    return completed, len(pdf_paths)


def extract_questions_for_files(
    pdf_paths: list[Path], config: Config, db: Database
) -> tuple[list, dict[str, int], dict[str, int]]:
    """Runs Stage 2 (question extraction) over `pdf_paths`, writing Stage 2
    output files, and returns (file_results, question_counts_by_paper,
    question_type_counts)."""
    file_results = []
    question_counts_by_paper: dict[str, int] = {}
    question_type_counts: dict[str, int] = {}
    for pdf_path in pdf_paths:
        logger.info("Processing %s", pdf_path.name)
        file_hash = compute_file_hash(pdf_path)
        total_pages = get_page_count(pdf_path)
        result = process_question_paper_file(pdf_path, file_hash, total_pages, config, db)
        file_results.append(result)

        questions_by_paper = {p.paper_id: db.get_questions_for_paper(p.paper_id) for p in result.papers}
        for paper_id, questions in questions_by_paper.items():
            question_counts_by_paper[paper_id] = len(questions)
            for q in questions:
                question_type_counts[q.question_type] = question_type_counts.get(q.question_type, 0) + 1
        written = write_stage2_outputs(config.output_directory, result, questions_by_paper)
        logger.info("%s: wrote %d output file(s)", pdf_path.name, len(written))
    return file_results, question_counts_by_paper, question_type_counts


def classify_papers_for_files(
    pdf_paths: list[Path], topics: list, textbook_file_hash: str, config: Config, db: Database
) -> tuple[list[dict], dict[str, list], dict[str, int], list[str]]:
    """Runs Stage 3 (topic classification) over `pdf_paths` against an
    already-resolved topic hierarchy. Returns (file_summaries,
    questions_by_paper, totals, skipped)."""
    file_summaries: list[dict] = []
    questions_by_paper: dict[str, list] = {}
    totals = {"classified": 0, "no_match": 0, "ambiguous": 0, "unclassified": 0}
    skipped: list[str] = []

    for pdf_path in pdf_paths:
        file_hash = compute_file_hash(pdf_path)
        status = db.get_question_paper_file_status(file_hash)
        if status not in ("success", "partial"):
            logger.warning(
                "Skipping %s: Stage 2 has not successfully run for this file yet (status=%s)", pdf_path.name, status,
            )
            skipped.append(pdf_path.name)
            continue

        papers = db.get_papers_for_file(file_hash)
        logger.info("Classifying %s (%d paper(s))", pdf_path.name, len(papers))

        paper_summaries = []
        for paper in papers:
            result = classify_paper(paper, topics, textbook_file_hash, config, db)
            paper_summaries.append(
                {
                    "paper_id": result.paper_id,
                    "status": result.status,
                    "total_questions": result.total_questions,
                    "classified_count": result.classified_count,
                    "no_match_count": result.no_match_count,
                    "ambiguous_count": result.ambiguous_count,
                    "unclassified_count": result.unclassified_count,
                    "error_message": result.error_message,
                }
            )
            for status_key in ("classified", "no_match", "ambiguous", "unclassified"):
                totals[status_key] += getattr(result, f"{status_key}_count")
            questions_by_paper[paper.paper_id] = db.get_questions_for_paper(paper.paper_id)

        file_summaries.append({"source_pdf": pdf_path.name, "file_hash": file_hash, "papers": paper_summaries})

    return file_summaries, questions_by_paper, totals, skipped
