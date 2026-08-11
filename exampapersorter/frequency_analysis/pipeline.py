"""Orchestrates Stage 5 end to end: loads already-persisted Stage 1-4
records for one textbook from the database and hands them to
frequency_analysis.aggregate's pure functions. No LLM calls, no embedding
calls -- this module's only job is reading the database and checking that
the Stage 1-4 prerequisites it depends on actually ran.

Deliberately does NOT re-run Stages 1-4 (mirrors topic_classification/
pipeline.py's and deduplication/pipeline.py's own relationship to their
prerequisite stages) -- if a prerequisite is missing, this raises a clear
error rather than silently doing partial work or invoking an earlier stage
itself (project notes section 34: "Do NOT make the user re-run earlier
stages" refers to already-completed prerequisites being read as-is, not to
Stage 5 re-triggering them).
"""
from __future__ import annotations

from exampapersorter.database import Database
from exampapersorter.frequency_analysis.aggregate import build_frequency_analysis
from exampapersorter.schemas import FrequencyAnalysisSummary


class FrequencyAnalysisPrerequisiteError(RuntimeError):
    """Raised when a required Stage 1-4 prerequisite has not been run yet
    for the given textbook -- caller (cli.py) is expected to report this
    and exit cleanly rather than attempting partial aggregation."""


def run_frequency_analysis(db: Database, textbook_file_hash: str) -> FrequencyAnalysisSummary:
    topics_result = db.get_latest_topic_extraction(textbook_file_hash)
    if topics_result is None:
        raise FrequencyAnalysisPrerequisiteError(
            "No topic list found for this textbook. Run 'extract-topics' (Stage 1) first."
        )

    questions = db.get_all_questions()
    if not questions:
        raise FrequencyAnalysisPrerequisiteError(
            "No questions found in the database. Run 'extract-questions' (Stage 2) first -- "
            "Stage 5 aggregates already-extracted question occurrences, it does not extract from PDFs itself."
        )

    canonical_questions = db.list_canonical_questions()
    if not canonical_questions:
        raise FrequencyAnalysisPrerequisiteError(
            "No canonical questions found in the database. Run 'deduplicate' (Stage 4) first -- "
            "Stage 5 aggregates over Stage 4's canonical groups, it does not deduplicate itself."
        )

    papers = db.get_all_papers()

    return build_frequency_analysis(
        topics=topics_result.topics,
        questions=questions,
        canonical_questions=canonical_questions,
        papers=papers,
        textbook_file_hash=textbook_file_hash,
    )
