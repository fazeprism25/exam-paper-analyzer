"""Orchestrates Stage 6 end to end: runs Stage 5's own aggregation (pure,
deterministic, zero LLM/embedding calls -- see frequency_analysis/
pipeline.py) against already-persisted Stage 1-4 records, then hands the
result plus a small amount of additional read-only hydration (Stage 4's
canonical_questions for paper-level provenance, Stage 2's questions for
no_match/ambiguous full text -- both already-persisted, deterministic reads)
to final_analysis.build's pure functions.

Deliberately does NOT re-run Stages 1-4, and does not require Stage 5's own
CLI command (`analyze-frequency`) to have been run first -- run_frequency_
analysis is itself pure aggregation with no LLM/embedding cost, so Stage 6
simply invokes it directly rather than requiring a separate prior run and
risking a stale summary.json. Reuses run_frequency_analysis's own
FrequencyAnalysisPrerequisiteError so cli.py's error handling is identical
to Stage 5's.
"""
from __future__ import annotations

from exampapersorter.database import Database
from exampapersorter.final_analysis.build import build_final_report
from exampapersorter.frequency_analysis.pipeline import FrequencyAnalysisPrerequisiteError, run_frequency_analysis
from exampapersorter.schemas import FinalAnalysisReport

__all__ = ["FrequencyAnalysisPrerequisiteError", "run_final_analysis"]


def run_final_analysis(db: Database, textbook_file_hash: str) -> FinalAnalysisReport:
    summary = run_frequency_analysis(db, textbook_file_hash)  # raises FrequencyAnalysisPrerequisiteError
    canonical_questions = db.list_canonical_questions()
    questions = db.get_all_questions()
    return build_final_report(summary, canonical_questions, questions)
