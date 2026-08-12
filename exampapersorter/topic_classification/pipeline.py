"""Orchestrates Stage 3 end to end: classify every already-extracted
question (Stage 2 output, already persisted) against a textbook's topic
list (Stage 1 output, already persisted) -- one LLM call per exam paper,
resumable at paper granularity via topic_classification_log, mirroring
Stage 2's own caching pattern (see question_extraction/pipeline.py).

Deliberately does NOT re-run Stage 1 or Stage 2 -- both are read from the
database as already-completed prerequisites. If either hasn't been run yet
for the given textbook/paper, this fails clearly (see cli.py) rather than
attempting to backfill it here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from exampapersorter.config import Config
from exampapersorter.database import Database
from exampapersorter.llm_client import LLMCallFailed, get_last_call_metadata
from exampapersorter.schemas import Paper, Topic
from exampapersorter.topic_classification.classify import classify_questions
from exampapersorter.topic_classification.reconciliation import reconcile_classifications

logger = logging.getLogger(__name__)


def _call_metadata() -> dict | None:
    meta = get_last_call_metadata()
    return meta.to_dict() if meta else None


@dataclass
class PaperClassificationResult:
    paper_id: str
    status: str  # success | failed
    total_questions: int = 0
    classified_count: int = 0
    no_match_count: int = 0
    ambiguous_count: int = 0
    unclassified_count: int = 0
    error_message: str | None = None


def classify_paper(
    paper: Paper, topics: list[Topic], textbook_file_hash: str, config: Config, db: Database
) -> PaperClassificationResult:
    questions = db.get_questions_for_paper(paper.paper_id)
    if not questions:
        return PaperClassificationResult(paper_id=paper.paper_id, status="success", total_questions=0)

    result = db.get_topic_classification_verdict(paper.paper_id, config.model_identifier)
    if result is None:
        try:
            result = classify_questions(questions, topics, config)
        except LLMCallFailed as exc:
            if exc.quota_exhausted:
                raise
            logger.error("Topic classification failed for %s: %s", paper.paper_id, exc)
            return PaperClassificationResult(
                paper_id=paper.paper_id, status="failed", total_questions=len(questions), error_message=str(exc),
            )
        db.save_topic_classification_verdict(
            paper.paper_id, textbook_file_hash, config.model_identifier, result, call_metadata=_call_metadata(),
        )
    else:
        logger.debug("Using cached topic classification verdict for %s", paper.paper_id)

    reconciled = reconcile_classifications(questions, topics, result)
    db.save_question_classifications(reconciled, {t.id: t for t in topics})

    counts = {"classified": 0, "no_match": 0, "ambiguous": 0, "unclassified": 0}
    for c in reconciled:
        counts[c.status] += 1

    return PaperClassificationResult(
        paper_id=paper.paper_id, status="success", total_questions=len(questions),
        classified_count=counts["classified"], no_match_count=counts["no_match"],
        ambiguous_count=counts["ambiguous"], unclassified_count=counts["unclassified"],
    )
