"""Tests for final_analysis/pipeline.py -- mirrors tests/
test_frequency_analysis_pipeline.py's style. Covers prerequisite failure
(same errors as Stage 5, since Stage 6 runs Stage 5's own aggregation),
end-to-end success against a real Database, determinism across repeated
runs, and that no LLM call machinery is ever touched.
"""
import pytest

from exampapersorter.database import Database
from exampapersorter.final_analysis.pipeline import FrequencyAnalysisPrerequisiteError, run_final_analysis
from exampapersorter.llm_client import get_call_metrics, reset_call_metrics
from exampapersorter.schemas import CanonicalQuestion, Question, Topic, TopicExtractionResult


def t(id, level=1, parent_id=None, name=None):
    return Topic(id=id, name=name or id, level=level, parent_id=parent_id, source_pages=[10])


def q(question_id, paper_id="p1", topic_id=None, status="unclassified", qtype="short_answer", paper_year=None):
    return Question(
        question_id=question_id, paper_id=paper_id, section_id=None,
        source_filename="paper.pdf", source_file_hash="hash1", paper_year=paper_year,
        question_number="1", question_text="Explain X.", question_type=qtype,
        source_pages=[1], extraction_confidence=0.9,
        topic_id=topic_id, classification_status=status,
    )


def test_run_final_analysis_fails_clearly_when_no_topics(tmp_path):
    db = Database(tmp_path / "t.db")
    with pytest.raises(FrequencyAnalysisPrerequisiteError, match="extract-topics"):
        run_final_analysis(db, "missing_hash")
    db.close()


def test_run_final_analysis_fails_clearly_when_no_questions(tmp_path):
    db = Database(tmp_path / "t.db")
    db.save_topic_extraction_run(
        "hash1", "model", 1, 10, status="success",
        topics=TopicExtractionResult(topics=[t("chapter_a")]), validation=None,
    )
    with pytest.raises(FrequencyAnalysisPrerequisiteError, match="extract-questions"):
        run_final_analysis(db, "hash1")
    db.close()


def test_run_final_analysis_fails_clearly_when_no_canonical_questions(tmp_path):
    db = Database(tmp_path / "t.db")
    db.save_topic_extraction_run(
        "hash1", "model", 1, 10, status="success",
        topics=TopicExtractionResult(topics=[t("chapter_a")]), validation=None,
    )
    db.save_questions([q("q1")])
    with pytest.raises(FrequencyAnalysisPrerequisiteError, match="deduplicate"):
        run_final_analysis(db, "hash1")
    db.close()


def _populate(db):
    db.save_topic_extraction_run(
        "hash1", "model", 1, 10, status="success",
        topics=TopicExtractionResult(topics=[t("chapter_a"), t("chapter_b")]), validation=None,
    )
    db.save_questions([
        q("q1", topic_id="chapter_a", status="classified", paper_year="2022"),
        q("q2", topic_id="chapter_a", status="classified", paper_year="2022"),
        q("q3", topic_id="chapter_b", status="classified"),
    ])
    db.replace_canonical_state(
        [
            CanonicalQuestion(
                canonical_question_id="c1", canonical_question_text="Explain X.",
                question_type="short_answer", representative_question_id="q1",
                dedup_status="exact_duplicate", dedup_confidence=1.0,
            ),
            CanonicalQuestion(
                canonical_question_id="c2", canonical_question_text="Explain X.",
                question_type="short_answer", representative_question_id="q3",
                dedup_status="singleton",
            ),
        ],
        {"q1": "c1", "q2": "c1", "q3": "c2"},
    )


def test_run_final_analysis_end_to_end_against_real_database(tmp_path):
    db = Database(tmp_path / "t.db")
    _populate(db)

    report = run_final_analysis(db, "hash1")
    assert report.summary.total_question_occurrences == 3
    assert report.summary.total_canonical_questions == 2
    assert report.summary.total_topics == 2
    assert report.summary.data_integrity_reconciled is True
    assert len(report.most_repeated_questions) == 1
    assert report.most_repeated_questions[0].canonical_question_id == "c1"
    assert report.most_repeated_questions[0].occurrence_count == 2
    assert report.most_repeated_questions[0].paper_ids == ["p1"]
    db.close()


def test_run_final_analysis_is_idempotent_across_two_runs(tmp_path):
    db = Database(tmp_path / "t.db")
    _populate(db)

    first = run_final_analysis(db, "hash1")
    second = run_final_analysis(db, "hash1")
    assert first.model_dump() == second.model_dump()
    db.close()


def test_run_final_analysis_makes_no_llm_calls(tmp_path):
    """Stage 6 must be zero-cost: no LLM calls, no embedding calls. There's
    no embedding-call counter to check (fastembed calls aren't tracked by
    llm_client's metrics), but Stage 6's code path never imports or calls
    anything in llm_client/llm_providers/deduplication.embeddings -- this
    asserts the one machine-checkable half of that (LLM call metrics stay
    at zero) directly."""
    reset_call_metrics()
    db = Database(tmp_path / "t.db")
    _populate(db)
    run_final_analysis(db, "hash1")
    db.close()

    metrics = get_call_metrics()
    assert metrics.total_calls == 0
    assert metrics.total_attempts == 0
