import pytest

from exampapersorter.database import Database
from exampapersorter.frequency_analysis.pipeline import FrequencyAnalysisPrerequisiteError, run_frequency_analysis
from exampapersorter.schemas import (
    CanonicalQuestion,
    Question,
    Topic,
    TopicExtractionResult,
)


def t(id, level=1, parent_id=None, name=None):
    return Topic(id=id, name=name or id, level=level, parent_id=parent_id, source_pages=[10])


def q(question_id, paper_id="p1", topic_id=None, status="unclassified", qtype="short_answer"):
    return Question(
        question_id=question_id, paper_id=paper_id, section_id=None,
        source_filename="paper.pdf", source_file_hash="hash1",
        question_number="1", question_text="Explain X.", question_type=qtype,
        source_pages=[1], extraction_confidence=0.9,
        topic_id=topic_id, classification_status=status,
    )


def test_run_frequency_analysis_fails_clearly_when_no_topics(tmp_path):
    db = Database(tmp_path / "t.db")
    with pytest.raises(FrequencyAnalysisPrerequisiteError, match="extract-topics"):
        run_frequency_analysis(db, "missing_hash")
    db.close()


def test_run_frequency_analysis_fails_clearly_when_no_questions(tmp_path):
    db = Database(tmp_path / "t.db")
    db.save_topic_extraction_run(
        "hash1", "model", 1, 10, status="success",
        topics=TopicExtractionResult(topics=[t("chapter_a")]), validation=None,
    )
    with pytest.raises(FrequencyAnalysisPrerequisiteError, match="extract-questions"):
        run_frequency_analysis(db, "hash1")
    db.close()


def test_run_frequency_analysis_fails_clearly_when_no_canonical_questions(tmp_path):
    db = Database(tmp_path / "t.db")
    db.save_topic_extraction_run(
        "hash1", "model", 1, 10, status="success",
        topics=TopicExtractionResult(topics=[t("chapter_a")]), validation=None,
    )
    db.save_questions([q("q1")])
    with pytest.raises(FrequencyAnalysisPrerequisiteError, match="deduplicate"):
        run_frequency_analysis(db, "hash1")
    db.close()


def test_run_frequency_analysis_end_to_end_against_real_database(tmp_path):
    db = Database(tmp_path / "t.db")
    db.save_topic_extraction_run(
        "hash1", "model", 1, 10, status="success",
        topics=TopicExtractionResult(topics=[t("chapter_a"), t("chapter_b")]), validation=None,
    )
    db.save_questions([
        q("q1", topic_id="chapter_a", status="classified"),
        q("q2", topic_id="chapter_a", status="classified"),
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

    summary = run_frequency_analysis(db, "hash1")
    assert summary.total_question_occurrences == 3
    assert summary.total_canonical_questions == 2
    assert summary.topic_count == 2
    assert summary.data_integrity.reconciled is True
    db.close()


def test_run_frequency_analysis_is_idempotent_across_two_runs(tmp_path):
    db = Database(tmp_path / "t.db")
    db.save_topic_extraction_run(
        "hash1", "model", 1, 10, status="success",
        topics=TopicExtractionResult(topics=[t("chapter_a")]), validation=None,
    )
    db.save_questions([q("q1", topic_id="chapter_a", status="classified")])
    db.replace_canonical_state(
        [CanonicalQuestion(
            canonical_question_id="c1", canonical_question_text="Explain X.",
            question_type="short_answer", representative_question_id="q1", dedup_status="singleton",
        )],
        {"q1": "c1"},
    )

    first = run_frequency_analysis(db, "hash1")
    second = run_frequency_analysis(db, "hash1")
    assert first.model_dump() == second.model_dump()
    db.close()
