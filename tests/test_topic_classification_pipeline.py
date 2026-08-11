from dataclasses import replace

from exampapersorter.config import DEFAULT_CONFIG
from exampapersorter.database import Database
from exampapersorter.llm_client import LLMCallFailed
from exampapersorter.schemas import (
    MetadataFieldValue,
    Paper,
    PaperMetadata,
    Question,
    QuestionTopicClassification,
    Topic,
    TopicClassificationResult,
)
from exampapersorter.topic_classification import pipeline as pipeline_module

TEXTBOOK_HASH = "textbook_hash_1"


def empty_metadata():
    f = MetadataFieldValue()
    return PaperMetadata(exam_name=f, institution=f, subject=f, date=f, year=f, paper_identifier=f)


def make_paper(paper_id="p1"):
    return Paper(
        paper_id=paper_id, file_hash="qhash1", start_page=1, end_page=2, metadata=empty_metadata(),
        sections=[], boundary_confidence=0.9, boundary_evidence=[], status="success", notes=[],
    )


def make_question(question_id, paper_id="p1"):
    return Question(
        question_id=question_id, paper_id=paper_id, section_id=None,
        source_filename="paper.pdf", source_file_hash="qhash1",
        question_number="1", question_text="Explain glycolysis.", question_type="short_answer",
        source_pages=[1], extraction_confidence=0.9,
    )


def topic(id_, name):
    return Topic(id=id_, name=name, level=1, parent_id=None, source_pages=[1])


def test_classify_paper_with_no_questions_is_a_success_no_op(tmp_path, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(pipeline_module, "classify_questions", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))
    db = Database(tmp_path / "test.db")
    result = pipeline_module.classify_paper(make_paper(), [topic("t1", "Lipids")], TEXTBOOK_HASH, DEFAULT_CONFIG, db)

    assert result.status == "success"
    assert result.total_questions == 0
    assert calls["n"] == 0  # never called the LLM for an empty paper
    db.close()


def test_classify_paper_calls_llm_and_persists_classification(tmp_path, monkeypatch):
    db = Database(tmp_path / "test.db")
    db.save_questions([make_question("p1_q1")])

    def fake_classify_questions(questions, topics, config):
        return TopicClassificationResult(
            classifications=[
                QuestionTopicClassification(question_id="p1_q1", status="classified", topic_id="t1", confidence=0.9)
            ]
        )

    monkeypatch.setattr(pipeline_module, "classify_questions", fake_classify_questions)

    result = pipeline_module.classify_paper(make_paper(), [topic("t1", "Lipids")], TEXTBOOK_HASH, DEFAULT_CONFIG, db)

    assert result.status == "success"
    assert result.classified_count == 1
    loaded = db.get_questions_for_paper("p1")[0]
    assert loaded.topic_id == "t1"
    assert loaded.topic_name == "Lipids"
    assert loaded.classification_status == "classified"
    db.close()


def test_classify_paper_uses_cached_verdict_on_resume(tmp_path, monkeypatch):
    db = Database(tmp_path / "test.db")
    db.save_questions([make_question("p1_q1")])
    calls = {"n": 0}

    def fake_classify_questions(questions, topics, config):
        calls["n"] += 1
        return TopicClassificationResult(
            classifications=[
                QuestionTopicClassification(question_id="p1_q1", status="classified", topic_id="t1", confidence=0.9)
            ]
        )

    monkeypatch.setattr(pipeline_module, "classify_questions", fake_classify_questions)
    topics = [topic("t1", "Lipids")]

    pipeline_module.classify_paper(make_paper(), topics, TEXTBOOK_HASH, DEFAULT_CONFIG, db)
    assert calls["n"] == 1

    pipeline_module.classify_paper(make_paper(), topics, TEXTBOOK_HASH, DEFAULT_CONFIG, db)
    assert calls["n"] == 1  # resumed run reused the cached verdict, no new LLM call
    db.close()


def test_classify_paper_llm_failure_is_reported_without_crashing(tmp_path, monkeypatch):
    db = Database(tmp_path / "test.db")
    db.save_questions([make_question("p1_q1")])

    def fake_classify_questions(questions, topics, config):
        raise LLMCallFailed("malformed output, retries exhausted")

    monkeypatch.setattr(pipeline_module, "classify_questions", fake_classify_questions)

    result = pipeline_module.classify_paper(make_paper(), [topic("t1", "Lipids")], TEXTBOOK_HASH, DEFAULT_CONFIG, db)

    assert result.status == "failed"
    assert result.error_message is not None
    # the question's classification_status is untouched -- still unclassified, not silently marked otherwise
    assert db.get_questions_for_paper("p1")[0].classification_status == "unclassified"
    db.close()


def test_classify_paper_downgrades_hallucinated_topic_id_before_persisting(tmp_path, monkeypatch):
    db = Database(tmp_path / "test.db")
    db.save_questions([make_question("p1_q1")])

    def fake_classify_questions(questions, topics, config):
        return TopicClassificationResult(
            classifications=[
                QuestionTopicClassification(question_id="p1_q1", status="classified", topic_id="t_invented", confidence=0.9)
            ]
        )

    monkeypatch.setattr(pipeline_module, "classify_questions", fake_classify_questions)

    result = pipeline_module.classify_paper(make_paper(), [topic("t1", "Lipids")], TEXTBOOK_HASH, DEFAULT_CONFIG, db)

    assert result.unclassified_count == 1
    assert result.classified_count == 0
    loaded = db.get_questions_for_paper("p1")[0]
    assert loaded.classification_status == "unclassified"
    assert loaded.topic_id is None
    db.close()
