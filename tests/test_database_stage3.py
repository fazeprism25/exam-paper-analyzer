from exampapersorter.database import Database
from exampapersorter.schemas import (
    Question,
    QuestionTopicClassification,
    Topic,
    TopicClassificationResult,
    TopicExtractionResult,
    ValidationReport,
)


def topic(id_, name, level=1, parent_id=None):
    return Topic(id=id_, name=name, level=level, parent_id=parent_id, source_pages=[1])


def question(question_id, paper_id="p1"):
    return Question(
        question_id=question_id, paper_id=paper_id, section_id=None,
        source_filename="paper.pdf", source_file_hash="hash1",
        question_number="1", question_text="Explain glycolysis.", question_type="short_answer",
        source_pages=[1], extraction_confidence=0.9,
    )


def test_new_questions_default_to_unclassified(tmp_path):
    db = Database(tmp_path / "test.db")
    db.save_questions([question("p1_q1")])
    loaded = db.get_questions_for_paper("p1")
    assert loaded[0].classification_status == "unclassified"
    assert loaded[0].topic_id is None
    assert loaded[0].topic_name is None
    assert loaded[0].topic_confidence is None
    db.close()


def test_get_latest_topic_extraction_returns_none_when_nothing_saved(tmp_path):
    db = Database(tmp_path / "test.db")
    assert db.get_latest_topic_extraction("hash1") is None
    db.close()


def test_get_latest_topic_extraction_skips_failed_runs_and_returns_most_recent(tmp_path):
    db = Database(tmp_path / "test.db")
    db.save_topic_extraction_run("hash1", "modelA", 1, 10, status="failed", topics=None, validation=None)
    first = TopicExtractionResult(topics=[topic("t1", "Lipids")])
    db.save_topic_extraction_run("hash1", "modelA", 1, 10, status="success", topics=first, validation=ValidationReport(passed=True, topic_count=1, issues=[]))
    second = TopicExtractionResult(topics=[topic("t1", "Lipids"), topic("t2", "Proteins")])
    db.save_topic_extraction_run("hash1", "modelA", 1, 10, status="success", topics=second, validation=ValidationReport(passed=True, topic_count=2, issues=[]))

    loaded = db.get_latest_topic_extraction("hash1")
    assert loaded == second
    db.close()


def test_topic_classification_verdict_cache_round_trips(tmp_path):
    db = Database(tmp_path / "test.db")
    result = TopicClassificationResult(
        classifications=[QuestionTopicClassification(question_id="p1_q1", status="classified", topic_id="t1", confidence=0.9)]
    )
    assert db.get_topic_classification_verdict("p1", "modelA") is None
    db.save_topic_classification_verdict("p1", "hash_textbook", "modelA", result)
    loaded = db.get_topic_classification_verdict("p1", "modelA")
    assert loaded == result
    db.close()


def test_save_question_classifications_updates_persisted_question_rows(tmp_path):
    db = Database(tmp_path / "test.db")
    db.save_questions([question("p1_q1"), question("p1_q2")])
    topics_by_id = {"t1": topic("t1", "Lipids")}

    classifications = [
        QuestionTopicClassification(question_id="p1_q1", status="classified", topic_id="t1", confidence=0.85),
        QuestionTopicClassification(question_id="p1_q2", status="unclassified", topic_id=None, confidence=0.0, reason="no verdict"),
    ]
    db.save_question_classifications(classifications, topics_by_id)

    loaded = {q.question_id: q for q in db.get_questions_for_paper("p1")}
    assert loaded["p1_q1"].topic_id == "t1"
    assert loaded["p1_q1"].topic_name == "Lipids"
    assert loaded["p1_q1"].topic_confidence == 0.85
    assert loaded["p1_q1"].classification_status == "classified"

    assert loaded["p1_q2"].topic_id is None
    assert loaded["p1_q2"].topic_name is None
    assert loaded["p1_q2"].topic_confidence is None
    assert loaded["p1_q2"].classification_status == "unclassified"
    db.close()


def test_rerunning_save_questions_for_a_fresh_extraction_resets_classification(tmp_path):
    """A paper that gets genuinely re-extracted (not just resumed) starts
    its classification over -- save_questions always writes the Question
    object's own classification_status, which is "unclassified" unless the
    caller explicitly constructed it otherwise."""
    db = Database(tmp_path / "test.db")
    db.save_questions([question("p1_q1")])
    db.save_question_classifications(
        [QuestionTopicClassification(question_id="p1_q1", status="classified", topic_id="t1", confidence=0.9)],
        {"t1": topic("t1", "Lipids")},
    )
    assert db.get_questions_for_paper("p1")[0].classification_status == "classified"

    db.save_questions([question("p1_q1")])  # fresh Question(), defaults to unclassified
    assert db.get_questions_for_paper("p1")[0].classification_status == "unclassified"
    db.close()
