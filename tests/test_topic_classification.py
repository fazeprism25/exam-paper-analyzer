import pytest
from pydantic import ValidationError

from exampapersorter.schemas import Question, QuestionTopicClassification, Topic, TopicClassificationResult
from exampapersorter.topic_classification.classify import _render_questions, _render_topics
from exampapersorter.topic_classification.reconciliation import reconcile_classifications


def topic(id_, name, level=1, parent_id=None):
    return Topic(id=id_, name=name, level=level, parent_id=parent_id, source_pages=[1])


def question(question_id, text="What is glycolysis?", qtype="short_answer", paper_id="p1"):
    return Question(
        question_id=question_id, paper_id=paper_id, section_id=None,
        source_filename="paper.pdf", source_file_hash="hash1",
        question_number="1", question_text=text, question_type=qtype,
        source_pages=[1], extraction_confidence=0.9,
    )


# --- QuestionTopicClassification schema-level validator ---


def test_classified_without_topic_id_is_rejected():
    with pytest.raises(ValidationError):
        QuestionTopicClassification(question_id="q1", status="classified", topic_id=None, confidence=0.8)


def test_no_match_with_topic_id_is_rejected():
    with pytest.raises(ValidationError):
        QuestionTopicClassification(question_id="q1", status="no_match", topic_id="t1", confidence=0.8)


def test_classified_with_topic_id_is_valid():
    c = QuestionTopicClassification(question_id="q1", status="classified", topic_id="t1", confidence=0.9)
    assert c.topic_id == "t1"


def test_unclassified_is_a_valid_status_for_reconciled_records():
    # Never produced by the LLM itself (see ClassificationVerdict), but
    # reconciliation.py builds these -- the model must accept them.
    c = QuestionTopicClassification(question_id="q1", status="unclassified", topic_id=None, confidence=0.0)
    assert c.status == "unclassified"


# --- reconciliation ---


def test_reconcile_passes_through_valid_classified_verdict():
    topics = [topic("t1", "Lipids")]
    questions = [question("q1")]
    result = TopicClassificationResult(
        classifications=[QuestionTopicClassification(question_id="q1", status="classified", topic_id="t1", confidence=0.9)]
    )
    reconciled = reconcile_classifications(questions, topics, result)
    assert len(reconciled) == 1
    assert reconciled[0].status == "classified"
    assert reconciled[0].topic_id == "t1"


def test_reconcile_downgrades_invented_topic_id_to_unclassified():
    topics = [topic("t1", "Lipids")]
    questions = [question("q1")]
    result = TopicClassificationResult(
        classifications=[
            QuestionTopicClassification(question_id="q1", status="classified", topic_id="t_invented", confidence=0.9)
        ]
    )
    reconciled = reconcile_classifications(questions, topics, result)
    assert reconciled[0].status == "unclassified"
    assert reconciled[0].topic_id is None
    assert "t_invented" in reconciled[0].reason


def test_reconcile_fills_gap_for_question_missing_from_response():
    topics = [topic("t1", "Lipids")]
    questions = [question("q1"), question("q2")]
    result = TopicClassificationResult(
        classifications=[QuestionTopicClassification(question_id="q1", status="classified", topic_id="t1", confidence=0.9)]
    )
    reconciled = reconcile_classifications(questions, topics, result)
    assert len(reconciled) == 2
    by_id = {c.question_id: c for c in reconciled}
    assert by_id["q1"].status == "classified"
    assert by_id["q2"].status == "unclassified"


def test_reconcile_drops_verdicts_for_unknown_question_ids():
    topics = [topic("t1", "Lipids")]
    questions = [question("q1")]
    result = TopicClassificationResult(
        classifications=[
            QuestionTopicClassification(question_id="q1", status="classified", topic_id="t1", confidence=0.9),
            QuestionTopicClassification(question_id="q_not_asked_about", status="classified", topic_id="t1", confidence=0.9),
        ]
    )
    reconciled = reconcile_classifications(questions, topics, result)
    assert len(reconciled) == 1
    assert reconciled[0].question_id == "q1"


def test_reconcile_passes_through_no_match_and_ambiguous_verdicts_unchanged():
    topics = [topic("t1", "Lipids")]
    questions = [question("q1"), question("q2")]
    result = TopicClassificationResult(
        classifications=[
            QuestionTopicClassification(question_id="q1", status="no_match", topic_id=None, confidence=0.7),
            QuestionTopicClassification(question_id="q2", status="ambiguous", topic_id=None, confidence=0.5, reason="could be t1 or t2"),
        ]
    )
    reconciled = reconcile_classifications(questions, topics, result)
    by_id = {c.question_id: c for c in reconciled}
    assert by_id["q1"].status == "no_match"
    assert by_id["q2"].status == "ambiguous"


# --- prompt rendering ---


def test_render_topics_shows_hierarchy_with_indentation():
    topics = [
        topic("t1", "Section One", level=1),
        topic("t1_1", "Lipids", level=2, parent_id="t1"),
    ]
    rendered = _render_topics(topics)
    assert "[t1] Section One" in rendered
    assert "  - [t1_1] Lipids" in rendered


def test_render_questions_includes_options_for_mcq():
    q = Question(
        question_id="q1", paper_id="p1", section_id=None,
        source_filename="paper.pdf", source_file_hash="hash1",
        question_number="1", question_text="Which enzyme?", question_type="mcq",
        options=["Hexokinase", "Pepsin"], source_pages=[1], extraction_confidence=0.9,
    )
    rendered = _render_questions([q])
    assert "[q1]" in rendered
    assert "A. Hexokinase" in rendered
    assert "B. Pepsin" in rendered
