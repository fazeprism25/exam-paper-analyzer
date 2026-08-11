from exampapersorter.deduplication.semantic_judge import _render_pairs, _render_question
from exampapersorter.schemas import Question


def q(question_id, text, qtype="short_answer", options=None, topic_name=None):
    return Question(
        question_id=question_id, paper_id="p1", section_id=None,
        source_filename="paper.pdf", source_file_hash="hash1",
        question_number="1", question_text=text, question_type=qtype,
        options=options or [], source_pages=[1], extraction_confidence=0.9,
        topic_name=topic_name,
    )


def test_render_question_includes_type_and_text():
    rendered = _render_question(q("q1", "Explain glycolysis.", qtype="long_answer"))
    assert "type=long_answer" in rendered
    assert "Explain glycolysis." in rendered


def test_render_question_includes_options_for_mcq():
    rendered = _render_question(q("q1", "Which enzyme?", qtype="mcq", options=["Hexokinase", "Pepsin"]))
    assert "A. Hexokinase" in rendered
    assert "B. Pepsin" in rendered


def test_render_question_omits_topic_line_when_absent():
    rendered = _render_question(q("q1", "Explain glycolysis."))
    assert "topic=" not in rendered


def test_render_question_includes_topic_when_present_as_context_only():
    rendered = _render_question(q("q1", "Explain glycolysis.", topic_name="Carbohydrate Metabolism"))
    assert "topic=Carbohydrate Metabolism" in rendered


def test_render_pairs_labels_each_pair_and_side():
    qa = q("q1", "Explain glycolysis.")
    qb = q("q2", "Describe glycolysis.")
    rendered = _render_pairs([("q1::q2", qa, qb)])
    assert "[q1::q2]" in rendered
    assert "Question A (q1)" in rendered
    assert "Question B (q2)" in rendered


def test_render_pairs_joins_multiple_pairs_with_a_separator():
    qa, qb, qc, qd = q("q1", "A"), q("q2", "B"), q("q3", "C"), q("q4", "D")
    rendered = _render_pairs([("q1::q2", qa, qb), ("q3::q4", qc, qd)])
    assert "[q1::q2]" in rendered and "[q3::q4]" in rendered
    assert rendered.index("[q1::q2]") < rendered.index("[q3::q4]")
