from exampapersorter.deduplication.normalize import exact_match_key, normalize_for_exact_match
from exampapersorter.schemas import Question


def q(question_id, text, qtype="short_answer", options=None):
    return Question(
        question_id=question_id, paper_id="p1", section_id=None,
        source_filename="paper.pdf", source_file_hash="hash1",
        question_number="1", question_text=text, question_type=qtype,
        options=options or [], source_pages=[1], extraction_confidence=0.9,
    )


# --- normalize_for_exact_match ---


def test_identical_text_normalizes_identically():
    assert normalize_for_exact_match("Explain glycolysis.") == normalize_for_exact_match("Explain glycolysis.")


def test_whitespace_variation_normalizes_the_same():
    assert normalize_for_exact_match("Explain   glycolysis.") == normalize_for_exact_match("Explain glycolysis.")
    assert normalize_for_exact_match("  Explain glycolysis.  ") == normalize_for_exact_match("Explain glycolysis.")


def test_trailing_period_variation_normalizes_the_same():
    assert normalize_for_exact_match("Explain glycolysis") == normalize_for_exact_match("Explain glycolysis.")


def test_case_variation_normalizes_the_same():
    assert normalize_for_exact_match("EXPLAIN GLYCOLYSIS.") == normalize_for_exact_match("explain glycolysis.")


def test_curly_quote_variation_normalizes_the_same():
    assert normalize_for_exact_match("Explain the body’s response.") == normalize_for_exact_match(
        "Explain the body's response."
    )


def test_question_mark_is_not_stripped_and_changes_the_result():
    # Only a trailing PERIOD is stripped -- a trailing "?" can be
    # semantically meaningful and is never harmonized away.
    assert normalize_for_exact_match("Explain glycolysis?") != normalize_for_exact_match("Explain glycolysis.")


def test_genuinely_different_text_normalizes_differently():
    assert normalize_for_exact_match("Explain glycolysis.") != normalize_for_exact_match("Explain glycogenolysis.")


# --- exact_match_key ---


def test_exact_match_key_identical_for_whitespace_punctuation_variants():
    a = q("q1", "Explain glycolysis.")
    b = q("q2", "Explain   glycolysis")
    assert exact_match_key(a) == exact_match_key(b)


def test_exact_match_key_differs_across_question_types():
    # Byte-identical text but a different question_type is treated as a
    # Stage 2 anomaly, not a confirmed exact duplicate -- see candidates.py.
    a = q("q1", "Explain glycolysis.", qtype="short_answer")
    b = q("q2", "Explain glycolysis.", qtype="long_answer")
    assert exact_match_key(a) != exact_match_key(b)


def test_exact_match_key_differs_when_mcq_options_differ():
    a = q("q1", "Which enzyme regulates glycolysis?", qtype="mcq", options=["Hexokinase", "Pepsin"])
    b = q("q2", "Which enzyme regulates glycolysis?", qtype="mcq", options=["Hexokinase", "Amylase"])
    assert exact_match_key(a) != exact_match_key(b)


def test_exact_match_key_identical_when_mcq_options_match():
    a = q("q1", "Which enzyme regulates glycolysis?", qtype="mcq", options=["Hexokinase", "Pepsin"])
    b = q("q2", "Which enzyme regulates glycolysis?", qtype="mcq", options=["Hexokinase", "Pepsin"])
    assert exact_match_key(a) == exact_match_key(b)
