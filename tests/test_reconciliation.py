from exampapersorter.question_extraction.reconciliation import reconcile_extracted_questions
from exampapersorter.schemas import EvidenceBlock, ExtractedQuestion


def block(block_type, text, marker=None, pages=(3,)):
    return EvidenceBlock(page_numbers=list(pages), block_type=block_type, text=text, marker=marker)


def q(number=None, text="", qtype="short_answer", options=None):
    return ExtractedQuestion(
        question_number=number, question_text=text, question_type=qtype,
        options=options or [], source_pages=[3], extraction_confidence=0.9,
    )


# --- MCQ options backfilled from an mcq_group block ---


def test_mcq_options_are_backfilled_from_mcq_group_block():
    blocks = [
        block(
            "mcq_group",
            "STEM: 2. Principal cation of the extracellular fluid\n"
            "OPTION: A) Sodium\nOPTION: B) Potassium\nOPTION: C) Chloride\nOPTION: D) Magnesium",
        )
    ]
    questions = [q(number=None, text="Principal cation of the extracellular fluid", qtype="mcq", options=[])]
    result = reconcile_extracted_questions(blocks, questions)
    assert result[0].question_number == "2"
    assert result[0].options == ["A) Sodium", "B) Potassium", "C) Chloride", "D) Magnesium"]


def test_mcq_options_already_populated_are_left_untouched():
    blocks = [
        block("mcq_group", "STEM: 1. Some stem\nOPTION: A) Wrong\nOPTION: B) Also wrong")
    ]
    questions = [q(number="1", text="Some stem", qtype="mcq", options=["A) Real", "B) Options"])]
    result = reconcile_extracted_questions(blocks, questions)
    assert result[0].options == ["A) Real", "B) Options"]


# --- question_number backfilled from a plain numbered block ---


def test_question_number_backfilled_from_marker():
    blocks = [block("list_item", "Explain the Cori cycle.", marker="3.")]
    questions = [q(number=None, text="Explain the Cori cycle.")]
    result = reconcile_extracted_questions(blocks, questions)
    assert result[0].question_number == "3"


def test_question_number_backfilled_from_leading_text_when_no_marker():
    blocks = [block("text", "9. The renal function test that may be used")]
    questions = [q(number=None, text="The renal function test that may be used")]
    result = reconcile_extracted_questions(blocks, questions)
    assert result[0].question_number == "9"


def test_already_populated_question_number_is_never_overridden():
    blocks = [block("list_item", "Explain the Cori cycle.", marker="3.")]
    questions = [q(number="3a", text="Explain the Cori cycle.")]
    result = reconcile_extracted_questions(blocks, questions)
    assert result[0].question_number == "3a"


# --- never invents: no confident match -> left alone ---


def test_no_matching_block_leaves_number_null():
    blocks = [block("list_item", "Something completely unrelated.", marker="1.")]
    questions = [q(number=None, text="A totally different question about lipids.")]
    result = reconcile_extracted_questions(blocks, questions)
    assert result[0].question_number is None


def test_ambiguous_match_to_two_candidates_is_not_guessed():
    """Two candidate blocks whose snippets both plausibly match the same
    short question text -- deliberately left unresolved rather than
    picking one arbitrarily."""
    blocks = [
        block("list_item", "Define Basal Metabolic Rate and its uses.", marker="2."),
        block("list_item", "Define Basal Metabolic Rate and its clinical relevance.", marker="7."),
    ]
    questions = [q(number=None, text="Define Basal Metabolic Rate and its")]
    result = reconcile_extracted_questions(blocks, questions)
    assert result[0].question_number is None


def test_a_candidate_is_only_used_once():
    """Two questions that both start similarly must not both claim the same
    single evidence candidate."""
    blocks = [block("list_item", "Explain glycolysis in detail.", marker="4.")]
    questions = [
        q(number=None, text="Explain glycolysis in detail."),
        q(number=None, text="Explain glycolysis in detail, step by step with enzymes named."),
    ]
    result = reconcile_extracted_questions(blocks, questions)
    numbered = [r for r in result if r.question_number is not None]
    assert len(numbered) == 1


# --- non-mcq / no-options-needed questions are untouched ---


def test_non_mcq_question_never_gets_options_populated():
    blocks = [block("list_item", "Explain glycolysis.", marker="1.")]
    questions = [q(number=None, text="Explain glycolysis.", qtype="short_answer")]
    result = reconcile_extracted_questions(blocks, questions)
    assert result[0].options == []


def test_mcq_group_with_no_options_in_evidence_leaves_options_empty():
    blocks = [block("mcq_group", "STEM: 1. Some incomplete stem\nOPTION: ")]
    questions = [q(number=None, text="Some incomplete stem", qtype="mcq", options=[])]
    result = reconcile_extracted_questions(blocks, questions)
    assert result[0].options == []
