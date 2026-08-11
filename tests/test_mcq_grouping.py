from exampapersorter.question_extraction.mcq_grouping import group_mcq_options
from exampapersorter.schemas import EvidenceBlock


def block(block_type, text, pages=(1,), marker=None):
    return EvidenceBlock(page_numbers=list(pages), block_type=block_type, text=text, marker=marker)


def test_stem_and_options_are_bundled_into_one_mcq_group_block():
    blocks = [
        block("list_item", "1. Which enzyme catalyses the rate-limiting step of glycolysis?"),
        block("list_item", "A. Hexokinase"),
        block("list_item", "B. Phosphofructokinase"),
        block("list_item", "C. Pyruvate kinase"),
        block("list_item", "D. Aldolase"),
    ]
    result = group_mcq_options(blocks)
    assert len(result) == 1
    merged = result[0]
    assert merged.block_type == "mcq_group"
    assert "STEM: 1. Which enzyme catalyses" in merged.text
    assert "OPTION: A. Hexokinase" in merged.text
    assert "OPTION: D. Aldolase" in merged.text
    assert merged.page_numbers == [1]


def test_stem_and_options_bundle_when_ocr_has_no_space_after_marker():
    """Regression test: a real-fixture rerun showed 89/103 MCQ options in
    one paper fell through ungrouped because the OCR concatenated the
    option letter directly onto the text ("A.Glycine", no space) and the
    marker regex originally required \\s+ after the punctuation."""
    blocks = [
        block("list_item", "9. One of the following is not an amino acid"),
        block("list_item", "A.Glycine"),
        block("list_item", "B.Hydroxy proline"),
        block("list_item", "C.Glutamic acid"),
        block("list_item", "D.Sleen"),
    ]
    result = group_mcq_options(blocks)
    assert len(result) == 1
    assert result[0].block_type == "mcq_group"
    assert "OPTION: A.Glycine" in result[0].text
    assert "OPTION: D.Sleen" in result[0].text


def test_marker_based_grouping_when_text_has_no_letter_at_all():
    """The real-fixture failure mode this was rewritten for: Docling parses
    the marker OUT of `text` entirely for some documents, so `text` is bare
    "Obstructive Jaundice" with NO letter anywhere in it -- only in the
    separate `marker` field. A text-only regex can never recover this; it
    has to use `marker`."""
    blocks = [
        block("list_item", "Serum Alkaline phosphatase is increased in all the following except", marker="1."),
        block("list_item", "Obstructive Jaundice", marker="A."),
        block("list_item", "Rickets", marker="B."),
        block("list_item", "Infectious hepatitis", marker="C."),
        block("list_item", "Myocardial Infraction", marker="D."),
    ]
    result = group_mcq_options(blocks)
    assert len(result) == 1
    assert result[0].block_type == "mcq_group"
    assert "OPTION: A. Obstructive Jaundice" in result[0].text
    assert "OPTION: D. Myocardial Infraction" in result[0].text


def test_overlong_option_run_is_left_ungrouped_and_flagged():
    """A run this long is more likely several real questions' options
    bleeding together (e.g. a garbled/missed stem in between) than one
    genuine MCQ -- left ungrouped and flagged rather than silently merged.
    Regression test for a real-fixture case where a 16-option run merged
    four unrelated questions into a single record."""
    # Two real 4-option MCQs' markers repeat back-to-back, simulating the
    # actual failure: a stem between them went undetected, so what should
    # have been two separate 4-option runs looks like one 8-option run.
    markers = ["A.", "B.", "C.", "D.", "A.", "B.", "C.", "D."]
    blocks = [block("list_item", "9. Stem question", marker="9.")] + [
        block("list_item", f"option {i}", marker=m) for i, m in enumerate(markers)
    ]
    result = group_mcq_options(blocks)
    assert not any(b.block_type == "mcq_group" for b in result)
    flagged = [b for b in result if "SUSPICIOUSLY LONG OPTION RUN" in b.text]
    assert len(flagged) == 8


def test_unrelated_blocks_pass_through_unchanged():
    blocks = [
        block("title", "Department of Biochemistry"),
        block("text", "Date: 04-07-2025"),
        block("list_item", "1. Explain glycolysis."),
    ]
    result = group_mcq_options(blocks)
    assert [b.text for b in result] == [b.text for b in blocks]
    assert all(b.block_type != "mcq_group" for b in result)


def test_single_option_like_block_is_not_grouped():
    # A lone option-looking line isn't a run -- e.g. a stray "(a)" inside
    # ordinary prose shouldn't be misread as the start of an MCQ.
    blocks = [
        block("list_item", "1. Explain the steps below."),
        block("list_item", "A. Step one only."),
        block("list_item", "2. Next question."),
    ]
    result = group_mcq_options(blocks)
    assert len(result) == 3
    assert all(b.block_type != "mcq_group" for b in result)


def test_options_without_a_findable_stem_are_flagged_not_dropped():
    blocks = [
        block("table", "unrelated table content"),  # not a valid stem type
        block("list_item", "A. Trypsin"),
        block("list_item", "B. Pepsin"),
        block("list_item", "C. Lipase"),
    ]
    result = group_mcq_options(blocks)
    assert not any(b.block_type == "mcq_group" for b in result)
    option_blocks = [b for b in result if b.block_type == "list_item"]
    assert len(option_blocks) == 3
    assert all("UNASSOCIATED OPTION" in b.text for b in option_blocks)
    # Original option text is preserved, not discarded.
    assert any("Trypsin" in b.text for b in option_blocks)


def test_two_consecutive_mcqs_each_get_their_own_group():
    blocks = [
        block("list_item", "1. First stem?"),
        block("list_item", "A. opt1a"),
        block("list_item", "B. opt1b"),
        block("list_item", "2. Second stem?"),
        block("list_item", "A. opt2a"),
        block("list_item", "B. opt2b"),
    ]
    result = group_mcq_options(blocks)
    groups = [b for b in result if b.block_type == "mcq_group"]
    assert len(groups) == 2
    assert "First stem" in groups[0].text
    assert "opt1a" in groups[0].text and "opt1b" in groups[0].text
    assert "Second stem" in groups[1].text
    assert "opt2a" in groups[1].text and "opt2b" in groups[1].text


def test_marks_annotation_between_stem_and_options_is_not_mistaken_for_the_stem():
    """Regression test for a real-fixture bug: a marks/marking-scheme
    annotation block ("10X1=10") sitting between a numbered stem and its
    option run was picked as the "stem" by the lookback search (it's
    groupable and not option-like), leaving the REAL numbered stem
    orphaned as its own unbundled block one position further back."""
    blocks = [
        block("list_item", "The most important intracellular buffer", marker="1."),
        block("text", "10X1=10"),
        block("list_item", "Bicarbonate", marker="A)"),
        block("list_item", "Protein buffer", marker="B)"),
        block("list_item", "Phosphorus buffer", marker="C)"),
        block("list_item", "Hemoglobin", marker="D)"),
    ]
    result = group_mcq_options(blocks)
    groups = [b for b in result if b.block_type == "mcq_group"]
    assert len(groups) == 1
    assert "STEM: 1. The most important intracellular buffer" in groups[0].text
    assert "OPTION: A) Bicarbonate" in groups[0].text
    # The marks annotation is left as its own harmless standalone block,
    # not swallowed into the group as a bogus "stem".
    assert any(b.text == "10X1=10" for b in result)


def test_non_mcq_short_essay_questions_are_left_alone():
    blocks = [
        block("list_item", "1. Explain the Cori cycle."),
        block("list_item", "2. Describe beta oxidation."),
        block("list_item", "3. Outline the urea cycle."),
    ]
    result = group_mcq_options(blocks)
    assert result == blocks
