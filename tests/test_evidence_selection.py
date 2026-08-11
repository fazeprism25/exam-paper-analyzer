from exampapersorter.question_extraction.evidence_selection import (
    detect_numbering_resets,
    extract_leading_number,
    select_structural_evidence,
    slice_evidence,
)
from exampapersorter.schemas import EvidenceBlock, PageRangeEvidence


def block(block_type, text, pages=(1,)):
    return EvidenceBlock(page_numbers=list(pages), block_type=block_type, text=text)


def test_select_structural_evidence_keeps_headers_and_short_text():
    blocks = [
        block("title", "NITTE University"),
        block("section_header", "II. Long Essay"),
        block("page_header", "Page 3"),
        block("page_footer", "3"),
        block("text", "Date: 04-07-2025"),  # short -- kept
        block("text", "a" * 200),  # long prose -- dropped
        block("list_item", "1. Explain glycolysis"),  # dropped (bulk noise for boundary detection)
    ]
    result = select_structural_evidence(blocks)
    assert [b.block_type for b in result] == ["title", "section_header", "page_header", "page_footer", "text"]
    assert result[-1].text == "Date: 04-07-2025"


def test_detect_numbering_resets_finds_the_drop():
    blocks = [
        block("list_item", "1. First question", pages=(1,)),
        block("list_item", "2. Second question", pages=(1,)),
        block("list_item", "3. Third question", pages=(2,)),
        block("list_item", "1. New paper's first question", pages=(3,)),
        block("list_item", "2. New paper's second question", pages=(3,)),
    ]
    assert detect_numbering_resets(blocks) == [3]


def test_detect_numbering_resets_empty_when_monotonic():
    blocks = [
        block("list_item", "1. a", pages=(1,)),
        block("list_item", "2. b", pages=(1,)),
        block("list_item", "3. c", pages=(2,)),
    ]
    assert detect_numbering_resets(blocks) == []


def test_detect_numbering_resets_ignores_blocks_without_a_leading_number():
    blocks = [
        block("list_item", "1. a", pages=(1,)),
        block("section_header", "MCQ", pages=(1,)),  # wrong type, ignored
        block("list_item", "Choose the correct option", pages=(1,)),  # no leading number
        block("list_item", "2. b", pages=(1,)),
    ]
    assert detect_numbering_resets(blocks) == []


def test_extract_leading_number_strips_digit_number():
    number, remainder = extract_leading_number("16. Hypocalcaemia can occur in all except:")
    assert number == "16"
    assert remainder == "Hypocalcaemia can occur in all except:"


def test_extract_leading_number_handles_subpart_letter():
    number, remainder = extract_leading_number("2a) Explain the Cori cycle.")
    assert number == "2a"
    assert remainder == "Explain the Cori cycle."


def test_extract_leading_number_returns_none_when_absent():
    number, remainder = extract_leading_number("Explain glycolysis.")
    assert number is None
    assert remainder == "Explain glycolysis."


def test_extract_leading_number_does_not_touch_roman_numerals():
    # Roman numerals are deliberately NOT auto-extracted -- "I." collides
    # with ordinary English sentences, unlike a digit-based number.
    number, remainder = extract_leading_number("IV. Short Notes on enzymes")
    assert number is None
    assert remainder == "IV. Short Notes on enzymes"


def test_slice_evidence_keeps_only_blocks_in_range():
    evidence = PageRangeEvidence(
        source_path="fake.pdf", start_page=1, end_page=10, conversion_status="success",
        blocks=[
            block("text", "page1", pages=(1,)),
            block("text", "page5", pages=(5,)),
            block("text", "spans4to6", pages=(4, 5, 6)),
            block("text", "page9", pages=(9,)),
        ],
    )
    sliced = slice_evidence(evidence, 4, 6)
    assert sliced.start_page == 4
    assert sliced.end_page == 6
    assert {b.text for b in sliced.blocks} == {"page5", "spans4to6"}
