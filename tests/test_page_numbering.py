from exampapersorter.schemas import EvidenceBlock, PageRangeEvidence
from exampapersorter.topic_extraction.page_numbering import (
    build_page_number_mapping,
    confirm_expected_page,
    extract_printed_page_number,
)


def header(pdf_page, text, block_type="page_footer"):
    return EvidenceBlock(page_numbers=[pdf_page], block_type=block_type, text=text)


def test_extract_printed_page_number_plain_digit():
    assert extract_printed_page_number("4") == 4
    assert extract_printed_page_number("  12  ") == 12


def test_extract_printed_page_number_rejects_roman_numerals():
    assert extract_printed_page_number("[ iv ]") is None
    assert extract_printed_page_number("ii") is None


def test_extract_printed_page_number_rejects_mixed_text():
    assert extract_printed_page_number("Chapter 1 : BIOMOLECULES AND THE CELL") is None


def test_build_mapping_from_consistent_offset():
    blocks = [
        header(14, "4"),
        header(15, "5"),
        header(19, "9"),
        header(20, "10"),
    ]
    mapping = build_page_number_mapping(blocks, min_samples=3)
    assert mapping is not None
    assert mapping.offset == 10
    assert mapping.book_page_to_pdf_page(244) == 254


def test_build_mapping_ignores_roman_numeral_front_matter():
    blocks = [
        header(7, "[ ii ]"),
        header(8, "[ iii ]"),
        header(14, "4"),
        header(15, "5"),
        header(19, "9"),
    ]
    mapping = build_page_number_mapping(blocks, min_samples=3)
    assert mapping is not None
    assert mapping.offset == 10


def test_build_mapping_returns_none_with_insufficient_samples():
    blocks = [header(14, "4")]
    assert build_page_number_mapping(blocks, min_samples=3) is None


def test_build_mapping_returns_none_when_no_clear_majority():
    # Noisy/conflicting offsets, no plurality -- should refuse to guess.
    blocks = [header(14, "4"), header(20, "3"), header(30, "1")]
    assert build_page_number_mapping(blocks, min_samples=3) is None


def test_confirm_expected_page_found():
    evidence = PageRangeEvidence(
        source_path="fake.pdf", start_page=253, end_page=256, conversion_status="success",
        blocks=[
            header(253, "243"),
            header(254, "244"),
        ],
    )
    assert confirm_expected_page(evidence, expected_book_page=244) == 254


def test_confirm_expected_page_not_found():
    evidence = PageRangeEvidence(
        source_path="fake.pdf", start_page=253, end_page=256, conversion_status="success",
        blocks=[header(253, "243"), header(254, "245")],  # 244 never appears
    )
    assert confirm_expected_page(evidence, expected_book_page=244) is None
