import pymupdf
import pytest

from exampapersorter.docling_processing import extract_page_range_evidence


@pytest.fixture(scope="module")
def synthetic_toc_pdf(tmp_path_factory):
    """A small born-digital PDF with a dot-leader table of contents --
    the exact pattern that, on the real textbook, Docling's layout model
    classified as a TableItem rather than plain text."""
    path = tmp_path_factory.mktemp("fixtures") / "synthetic_toc.pdf"
    doc = pymupdf.open()

    page = doc.new_page()
    page.insert_text((72, 72), "Preface", fontsize=14)
    page.insert_text((72, 100), "This page has no contents listing.", fontsize=11)

    page = doc.new_page()
    page.insert_text((72, 72), "Table of Contents", fontsize=16)
    lines = [
        "Chapter 1: Carbohydrate Metabolism ..................... 1",
        "Chapter 2: Lipid Metabolism ............................. 25",
        "Chapter 3: Amino Acid Metabolism ........................ 40",
    ]
    y = 110
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 20

    doc.save(str(path))
    doc.close()
    return path


def test_dot_leader_toc_content_is_captured(synthetic_toc_pdf):
    """Regression test for the bug found against the real textbook: if
    evidence extraction only walked doc.texts, dot-leader TOC lines
    (classified as a table by Docling) would be silently missing."""
    evidence = extract_page_range_evidence(synthetic_toc_pdf, 1, 2)
    assert evidence.conversion_status in ("success", "partial_success")

    all_text = " ".join(b.text for b in evidence.blocks)
    assert "Carbohydrate Metabolism" in all_text
    assert "Lipid Metabolism" in all_text
    assert "Amino Acid Metabolism" in all_text


def test_page_numbers_preserved_across_range(synthetic_toc_pdf):
    evidence = extract_page_range_evidence(synthetic_toc_pdf, 2, 2)
    assert evidence.start_page == 2
    assert evidence.end_page == 2
    pages_seen = {p for b in evidence.blocks for p in b.page_numbers}
    assert pages_seen == {2}


def test_missing_file_reports_failure_without_raising(tmp_path):
    evidence = extract_page_range_evidence(tmp_path / "does_not_exist.pdf", 1, 5)
    assert evidence.conversion_status == "failure"
    assert evidence.blocks == []
    assert evidence.error_message is not None


def test_corrupt_file_reports_failure_without_raising(tmp_path):
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"not a real pdf")
    evidence = extract_page_range_evidence(corrupt, 1, 5)
    assert evidence.conversion_status == "failure"
    assert evidence.blocks == []
