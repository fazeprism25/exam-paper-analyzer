import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont

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


def _render_scanned_page_image() -> Image.Image:
    """A raster image with real drawn text but no PDF text layer --
    stands in for a scanned page. Placed on the page as a bitmap so
    Docling's own bitmap-coverage check (get_ocr_rects) decides OCR is
    needed there, independent of our do_ocr routing."""
    img = Image.new("RGB", (1275, 1650), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except Exception:
        font = ImageFont.load_default(size=32)
    draw.text((100, 150), "Scanned page with no embedded text layer.", fill="black", font=font)
    return img


@pytest.fixture(scope="module")
def already_ocred_pdf(tmp_path_factory):
    """A scanned page image with an invisible text layer already burned
    in on top of it -- what a PDF that's already been through
    OCRmyPDF/Tesseract looks like: pixels plus a real (if imperfect)
    embedded text layer at render_mode=3 (invisible). Deliberately
    phrased differently from the image's drawn text so a passing test
    proves the pre-existing layer -- not a fresh re-OCR of the pixels --
    is what came through."""
    path = tmp_path_factory.mktemp("fixtures") / "already_ocred.pdf"
    img_path = tmp_path_factory.mktemp("fixtures") / "scan3.png"
    _render_scanned_page_image().save(img_path)

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(page.rect, filename=str(img_path))
    page.insert_text((50, 60), "Pre-existing OCR text layer already on this page.", fontsize=11, render_mode=3)
    doc.save(str(path))
    doc.close()
    return path


def test_already_ocred_pdf_uses_existing_text_layer(already_ocred_pdf):
    """Case C from the OCR audit: a page that already has an OCR text
    layer must keep using it (and take the do_ocr=False fast path via
    page_range_has_text_layer) rather than being OCR'd again."""
    evidence = extract_page_range_evidence(already_ocred_pdf, 1, 1)
    assert evidence.conversion_status in ("success", "partial_success")
    all_text = " ".join(b.text for b in evidence.blocks)
    assert "pre-existing ocr text layer" in all_text.lower()


@pytest.fixture(scope="module")
def image_only_pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("fixtures") / "image_only.pdf"
    img_path = tmp_path_factory.mktemp("fixtures") / "scan.png"
    _render_scanned_page_image().save(img_path)

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(page.rect, filename=str(img_path))
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture(scope="module")
def mixed_pdf(tmp_path_factory):
    """Page 1: native text layer. Page 2: image-only (no text layer) --
    regression fixture for the "mixed PDF" behavior this experiment was
    scoped around: a range spanning both must not silently drop page 2's
    content just because page 1 already has text."""
    path = tmp_path_factory.mktemp("fixtures") / "mixed.pdf"
    img_path = tmp_path_factory.mktemp("fixtures") / "scan2.png"
    _render_scanned_page_image().save(img_path)

    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Native text page with a real embedded text layer.", fontsize=12)
    page2 = doc.new_page(width=612, height=792)
    page2.insert_image(page2.rect, filename=str(img_path))
    doc.save(str(path))
    doc.close()
    return path


def test_image_only_pdf_is_still_ocred(image_only_pdf):
    """Guards against the do_ocr routing regressing to skip OCR on a page
    that has no pre-existing text -- that would silently return empty
    evidence for a genuinely scanned page."""
    evidence = extract_page_range_evidence(image_only_pdf, 1, 1)
    assert evidence.conversion_status in ("success", "partial_success")
    all_text = " ".join(b.text for b in evidence.blocks)
    assert "scanned page" in all_text.lower()


def test_mixed_pdf_full_range_extracts_both_pages(mixed_pdf):
    """The range as a whole has one textless page, so do_ocr must stay on
    for this call -- both the native text and the OCR'd image text must
    come through."""
    evidence = extract_page_range_evidence(mixed_pdf, 1, 2)
    assert evidence.conversion_status in ("success", "partial_success")
    all_text = " ".join(b.text for b in evidence.blocks)
    assert "native text page" in all_text.lower()
    assert "scanned page" in all_text.lower()


def test_mixed_pdf_image_only_subrange_is_still_ocred(mixed_pdf):
    """A sub-range that happens to be just the image-only page must not be
    mistaken for "already has text" -- this is the do_ocr fast path's
    highest-risk failure mode (see page_range_has_text_layer)."""
    evidence = extract_page_range_evidence(mixed_pdf, 2, 2)
    assert evidence.conversion_status in ("success", "partial_success")
    all_text = " ".join(b.text for b in evidence.blocks)
    assert "scanned page" in all_text.lower()
