import pymupdf
import pytest

from exampapersorter.pdf_utils import page_range_has_text_layer


@pytest.fixture(scope="module")
def mixed_pdf(tmp_path_factory):
    """Page 1 has a real text layer; page 2 is text-free (simulates an
    image-only/scanned page -- no image needed, just no text)."""
    path = tmp_path_factory.mktemp("fixtures") / "mixed.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This page has a substantial amount of real text on it.", fontsize=12)
    doc.new_page()  # page 2: nothing on it at all
    doc.save(str(path))
    doc.close()
    return path


def test_true_when_every_page_in_range_has_text(mixed_pdf):
    assert page_range_has_text_layer(mixed_pdf, 1, 1) is True


def test_false_when_page_in_range_lacks_text(mixed_pdf):
    assert page_range_has_text_layer(mixed_pdf, 2, 2) is False


def test_false_when_range_mixes_texted_and_textless_pages(mixed_pdf):
    assert page_range_has_text_layer(mixed_pdf, 1, 2) is False


def test_false_when_page_text_is_below_min_chars(tmp_path):
    path = tmp_path / "sparse.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "42", fontsize=12)  # e.g. a stray page-number stamp
    doc.save(str(path))
    doc.close()
    assert page_range_has_text_layer(path, 1, 1, min_chars=40) is False


def test_false_when_range_exceeds_page_count(mixed_pdf):
    assert page_range_has_text_layer(mixed_pdf, 1, 5) is False
