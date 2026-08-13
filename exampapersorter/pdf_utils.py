"""Cheap, deterministic PDF metadata via PyMuPDF -- no Docling involved.

Kept separate from docling_processing.py because these operations (page
count, content hash) are needed before we decide how much Docling work to
do, and are orders of magnitude cheaper than invoking Docling.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf


def get_page_count(pdf_path: Path) -> int:
    doc = pymupdf.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def compute_file_hash(pdf_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """SHA-256 of file bytes -- identifies content, not filename/path."""
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# Below this, a substantial-but-not-huge amount of text on a page is a
# reliable enough signal that the page already carries a usable text layer
# (born-digital or already run through OCR) that Docling's own OCR stage
# doesn't need to touch it. Deliberately conservative -- low enough that a
# short/sparse-but-real page of prose still counts, high enough that a
# stray page-number stamp on an otherwise image-only page doesn't.
MIN_USABLE_TEXT_CHARS = 40


def page_range_has_text_layer(pdf_path: Path, start_page: int, end_page: int, min_chars: int = MIN_USABLE_TEXT_CHARS) -> bool:
    """True only if EVERY page in [start_page, end_page] (1-indexed,
    inclusive) already has a text layer of at least `min_chars` non-
    whitespace characters. A single sparse or image-only page makes this
    False -- callers should treat False as "OCR may still be needed
    somewhere in this range" rather than assume the whole range needs it;
    Docling's own per-page bitmap-coverage check (see
    docling.models.base_ocr_model.BaseOcrModel.get_ocr_rects) already
    handles selectively OCRing just the pages that need it.
    """
    doc = pymupdf.open(pdf_path)
    try:
        if start_page < 1 or end_page > doc.page_count:
            return False
        for page_no in range(start_page, end_page + 1):
            text = doc[page_no - 1].get_text()
            if len(text.strip()) < min_chars:
                return False
    finally:
        doc.close()
    return True
