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
