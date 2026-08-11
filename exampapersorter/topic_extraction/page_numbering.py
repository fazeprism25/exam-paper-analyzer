"""Deterministic mapping between a textbook's own printed page numbers
(as they appear in page headers/footers) and the PDF's absolute page
index.

Why this exists: a table of contents' page-number column refers to the
book's own numbering, which is almost always offset from the PDF's page
index by however many unnumbered/differently-numbered front-matter pages
(title page, preface, roman-numeral pages) precede page 1 of the main
text. To go from "this chapter starts on the book's own page 244" to
"fetch PDF page N", that offset has to be recovered from evidence, not
assumed.

Entirely deterministic -- no LLM involved. If no reliable numbering
pattern can be established, callers get None and must not guess.

KNOWN GAP (confirmed directly against our real test textbook): this only
works if Docling's extraction actually surfaces bare printed page numbers
as page_header/page_footer blocks somewhere in the evidence it's given.
On our real test textbook, Docling produced ZERO such blocks anywhere
checked (the TOC window, pages 26-40, and specifically the page a much
earlier draft of this comment incorrectly claimed showed "4" in the
footer -- it doesn't, in the actual extraction). Broadening the block_type
filter would not help: there's no bare-digit text to find at all in this
book's Docling output, under any block type. When that's the case,
build_page_number_mapping correctly returns None and name_recovery.py
falls back to resolution_status="unidentified" -- never a guess -- but
recovery genuinely cannot proceed for a book in this situation. A more
capable mitigation (e.g. inferring page correspondence from some other
structural signal) is a real design question, not something to improvise
here.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from exampapersorter.schemas import EvidenceBlock, PageRangeEvidence

# The ENTIRE (stripped) string must be nothing but an optional bracket/
# paren wrapper around digits -- e.g. "4", "[4]", "( 12 )". Deliberately
# strict: an earlier version used `\D*\d+\D*`, which matched a bare digit
# ANYWHERE in the string and incorrectly treated header text like
# "Chapter 1 : BIOMOLECULES AND THE CELL" as printed page number "1".
_PAGE_NUMBER_ONLY = re.compile(r"^[\[(]?\s*(\d{1,4})\s*[\])]?$")


def extract_printed_page_number(text: str) -> int | None:
    """Parses a page_header/page_footer block's text as a bare printed
    page number (e.g. "4", "[4]" -> 4; "[ iv ]" -> None, roman numeral;
    "Chapter 1 : BIOMOLECULES" -> None, not just a number)."""
    match = _PAGE_NUMBER_ONLY.match(text.strip())
    if not match:
        return None
    return int(match.group(1))


@dataclass
class PageNumberMapping:
    offset: int  # pdf_page = book_page + offset
    sample_size: int

    def book_page_to_pdf_page(self, book_page: int) -> int:
        return book_page + self.offset


def build_page_number_mapping(blocks: list[EvidenceBlock], min_samples: int = 3) -> PageNumberMapping | None:
    offsets: list[int] = []
    for block in blocks:
        if block.block_type not in ("page_header", "page_footer"):
            continue
        book_page = extract_printed_page_number(block.text)
        if book_page is None:
            continue
        for pdf_page in block.page_numbers:
            offsets.append(pdf_page - book_page)

    if len(offsets) < min_samples:
        return None

    counts = Counter(offsets)
    best_offset, best_count = counts.most_common(1)[0]
    # The winning offset must be a clear plurality, not one lucky match
    # among noisy/misparsed candidates.
    if best_count < max(min_samples, len(offsets) // 2):
        return None

    return PageNumberMapping(offset=best_offset, sample_size=best_count)


def confirm_expected_page(evidence: PageRangeEvidence, expected_book_page: int) -> int | None:
    """Deterministically confirms which PDF page within a fetched window
    actually carries the expected printed page number in its header/
    footer, and returns that PDF page number.

    Returns None if the expected number isn't found anywhere in the
    window. That's a real, useful signal: it means the offset-based guess
    landed somewhere unexpected (e.g. an inserted unnumbered figure page
    threw the count off), and evidence from this window should NOT be
    trusted for name recovery -- better to report the topic unidentified
    than to pair its number with the wrong page's content.
    """
    for block in evidence.blocks:
        if block.block_type not in ("page_header", "page_footer"):
            continue
        if extract_printed_page_number(block.text) == expected_book_page:
            return min(block.page_numbers)
    return None
