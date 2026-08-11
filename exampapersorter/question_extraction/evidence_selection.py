"""Deterministic evidence selection/signal-derivation for Stage 2.

Two things live here, both non-LLM:

- select_structural_evidence: a condensed view of a (possibly large,
  multi-paper) document's evidence, used only for paper-boundary detection.
  Unlike Stage 1's topic extraction, Stage 2 must NOT drop generic "text"
  blocks wholesale -- on a real exam paper, a short "text" block is often
  the exam date ("Date: 04-07-2025"), which is exactly the kind of signal
  boundary detection needs. What actually needs dropping here is bulk: the
  MCQ options and question stems (list_item/long text), which dominate
  block count on a real paper (670 of 878 blocks on our test fixture) and
  would blow the context budget without helping decide where one paper
  ends and the next begins.

- detect_numbering_resets: a generic, format-agnostic heuristic that flags
  PDF pages where a leading question/item number drops back down compared
  to the page before -- a common (not universal) signal of a new paper
  starting. This is a HINT handed to the LLM alongside the evidence, not a
  verdict: a numbering reset can also happen for benign reasons (e.g. a
  section restarting its own count within the same paper), so the semantic
  call of whether it actually marks a paper boundary is left to the LLM.
"""
from __future__ import annotations

import re

from exampapersorter.schemas import EvidenceBlock, PageRangeEvidence

# Condensed-view threshold: a "text" block this short or shorter is treated
# as a likely metadata/header line (date, marks, time allowed) rather than
# question prose, and kept. Longer "text" blocks are dropped from the
# condensed view -- they're still available, unfiltered, once a paper's
# page range is known and its own sections/questions are being extracted.
_SHORT_TEXT_THRESHOLD = 150

_STRUCTURAL_BLOCK_TYPES = {"title", "section_header", "page_header", "page_footer"}


def select_structural_evidence(blocks: list[EvidenceBlock]) -> list[EvidenceBlock]:
    selected = []
    for block in blocks:
        if block.block_type in _STRUCTURAL_BLOCK_TYPES:
            selected.append(block)
        elif block.block_type == "text" and len(block.text) <= _SHORT_TEXT_THRESHOLD:
            selected.append(block)
    return selected


_LEADING_NUMBER = re.compile(r"^\s*(\d{1,3})[.\):]")


def detect_numbering_resets(blocks: list[EvidenceBlock]) -> list[int]:
    """Returns the sorted, deduplicated PDF page numbers where a leading
    question/item number (e.g. "7. ..." or "3) ...") is lower than the
    last one seen, scanning blocks in page order. Independent of any
    exam-specific vocabulary -- purely a numeric pattern over whatever
    list_item/text blocks happen to start with "<number><punctuation>"."""
    ordered = sorted(
        (b for b in blocks if b.block_type in ("list_item", "text") and b.page_numbers),
        key=lambda b: min(b.page_numbers),
    )
    prev_num: int | None = None
    reset_pages: set[int] = set()
    for block in ordered:
        match = _LEADING_NUMBER.match(block.text)
        if not match:
            continue
        num = int(match.group(1))
        if prev_num is not None and num < prev_num:
            reset_pages.add(min(block.page_numbers))
        prev_num = num
    return sorted(reset_pages)


# Deliberately digits-only (with an optional trailing sub-part letter, e.g.
# "2a"): a safety-net for when the model leaves a visibly-printed number
# embedded in question_text instead of populating question_number itself
# (observed on every question in the first real run). Roman numerals are
# NOT auto-extracted here -- "I." collides with ordinary English sentences
# ("I. Long Essay" is a section label, but "I. was absent" is a sentence),
# so that class of numbering is left to the LLM's own judgment rather than
# risking a wrong deterministic strip. This only ever recovers a number
# that is already visibly present in the text -- it never invents one.
_LEADING_QUESTION_NUMBER = re.compile(r"^\s*(\d{1,3}[a-zA-Z]?)[.\):]\s+")


def extract_leading_number(text: str) -> tuple[str | None, str]:
    """If `text` starts with a printed digit-based number/label, returns
    (number, remaining_text_with_the_label_stripped). Otherwise returns
    (None, text) unchanged."""
    match = _LEADING_QUESTION_NUMBER.match(text)
    if not match:
        return None, text
    return match.group(1), text[match.end():]


def slice_evidence(evidence: PageRangeEvidence, start_page: int, end_page: int) -> PageRangeEvidence:
    """Client-side slice of an already-fetched whole-document evidence
    object down to one paper's page range -- no extra Docling call, since
    the full document was already OCR'd once. A block that spans pages is
    kept if ANY of its pages falls in range."""
    blocks = [b for b in evidence.blocks if any(start_page <= p <= end_page for p in b.page_numbers)]
    return evidence.model_copy(update={"start_page": start_page, "end_page": end_page, "blocks": blocks})
