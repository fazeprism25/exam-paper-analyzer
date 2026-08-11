"""Identify distinct exam papers within one (possibly multi-paper) source PDF.

Two-step design, mirroring how Stage 1 separates "ask the LLM" from
"deterministically validate/repair the structure it gave back":

1. detect_paper_boundaries -- one LLM call, grounded in condensed
   structural evidence plus a deterministic numbering-reset hint. Returns
   candidate spans, each with its own evidence citations.
2. resolve_paper_spans -- purely deterministic. Sorts candidates, clips
   overlaps, and decides what happens to any pages no candidate covers.
   Never invents paper content -- gaps are either attached to an adjacent
   paper (with an explicit, auditable note) or, if there's no adjacent
   paper to attach to, left unassigned and reported rather than dropped
   silently.
"""
from __future__ import annotations

import logging

from exampapersorter.config import Config
from exampapersorter.evidence_format import render_evidence
from exampapersorter.llm_client import call_structured
from exampapersorter.question_extraction.evidence_selection import (
    detect_numbering_resets,
    select_structural_evidence,
)
from exampapersorter.schemas import (
    PageRangeEvidence,
    PaperBoundaryCandidate,
    PaperBoundaryDetectionResult,
    PaperBoundaryIssue,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are identifying where distinct exam papers begin and end within a single scanned \
PDF that may contain more than one exam paper concatenated together (e.g. a compiled question bank \
spanning several years/batches).

Signals that indicate a NEW paper has started (any may be present; none are guaranteed, and none alone \
is conclusive):
- a repeated institutional/department header reappearing
- a new date, batch year, or "Nth sessional/final examination" style heading
- question/item numbering restarting from near 1
- a distinct exam title/subject header

A single exam paper is usually contiguous pages sharing one header/date/institution context, but do NOT \
assume every paper spans more than one page -- a compact paper can occupy a single page, and a compiled \
question bank can pack one paper per page.

You are given: (a) structural evidence blocks (titles, section headers, page headers/footers, and short \
text lines such as dates) with their PDF page numbers, and (b) a deterministic hint listing pages where \
question/item numbering appears to reset. Treat the hint as a clue, not a verdict -- a numbering reset \
can also happen for a benign reason, e.g. a section restarting its own count within the same paper.

For each paper you identify, report its start and end PDF page (inclusive), a confidence 0-1, and cite \
the specific evidence (page + text snippet) that grounds your decision. If the whole document is one \
paper, report exactly one entry spanning all pages given.

Never invent page numbers or evidence that isn't in what you were given."""


def detect_paper_boundaries(evidence: PageRangeEvidence, config: Config) -> PaperBoundaryDetectionResult:
    structural = select_structural_evidence(evidence.blocks)
    reset_pages = detect_numbering_resets(evidence.blocks)
    hint = (
        f"Deterministic hint -- pages where question/item numbering appears to reset: {reset_pages}\n\n"
        if reset_pages
        else "Deterministic hint -- no question/item numbering resets were detected.\n\n"
    )
    user_prompt = (
        f"Structural evidence from PDF pages {evidence.start_page}-{evidence.end_page} "
        f"({evidence.end_page - evidence.start_page + 1} total pages):\n\n"
        f"{hint}"
        f"{render_evidence(structural)}\n\n"
        "Identify the distinct exam paper(s) contained in this document."
    )
    return call_structured(config, SYSTEM_PROMPT, user_prompt, PaperBoundaryDetectionResult, call_label="paper_boundaries")


def resolve_paper_spans(
    candidates: list[PaperBoundaryCandidate], total_pages: int
) -> tuple[list[PaperBoundaryCandidate], list[PaperBoundaryIssue]]:
    """Deterministically sorts/clips LLM-proposed spans into a
    non-overlapping sequence covering as much of [1, total_pages] as
    possible, reporting (not silently repairing) anything it had to
    clip or couldn't cover."""
    issues: list[PaperBoundaryIssue] = []
    ordered = sorted(candidates, key=lambda c: (c.start_page, c.end_page))

    clipped: list[PaperBoundaryCandidate] = []
    prev_end = 0
    for c in ordered:
        start = max(c.start_page, prev_end + 1)
        end = min(c.end_page, total_pages)
        if start > end:
            issues.append(
                PaperBoundaryIssue(
                    code="overlap_clipped",
                    message=f"Candidate span {c.start_page}-{c.end_page} was entirely overlapped by a "
                    f"preceding paper and dropped",
                    pages=list(range(c.start_page, min(c.end_page, total_pages) + 1)),
                )
            )
            continue
        if start != c.start_page:
            issues.append(
                PaperBoundaryIssue(
                    code="overlap_clipped",
                    message=f"Candidate span start clipped from page {c.start_page} to {start} to avoid "
                    f"overlapping the preceding paper",
                    pages=list(range(c.start_page, start)),
                )
            )
        clipped.append(c.model_copy(update={"start_page": start, "end_page": end}))
        prev_end = end

    filled: list[PaperBoundaryCandidate] = []
    cursor = 1
    for span in clipped:
        if span.start_page > cursor:
            gap_pages = list(range(cursor, span.start_page))
            if filled:
                prev = filled[-1]
                filled[-1] = prev.model_copy(update={"end_page": span.start_page - 1})
                issues.append(
                    PaperBoundaryIssue(
                        code="gap_filled",
                        message=f"Pages {gap_pages} were not covered by any detected paper; attached to "
                        f"the preceding paper (now {filled[-1].start_page}-{filled[-1].end_page}) rather "
                        f"than dropped",
                        pages=gap_pages,
                    )
                )
            else:
                issues.append(
                    PaperBoundaryIssue(
                        code="gap_unassigned",
                        message=f"Pages {gap_pages} precede the first detected paper and were not covered "
                        f"by any boundary; left unassigned for manual review",
                        pages=gap_pages,
                    )
                )
        filled.append(span)
        cursor = span.end_page + 1

    if cursor <= total_pages:
        trailing_gap = list(range(cursor, total_pages + 1))
        if filled:
            prev = filled[-1]
            filled[-1] = prev.model_copy(update={"end_page": total_pages})
            issues.append(
                PaperBoundaryIssue(
                    code="gap_filled",
                    message=f"Trailing pages {trailing_gap} were not covered by any detected paper; "
                    f"attached to the last paper (now {filled[-1].start_page}-{filled[-1].end_page})",
                    pages=trailing_gap,
                )
            )
        else:
            issues.append(
                PaperBoundaryIssue(
                    code="gap_unassigned",
                    message=f"No papers were detected at all; pages {trailing_gap} left unassigned",
                    pages=trailing_gap,
                )
            )

    return filled, issues
