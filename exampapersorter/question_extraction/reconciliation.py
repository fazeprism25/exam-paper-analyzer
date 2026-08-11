"""Deterministic post-hoc reconciliation between the LLM's extracted
questions and the (already deterministically bundled) evidence blocks they
came from.

Two real-fixture failure modes this recovers WITHOUT another LLM call,
because the answer is already sitting in evidence built by our own code:

  - question_number: every block that started with a printed number/marker
    already told us that number BEFORE it was ever rendered for the LLM
    (see EvidenceBlock.marker / evidence_selection.extract_leading_number).
    Observed directly on the real fixture: the model dropped question_number
    on every single question (0/215) even when the number was plainly
    visible in the evidence it was shown -- so rather than trust it to copy
    a value it has already demonstrated it won't reliably copy, match the
    returned question_text back to the evidence block it almost certainly
    came from and backfill the number from the block itself.

  - MCQ options: an `mcq_group` block (mcq_grouping.py) already contains its
    own bundled "OPTION: ..." lines verbatim. Observed directly: on roughly
    half the real fixture's papers the model returned question_type=mcq
    with an empty options list despite the evidence block it read plainly
    listing all the options -- so instead of asking it to reproduce text
    that's already sitting in our own evidence, read the options back out
    of the block.

Matching is deliberately conservative: a candidate block is only used if
its associated text snippet is a confident, UNIQUE match to exactly one
still-incomplete question. No match (or an ambiguous multi-match) leaves
the field exactly as the LLM returned it (None / empty) -- never guessed.
"""
from __future__ import annotations

import re

from exampapersorter.question_extraction.evidence_selection import extract_leading_number
from exampapersorter.schemas import EvidenceBlock, ExtractedQuestion

_STEM_LINE = re.compile(r"STEM:\s*(.*?)(?=\nOPTION:|\Z)", re.DOTALL)
_OPTION_LINE = re.compile(r"^OPTION:\s*(.*)$", re.MULTILINE)
_DIGIT_MARKER = re.compile(r"^(\d{1,3}[a-zA-Z]?)[.\):]?$")

# Block types that are never a question's own stem (headers/captions/etc.) --
# mirrors the structural-vs-content distinction used elsewhere (see
# evidence_selection._STRUCTURAL_BLOCK_TYPES), kept local since the exact
# set relevant to "could this be a question stem" isn't identical.
_NON_STEM_TYPES = {"title", "section_header", "page_header", "page_footer", "caption", "document_index", "table", "other"}

# How much of a normalized snippet must agree, at minimum, to count as
# plausibly "the same question" -- short enough to tolerate the LLM's own
# minor rephrasing/OCR-correction, long enough that two different short
# questions can't collide by chance.
_MATCH_PREFIX_LEN = 30
_MIN_SNIPPET_LEN = 12


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class _Candidate:
    __slots__ = ("number", "snippet", "options")

    def __init__(self, number: str | None, snippet: str, options: list[str]):
        self.number = number
        self.snippet = snippet
        self.options = options


def _mcq_group_candidate(block: EvidenceBlock) -> _Candidate:
    stem_match = _STEM_LINE.search(block.text)
    stem = stem_match.group(1).strip() if stem_match else block.text
    options = [o.strip() for o in _OPTION_LINE.findall(block.text) if o.strip()]
    number, remainder = extract_leading_number(stem)
    return _Candidate(number, _normalize(remainder if number else stem), options)


def _plain_block_candidate(block: EvidenceBlock) -> _Candidate | None:
    if block.marker:
        m = _DIGIT_MARKER.match(block.marker.strip())
        if m:
            return _Candidate(m.group(1), _normalize(block.text), [])
        return None  # a non-digit marker here (e.g. a lettered option) is never a question's own stem
    number, remainder = extract_leading_number(block.text)
    if number:
        return _Candidate(number, _normalize(remainder), [])
    return None


def _build_candidates(blocks: list[EvidenceBlock]) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for b in blocks:
        if b.block_type == "mcq_group":
            candidate = _mcq_group_candidate(b)
        elif b.block_type in ("list_item", "text") and b.block_type not in _NON_STEM_TYPES:
            candidate = _plain_block_candidate(b)
        else:
            candidate = None
        if candidate is not None and len(candidate.snippet) >= _MIN_SNIPPET_LEN:
            candidates.append(candidate)
    return candidates


def _snippet_matches(candidate_snippet: str, question_norm_text: str) -> bool:
    n = min(len(candidate_snippet), len(question_norm_text), _MATCH_PREFIX_LEN)
    if n < _MIN_SNIPPET_LEN:
        return False
    return candidate_snippet[:n] == question_norm_text[:n]


def reconcile_extracted_questions(
    blocks: list[EvidenceBlock], questions: list[ExtractedQuestion]
) -> list[ExtractedQuestion]:
    """Backfills question_number and (for MCQs) options straight from the
    deterministic evidence, for any question the LLM returned without them.
    See module docstring for why this is safe/conservative."""
    candidates = _build_candidates(blocks)
    used: set[int] = set()
    result: list[ExtractedQuestion] = []
    for q in questions:
        needs_number = q.question_number is None
        needs_options = q.question_type == "mcq" and not q.options
        if not needs_number and not needs_options:
            result.append(q)
            continue

        norm_text = _normalize(q.question_text)
        matches = [i for i, c in enumerate(candidates) if i not in used and _snippet_matches(c.snippet, norm_text)]
        if len(matches) != 1:
            result.append(q)
            continue

        idx = matches[0]
        candidate = candidates[idx]
        used.add(idx)
        update = {}
        if needs_number and candidate.number:
            update["question_number"] = candidate.number
        if needs_options and candidate.options:
            update["options"] = candidate.options
        result.append(q.model_copy(update=update) if update else q)
    return result
