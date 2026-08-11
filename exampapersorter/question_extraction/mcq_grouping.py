"""Deterministic MCQ stem+option bundling.

Docling emits each MCQ answer option as its own list_item block, with no
built-in signal tying it back to its question stem. This was confirmed as a
real, not hypothetical, failure mode: the first real-fixture run produced
160/264 mcq-typed records that were bare option fragments ("Phenyl
Pyruvate", "B. 4", "Citrate") rather than a full question with its options
attached, because the LLM was asked to reconstruct the stem/option
relationship purely from a flat rendered list of same-looking list_items.

Rather than push harder on prompting alone, this groups option-like runs
with their preceding stem BEFORE the evidence is ever rendered for the
question-extraction call -- the LLM's job becomes "read this already-
bundled MCQ", not "figure out which of 20 nearby list_items belong
together". This only ever runs on the evidence handed to the
question-extraction prompt; it never touches the cached whole-document
evidence (which stays exactly what Docling produced).

An "option-like" block is one whose marker (Docling's own parsed list
marker -- see EvidenceBlock.marker) is a short, generic answer marker --
"A.", "B)", "(a)", "i.", "iv)" -- not a fixed exam vocabulary, just the
small set of ways human-authored answer choices are conventionally
lettered/numbered worldwide. Digits are deliberately excluded from this
pattern (reserved for question numbering, see question_number handling in
paper_extraction.py), so an option run can never be confused with a run of
numbered questions.

Marker, not text, is the primary signal: confirmed directly against the
real fixture that Docling frequently parses the marker OUT of `text`
entirely rather than leaving it concatenated -- an option's `text` can be
bare "Obstructive Jaundice" with "A." living only in `marker`. An earlier
version of this module matched against `text` alone and, on real OCR
output, left 86% of one paper's MCQ options ungrouped as a result. A
regex fallback over `text` is kept for blocks with no marker at all (e.g.
a born-digital PDF path that doesn't populate it), covering the more
common "A. Option text" concatenated form.

A run of >=2 consecutive option-like blocks is treated as one MCQ's
options. The block immediately preceding the run (within a short lookback
window) that is itself NOT option-like is treated as the stem. If no such
block is found, the options are left as individual blocks but their text is
tagged so the question-extraction prompt can point at them explicitly as
unassociated rather than silently guessing which stem they belong to --
never attached to an arbitrary preceding question.
"""
from __future__ import annotations

import re

from exampapersorter.schemas import EvidenceBlock

# Matches a marker in isolation, e.g. Docling's own block.marker value
# ("A.", "(iii)") compared as a whole string.
_MARKER_WHOLE = re.compile(r"^\(?([A-Da-d]|[ivx]{1,4})\)?[.\):]?$")
# Fallback for blocks with no separate marker at all: a prefix of `text`
# itself. No \s+ requirement after the punctuation -- real OCR on the
# actual fixture routinely concatenates the letter straight onto the
# option text with no space ("A.Glycine").
_MARKER_TEXT_PREFIX = re.compile(r"^\s*\(?([A-Da-d]|[ivx]{1,4})\)?[.\):]\s*(?=\S)")

_GROUPABLE_TYPES = {"list_item", "text"}

# How far back (in block-list positions, not pages) to look for a stem once
# an option run is found. Kept short and deliberately format-agnostic --
# this is about the stem being the block immediately before its options
# started, not about a page-distance heuristic.
_STEM_LOOKBACK = 3

# A marks/marking-scheme annotation (e.g. "10X1=10", "5x3=15", "[1+2]",
# "1X10 [1+1+4+4]") commonly sits BETWEEN a numbered stem and its option
# run -- Docling emits it as its own text block, not attached to either
# side. Confirmed directly against the real fixture: without this check,
# the stem-lookback below picked "10X1=10" itself as the "stem" for the
# first MCQ of a section (it's a groupable, non-option-like block
# immediately preceding the options), leaving the REAL numbered stem
# ("1. The most important intracellular buffer") orphaned as a separate,
# un-bundled block one position further back. Generic on purpose: a marks
# line is built entirely from digits/whitespace/arithmetic symbols, while
# a real question stem always contains actual words -- this is true
# regardless of exam format or subject, not a fact about this one fixture.
_MARKS_ANNOTATION = re.compile(r"^[\s\d xX+\-=./,\[\]()]+$")


def _is_marks_annotation(block: EvidenceBlock) -> bool:
    return block.block_type == "text" and bool(_MARKS_ANNOTATION.match(block.text.strip())) and block.text.strip() != ""

_UNASSOCIATED_TAG = "[UNASSOCIATED OPTION -- no preceding question stem was found for this]"

# A real MCQ practically never has more than ~6 options. A longer run is
# more likely several real questions' options bleeding together (e.g. a
# garbled/missed stem between two MCQs that let the run keep extending) --
# observed directly on the real fixture before marker-based detection was
# added: a 16-option run merged four unrelated questions into one record.
# Rather than silently accept a merge that large, leave the run ungrouped
# and flagged so it's visible for review instead of silently wrong.
_MAX_RUN_LENGTH = 6
_SUSPICIOUS_RUN_TAG = "[SUSPICIOUSLY LONG OPTION RUN -- left ungrouped rather than risk merging multiple questions]"


def _is_option_like(block: EvidenceBlock) -> bool:
    if block.block_type not in _GROUPABLE_TYPES:
        return False
    if block.marker is not None:
        return bool(_MARKER_WHOLE.match(block.marker.strip()))
    return bool(_MARKER_TEXT_PREFIX.match(block.text))


def group_mcq_options(blocks: list[EvidenceBlock]) -> list[EvidenceBlock]:
    result: list[EvidenceBlock] = []
    consumed: set[int] = set()
    n = len(blocks)
    i = 0
    while i < n:
        if i in consumed:
            i += 1
            continue

        if not _is_option_like(blocks[i]):
            result.append(blocks[i])
            i += 1
            continue

        # Found the start of an option-like run; extend it.
        j = i
        while j < n and _is_option_like(blocks[j]):
            j += 1
        run = blocks[i:j]

        if len(run) < 2:
            result.append(blocks[i])
            i += 1
            continue

        if len(run) > _MAX_RUN_LENGTH:
            for opt in run:
                display = f"{opt.marker} {opt.text}" if opt.marker else opt.text
                result.append(opt.model_copy(update={"text": f"{_SUSPICIOUS_RUN_TAG} {display}"}))
            consumed.update(range(i, j))
            i = j
            continue

        stem_idx = None
        for k in range(i - 1, max(-1, i - 1 - _STEM_LOOKBACK), -1):
            if k in consumed:
                break  # already part of another group -- don't reach past it
            if _is_marks_annotation(blocks[k]):
                continue  # not a real stem candidate -- keep looking further back
            if blocks[k].block_type in _GROUPABLE_TYPES and not _is_option_like(blocks[k]):
                stem_idx = k
                break

        if stem_idx is not None:
            stem = blocks[stem_idx]

            def _display(b: EvidenceBlock) -> str:
                # Reattach the marker Docling parsed out separately, purely
                # for human/LLM legibility in the bundled text -- the
                # underlying `options` field the LLM produces should still
                # be marker-free (see ExtractedQuestion.options).
                return f"{b.marker} {b.text}" if b.marker else b.text

            pages = sorted({p for b in [stem, *run] for p in b.page_numbers})
            text = "STEM: " + _display(stem) + "\n" + "\n".join(f"OPTION: {_display(opt)}" for opt in run)
            merged = EvidenceBlock(page_numbers=pages, block_type="mcq_group", text=text)
            # The stem was already appended to `result` in an earlier
            # iteration (it wasn't option-like) -- replace it in place.
            for idx, placed in enumerate(result):
                if placed is stem:
                    result[idx] = merged
                    break
            else:
                result.append(merged)
            consumed.add(stem_idx)
            consumed.update(range(i, j))
        else:
            for opt in run:
                display = f"{opt.marker} {opt.text}" if opt.marker else opt.text
                result.append(opt.model_copy(update={"text": f"{_UNASSOCIATED_TAG} {display}"}))
            consumed.update(range(i, j))

        i = j

    return result
