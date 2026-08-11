"""Two LLM calls that operate on ONE already-isolated exam paper (a page
range already decided by paper_boundaries.py):

1. extract_paper_structure -- the paper's own identifying metadata (read
   from the document only; filename evidence is merged in separately, see
   metadata.py) and its internal section structure.
2. extract_questions_for_paper -- every individual question in the paper,
   informed by (but not constrained to exactly follow) the section list
   from (1). Tries the whole paper in one call first; if that call fails
   specifically because its response was truncated (see
   llm_client._looks_truncated / LLMCallFailed.truncated), falls back to
   splitting the extraction into smaller chunks (per section, or per page
   if there's only one section) rather than accepting an incomplete
   response or giving up outright -- see the module docstring on
   _split_and_extract for why per-section/page rather than any other split.

Kept as two calls rather than one so each stays small and tightly scoped:
metadata+sections is a short, low-token-count judgment, while question
extraction needs the full, unfiltered evidence for the whole paper and is
where OCR-correction/ambiguity judgment calls actually happen. This also
mirrors the pipeline the requirements describe: paper/document structure,
then individual question extraction, as two distinct stages.
"""
from __future__ import annotations

import logging

from exampapersorter.config import Config
from exampapersorter.evidence_format import render_evidence
from exampapersorter.llm_client import LLMCallFailed, call_structured
from exampapersorter.question_extraction.evidence_selection import slice_evidence
from exampapersorter.question_extraction.mcq_grouping import group_mcq_options
from exampapersorter.schemas import (
    DetectedSection,
    ExtractedQuestion,
    PageRangeEvidence,
    PaperStructureExtraction,
    QuestionExtractionResult,
)

logger = logging.getLogger(__name__)

STRUCTURE_SYSTEM_PROMPT = """You are reading ONE exam paper (already isolated from any other papers that \
may have been in the same scanned document) to identify its own identifying metadata and internal \
section structure.

For each metadata field (exam_name, institution, subject, date, year, paper_identifier), report an \
object with `value`, `evidence`, `source_pages`, and `confidence` -- never a bare string. `value` is the \
field's own text exactly as written (a batch range like "2024-25" should be reported as-is, not \
normalized). `evidence` MUST be the exact snippet of evidence text you read `value` from. If a field \
genuinely is not shown anywhere in the evidence, leave BOTH `value` and `evidence` null -- never report \
one without the other, and never invent a value.

Exam papers label their question groups in many different ways -- for example (not an exhaustive list): \
"LONG ESSAY", "LONG ESSAY QUESTIONS", "LONG ANSWER", "ESSAY QUESTIONS", "SHORT ESSAY", "SHORT ANSWER", \
"SHORT NOTES", "SECTION A", "SECTION B", "PART I", "PART II", "MCQ", "IMCQ". Read the paper's OWN \
wording into `raw_label` verbatim, then separately classify it into one of: mcq, long_answer, \
short_answer, essay, short_essay, other, unknown -- based on what the section's own questions actually \
ask for, not just the label text. Use "other" or "unknown" rather than forcing a poor fit. If a paper \
has no section headings at all, return an empty `sections` list rather than inventing one.

Never invent page numbers or evidence not present in what you were given."""


def extract_paper_structure(evidence: PageRangeEvidence, config: Config) -> PaperStructureExtraction:
    user_prompt = (
        f"Evidence from PDF pages {evidence.start_page}-{evidence.end_page} (one exam paper):\n\n"
        f"{render_evidence(evidence.blocks)}\n\n"
        "Extract this paper's identifying metadata and section structure."
    )
    return call_structured(
        config, STRUCTURE_SYSTEM_PROMPT, user_prompt, PaperStructureExtraction, call_label="paper_structure"
    )


QUESTIONS_SYSTEM_PROMPT = """You are extracting individual questions from ONE exam paper.

Some evidence blocks are labeled `(mcq_group)` -- these have ALREADY been deterministically bundled by \
our own code as one question stem plus its answer options (formatted as "STEM: ..." followed by one or \
more "OPTION: ..." lines). Every `(mcq_group)` block is exactly ONE question: put the stem text in \
`question_text`, put each option's text (without re-adding your own "A./B./..." marker) as its own entry \
in `options`, in the order given, and set `question_type` to "mcq". Do NOT split an mcq_group's options \
into separate questions, and do NOT leave `options` empty for one.

Some list_item blocks may instead be tagged "[UNASSOCIATED OPTION -- no preceding question stem was \
found for this]" or "[SUSPICIOUSLY LONG OPTION RUN -- left ungrouped rather than risk merging multiple \
questions]" -- these are answer-option-looking fragments our code could not confidently (or safely) \
attach to a stem. Do not guess which earlier question they belong to, and do NOT merge a long run of \
these into one question yourself. If you can confidently identify the correct stem from nearby context, \
you may bundle it yourself; otherwise extract each as its own record with `question_type` "mcq", \
`ambiguous` true, and an `ambiguity_note` explaining it could not be linked to a stem.

For every individual question you can identify:
- `question_number`: the number/label exactly as printed (e.g. "1", "2a", "IV"). If a question begins \
with a printed number/label, it MUST go here, not as part of `question_text`. If no number is visible, \
leave this null -- never invent one.
- `question_text`: the question stem only, as legibly readable -- do NOT include a leading number/label \
here (see question_number above), and for an MCQ do NOT append its options here (see `options` above).
- `options`: for `question_type` "mcq" only, each answer option's text in printed order. Empty list for \
every non-MCQ question.
- If the OCR text contains an obvious, unambiguous typo (e.g. "glyco1ysis" for "glycolysis", a stray "]" \
inserted mid-word), you may correct it in `question_text` -- but you MUST then also set `original_text` \
to the exact uncorrected OCR text and `correction_applied` to true, so the correction can be audited. If \
you make no correction, leave `original_text` null and `correction_applied` false. Do not set \
`correction_applied` true unless you are also providing `original_text` -- an unauditable correction \
claim will be rejected.
- If the evidence for a question is genuinely ambiguous (illegible, cut off, contradictory, or you are \
not confident of your reading), set `ambiguous` to true and briefly explain why in `ambiguity_note` -- do \
not guess at missing or unclear content.
- `question_type`: classify as one of mcq, long_answer, short_answer, essay, short_essay, other, unknown, \
based on the question's own phrasing/format and, if a section list is given below, which section it falls \
under.
- `source_pages`: the PDF page(s) the question's evidence came from.
- `extraction_confidence`: your own confidence (0-1) that this record is complete and accurate.
- `section_index`: if a section list is given below, the 0-based index of the section this question \
belongs to; null if it doesn't clearly belong to any of them, or if no section list is given.

Never invent a question that isn't present in the evidence."""


def _section_list_text(sections: list[DetectedSection]) -> str:
    if not sections:
        return "(no sections were detected for this paper)"
    return "\n".join(
        f"[{i}] {s.raw_label or s.section_type} (pages {s.start_page}-{s.end_page})" for i, s in enumerate(sections)
    )


def extract_questions(
    evidence: PageRangeEvidence, sections: list[DetectedSection], config: Config
) -> QuestionExtractionResult:
    grouped_blocks = group_mcq_options(evidence.blocks)
    user_prompt = (
        f"Sections detected in this paper:\n{_section_list_text(sections)}\n\n"
        f"Evidence from PDF pages {evidence.start_page}-{evidence.end_page} (one exam paper):\n\n"
        f"{render_evidence(grouped_blocks)}\n\n"
        "Extract every individual question."
    )
    return call_structured(
        config, QUESTIONS_SYSTEM_PROMPT, user_prompt, QuestionExtractionResult, call_label="question_extraction"
    )


def _shift_section_index(questions: list[ExtractedQuestion], real_index: int | None) -> list[ExtractedQuestion]:
    return [q.model_copy(update={"section_index": real_index}) for q in questions]


def _split_and_extract(
    evidence: PageRangeEvidence, sections: list[DetectedSection], config: Config
) -> tuple[QuestionExtractionResult, list[str]]:
    """Called only after a whole-paper extract_questions call failed with a
    truncated response (see extract_questions_for_paper). Splits the work
    into smaller chunks that each generate a smaller response, on the
    theory that the paper's total question volume -- not any single
    question -- is what pushed generation past num_predict. Prefers
    splitting by section (keeps each call's evidence topically coherent);
    falls back to splitting by page only when there's no more than one
    section to split by. Each chunk's own failure is caught and skipped
    (noted, not silently dropped) rather than failing the whole paper --
    partial recovery beats an all-or-nothing retry of a call already known
    to be too large."""
    notes: list[str] = []
    all_questions: list[ExtractedQuestion] = []

    if len(sections) > 1:
        chunks = [(s.start_page, s.end_page, [s], i) for i, s in enumerate(sections)]
    else:
        chunks = [(p, p, sections, None) for p in range(evidence.start_page, evidence.end_page + 1)]

    for start, end, chunk_sections, real_index in chunks:
        chunk_evidence = slice_evidence(evidence, start, end)
        if not chunk_evidence.blocks:
            continue
        try:
            result = extract_questions(chunk_evidence, chunk_sections, config)
        except LLMCallFailed as exc:
            notes.append(f"question extraction split-fallback: chunk pages {start}-{end} failed: {exc}")
            logger.error("Split fallback: chunk pages %d-%d failed: %s", start, end, exc)
            continue
        questions = result.questions
        if len(chunk_sections) == 1 and real_index is not None:
            # Chunk was scoped to exactly one real section, so every
            # question in this response belongs to it regardless of what
            # section_index (always 0, since chunk_sections has one entry)
            # the model reported.
            questions = _shift_section_index(questions, real_index)
        all_questions.extend(questions)

    return QuestionExtractionResult(questions=all_questions), notes


def extract_questions_for_paper(
    evidence: PageRangeEvidence, sections: list[DetectedSection], config: Config
) -> tuple[QuestionExtractionResult, list[str]]:
    """Orchestrates extract_questions with the truncation-driven split
    fallback. Returns (result, notes) -- notes is empty on the common path
    (single successful whole-paper call) and only populated when the split
    fallback was used, so callers can record what happened without every
    caller needing to know about the fallback mechanism."""
    try:
        return extract_questions(evidence, sections, config), []
    except LLMCallFailed as exc:
        if not exc.truncated:
            raise
        logger.warning(
            "Whole-paper question extraction truncated for pages %d-%d even after in-call retries; "
            "splitting into smaller chunks", evidence.start_page, evidence.end_page,
        )
        result, notes = _split_and_extract(evidence, sections, config)
        notes.insert(
            0,
            f"whole-paper question extraction was truncated ({exc}); recovered via per-"
            f"{'section' if len(sections) > 1 else 'page'} split",
        )
        if not result.questions:
            # The split fallback itself produced nothing usable -- surface
            # the original failure rather than silently returning an empty
            # result that looks identical to "this paper has zero questions".
            raise LLMCallFailed(
                f"Truncated on whole-paper call and split fallback recovered no questions: {exc}", truncated=True
            ) from exc
        return result, notes
