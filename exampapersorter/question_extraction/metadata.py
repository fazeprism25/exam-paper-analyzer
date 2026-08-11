"""Deterministic filename parsing + provenance-preserving metadata merge.

Two provenances feed a paper's metadata: what the LLM read out of the
document itself (PaperStructureExtraction), and what can be deterministically
parsed out of the filename. They are never silently reconciled -- see
merge_metadata. The filename side is intentionally limited to what a
generic regex can safely claim (years, dates, raw tokens): we do NOT try to
guess institution/subject/exam-name from filename text, since that would
mean inventing exam-specific parsing rules the project's constraints
explicitly rule out.
"""
from __future__ import annotations

import re
from pathlib import Path

from exampapersorter.schemas import (
    FilenameMetadata,
    MetadataFieldExtraction,
    MetadataFieldValue,
    PaperMetadata,
    PaperStructureExtraction,
)

_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
# No \b anchors: filenames commonly use "_" as the token separator around a
# date, and \b does not fire between "_" and a digit (both are \w) -- e.g.
# "exam_04-07-2025_final" would otherwise never match.
_DATE_RE = re.compile(r"(?<!\d)(\d{1,4})[-./](\d{1,2})[-./](\d{1,4})(?!\d)")
_TOKEN_SPLIT_RE = re.compile(r"[_\-\s]+")


def parse_filename_metadata(filename: str) -> FilenameMetadata:
    stem = Path(filename).stem
    years = sorted(set(_YEAR_RE.findall(stem)))
    dates = ["-".join(groups) for groups in _DATE_RE.findall(stem)]
    tokens = [t for t in _TOKEN_SPLIT_RE.split(stem) if t]
    return FilenameMetadata(raw_filename=filename, candidate_years=years, candidate_dates=dates, tokens=tokens)


def _digits_in(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


def _field(doc_field: MetadataFieldExtraction, filename_value: str | None, *, check_conflict: bool) -> MetadataFieldValue:
    """Never picks a winner. `conflict` is only ever set True (a positive
    claim that the two sources disagree) -- it defaults False whenever
    either side is missing, since "we only have one source" is not a
    conflict, it's just incomplete information."""
    conflict = False
    if check_conflict and doc_field.value and filename_value:
        dv, fv = doc_field.value.strip().lower(), filename_value.strip().lower()
        if dv != fv:
            dv_digits, fv_digits = _digits_in(dv), _digits_in(fv)
            if dv_digits and fv_digits and not (dv_digits & fv_digits):
                conflict = True
    return MetadataFieldValue(
        document_value=doc_field.value or None,
        document_evidence=doc_field.evidence,
        document_source_pages=doc_field.source_pages,
        document_confidence=doc_field.confidence,
        filename_value=filename_value or None,
        conflict=conflict,
    )


def merge_metadata(structure: PaperStructureExtraction, filename: str) -> PaperMetadata:
    fn = parse_filename_metadata(filename)
    filename_year = fn.candidate_years[0] if fn.candidate_years else None
    filename_date = fn.candidate_dates[0] if fn.candidate_dates else None

    return PaperMetadata(
        exam_name=_field(structure.exam_name, None, check_conflict=False),
        institution=_field(structure.institution, None, check_conflict=False),
        subject=_field(structure.subject, None, check_conflict=False),
        date=_field(structure.date, filename_date, check_conflict=True),
        year=_field(structure.year, filename_year, check_conflict=True),
        paper_identifier=_field(structure.paper_identifier, None, check_conflict=False),
        other_identifiers_document=structure.other_identifiers,
        other_identifiers_filename=fn.tokens,
    )


def check_cross_field_consistency(metadata: PaperMetadata) -> list[str]:
    """Deterministic plausibility check ACROSS a paper's own document-derived
    fields (not document-vs-filename -- that's MetadataFieldValue.conflict).
    Currently: does the year embedded in `date` (if any) agree with `year`?
    Both fields can independently be well-grounded in real evidence snippets
    and still disagree -- observed directly on a real fixture where `year`
    was read from one paper header ("...EXAMINATIONS, 2022") and `date` from
    a different snippet ("Date: 19-10-2023"), both real OCR text. This never
    picks a winner or drops either value; it only surfaces the disagreement
    as a note for human review, same as any other flagged-not-resolved
    ambiguity in this pipeline."""
    issues: list[str] = []
    year_val = metadata.year.document_value
    date_val = metadata.date.document_value
    if year_val and date_val:
        year_digits = set(_YEAR_RE.findall(year_val))
        date_digits = set(_YEAR_RE.findall(date_val))
        if year_digits and date_digits and not (year_digits & date_digits):
            issues.append(
                f"metadata inconsistency: year field is {year_val!r} but date field {date_val!r} "
                f"implies a different year -- both are evidence-grounded, neither was auto-resolved"
            )
    return issues
