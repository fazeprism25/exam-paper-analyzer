from exampapersorter.question_extraction.metadata import (
    check_cross_field_consistency,
    merge_metadata,
    parse_filename_metadata,
)
from exampapersorter.schemas import MetadataFieldExtraction, PaperStructureExtraction


def mf(value=None, evidence=None, confidence=0.9):
    """Test helper: a grounded MetadataFieldExtraction, or the null default
    when no value is given (evidence is meaningless without a value, and
    the schema itself now enforces that pairing -- see test_paper_extraction.py)."""
    if value is None:
        return MetadataFieldExtraction()
    return MetadataFieldExtraction(value=value, evidence=evidence or f"evidence for {value}", confidence=confidence)


def structure(**kwargs) -> PaperStructureExtraction:
    field_names = {"exam_name", "institution", "subject", "date", "year", "paper_identifier"}
    wrapped = {k: (mf(v) if k in field_names else v) for k, v in kwargs.items()}
    return PaperStructureExtraction(**wrapped)


# --- filename parsing ---


def test_parse_filename_metadata_extracts_year_and_tokens():
    meta = parse_filename_metadata("BIOCHEM_3rd_Sessionals_2024.pdf")
    assert meta.candidate_years == ["2024"]
    assert "BIOCHEM" in meta.tokens
    assert "3rd" in meta.tokens


def test_parse_filename_metadata_handles_no_date_or_year():
    meta = parse_filename_metadata("BIOCHEM 3RD SESSIONALS.pdf")
    assert meta.candidate_years == []
    assert meta.candidate_dates == []
    assert meta.tokens == ["BIOCHEM", "3RD", "SESSIONALS"]


def test_parse_filename_metadata_extracts_date_pattern():
    meta = parse_filename_metadata("exam_04-07-2025_final.pdf")
    assert meta.candidate_dates == ["04-07-2025"]


# --- merge: missing dates ---


def test_missing_date_on_both_sides_is_not_invented():
    result = merge_metadata(structure(), "no_date_here.pdf")
    assert result.date.document_value is None
    assert result.date.filename_value is None
    assert result.date.conflict is False
    assert result.year.document_value is None
    assert result.year.filename_value is None


# --- merge: conflicting filename/document dates ---


def test_conflicting_year_is_flagged_not_resolved():
    result = merge_metadata(structure(year="2025"), "biochem_2024_sessionals.pdf")
    assert result.year.document_value == "2025"
    assert result.year.filename_value == "2024"
    assert result.year.conflict is True
    # Neither value was dropped or overwritten -- both survive.


def test_agreeing_year_is_not_flagged():
    result = merge_metadata(structure(year="2024"), "biochem_2024_sessionals.pdf")
    assert result.year.conflict is False


def test_batch_range_document_year_overlapping_filename_year_is_not_a_conflict():
    # "2024-25" (a batch range, as this real textbook's exam papers use)
    # contains "2024" -- the filename's "2024" is corroborating, not conflicting.
    result = merge_metadata(structure(year="2024-25"), "biochem_2024_sessionals.pdf")
    assert result.year.conflict is False


# --- merge: never fabricates institution/subject/exam_name from filename ---


def test_filename_never_supplies_institution_or_subject_or_exam_name():
    result = merge_metadata(
        structure(institution="Some College", subject="Biochemistry", exam_name="Sessional Exam"),
        "biochem_sessional_2024.pdf",
    )
    assert result.institution.filename_value is None
    assert result.subject.filename_value is None
    assert result.exam_name.filename_value is None
    assert result.institution.document_value == "Some College"


def test_other_identifiers_preserved_from_both_sources_separately():
    result = merge_metadata(structure(other_identifiers=["Batch 2024-25"]), "biochem_final.pdf")
    assert result.other_identifiers_document == ["Batch 2024-25"]
    assert result.other_identifiers_filename == ["biochem", "final"]


# --- merge: document-side evidence/confidence carried through ---


def test_document_evidence_and_confidence_are_preserved_on_merge():
    result = merge_metadata(
        structure(institution="Some College"),
        "no_date_here.pdf",
    )
    assert result.institution.document_evidence == "evidence for Some College"
    assert result.institution.document_confidence == 0.9


def test_null_field_has_no_document_evidence_or_confidence():
    result = merge_metadata(structure(), "no_date_here.pdf")
    assert result.exam_name.document_value is None
    assert result.exam_name.document_evidence is None
    assert result.exam_name.document_confidence == 0.0


# --- cross-field consistency (within the document's own metadata) ---


def test_year_disagreeing_with_date_is_flagged():
    """Regression test for a real-fixture case: year='2022' was grounded in
    one snippet ("...EXAMINATIONS, 2022") and date='19-10-2023' in a
    different, also-real snippet -- both individually well-grounded, but
    mutually inconsistent."""
    result = merge_metadata(structure(year="2022", date="19-10-2023"), "no_date_here.pdf")
    issues = check_cross_field_consistency(result)
    assert len(issues) == 1
    assert "2022" in issues[0] and "19-10-2023" in issues[0]


def test_year_agreeing_with_date_is_not_flagged():
    result = merge_metadata(structure(year="2024", date="11-06-2024"), "no_date_here.pdf")
    assert check_cross_field_consistency(result) == []


def test_missing_year_or_date_is_not_flagged():
    result = merge_metadata(structure(year="2024"), "no_date_here.pdf")
    assert check_cross_field_consistency(result) == []
