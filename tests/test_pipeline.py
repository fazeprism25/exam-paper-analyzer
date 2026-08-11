from dataclasses import replace
from pathlib import Path

from exampapersorter.config import DEFAULT_CONFIG
from exampapersorter.database import Database
from exampapersorter.llm_client import LLMCallFailed
from exampapersorter.question_extraction import pipeline as pipeline_module
from exampapersorter.schemas import (
    DetectedSection,
    EvidenceBlock,
    ExtractedQuestion,
    MetadataFieldExtraction,
    PageRangeEvidence,
    PaperBoundaryCandidate,
    PaperBoundaryDetectionResult,
    PaperStructureExtraction,
    QuestionExtractionResult,
)

FILE_HASH = "deadbeef1234"


def block(block_type, text, pages=(1,)):
    return EvidenceBlock(page_numbers=list(pages), block_type=block_type, text=text)


def fake_full_evidence(total_pages=6):
    return PageRangeEvidence(
        source_path="fake.pdf", start_page=1, end_page=total_pages, conversion_status="success",
        blocks=[block("title", f"header on page {p}", pages=(p,)) for p in range(1, total_pages + 1)],
    )


def default_structure_result() -> PaperStructureExtraction:
    return PaperStructureExtraction(
        institution=MetadataFieldExtraction(value="Some College", evidence="Some College letterhead"), sections=[]
    )


def install_fakes(
    monkeypatch,
    *,
    total_pages=6,
    boundary_result=None,
    structure_result=None,
    questions_result=None,
    structure_side_effect=None,
    questions_side_effect=None,
):
    calls = {"docling": 0, "boundaries": 0, "structure": 0, "questions": 0, "unload": 0}

    def fake_extract_page_range_evidence(path, start, end):
        calls["docling"] += 1
        return fake_full_evidence(total_pages)

    def fake_detect_paper_boundaries(evidence, config):
        calls["boundaries"] += 1
        return boundary_result or PaperBoundaryDetectionResult(
            papers=[PaperBoundaryCandidate(start_page=1, end_page=total_pages, confidence=0.9, evidence=[])]
        )

    def fake_extract_paper_structure(evidence, config):
        calls["structure"] += 1
        if structure_side_effect:
            raise structure_side_effect
        return structure_result or default_structure_result()

    def fake_extract_questions_for_paper(evidence, sections, config):
        calls["questions"] += 1
        if questions_side_effect:
            raise questions_side_effect
        return (questions_result or QuestionExtractionResult(questions=[])), []

    def fake_unload(config):
        calls["unload"] += 1
        return True

    monkeypatch.setattr(pipeline_module, "extract_page_range_evidence", fake_extract_page_range_evidence)
    monkeypatch.setattr(pipeline_module, "detect_paper_boundaries", fake_detect_paper_boundaries)
    monkeypatch.setattr(pipeline_module, "extract_paper_structure", fake_extract_paper_structure)
    monkeypatch.setattr(pipeline_module, "extract_questions_for_paper", fake_extract_questions_for_paper)
    monkeypatch.setattr(pipeline_module, "unload_current_model", fake_unload)
    return calls


def run_pipeline(db, config=DEFAULT_CONFIG, total_pages=6):
    return pipeline_module.process_question_paper_file(Path("fake.pdf"), FILE_HASH, total_pages, config, db)


# --- one paper per PDF ---


def test_single_paper_pdf_produces_one_paper(tmp_path, monkeypatch):
    install_fakes(monkeypatch, total_pages=5)
    db = Database(tmp_path / "test.db")
    result = run_pipeline(db, total_pages=5)

    assert result.status == "success"
    assert len(result.papers) == 1
    assert result.papers[0].start_page == 1 and result.papers[0].end_page == 5
    db.close()


# --- multiple papers in one PDF ---


def test_multi_paper_pdf_produces_multiple_papers(tmp_path, monkeypatch):
    boundary_result = PaperBoundaryDetectionResult(
        papers=[
            PaperBoundaryCandidate(start_page=1, end_page=3, confidence=0.8, evidence=[]),
            PaperBoundaryCandidate(start_page=4, end_page=6, confidence=0.9, evidence=[]),
        ]
    )
    calls = install_fakes(monkeypatch, total_pages=6, boundary_result=boundary_result)
    db = Database(tmp_path / "test.db")
    result = run_pipeline(db)

    assert len(result.papers) == 2
    assert [p.paper_id for p in result.papers] == [f"{FILE_HASH[:12]}_paper1", f"{FILE_HASH[:12]}_paper2"]
    assert calls["structure"] == 2
    assert calls["questions"] == 2
    db.close()


# --- empty extraction ---


def test_no_questions_extracted_is_not_an_error(tmp_path, monkeypatch):
    install_fakes(monkeypatch, questions_result=QuestionExtractionResult(questions=[]))
    db = Database(tmp_path / "test.db")
    result = run_pipeline(db)

    assert result.status == "success"
    assert db.get_questions_for_paper(result.papers[0].paper_id) == []
    db.close()


# --- malformed LLM output -> failure isolation, never fabricated ---


def test_structure_failure_marks_paper_failed_without_crashing_the_run(tmp_path, monkeypatch):
    calls = install_fakes(monkeypatch, structure_side_effect=LLMCallFailed("malformed output, retries exhausted"))
    db = Database(tmp_path / "test.db")
    result = run_pipeline(db)

    # File-level status is "partial", not "failed" -- boundary detection
    # succeeded and the paper record IS visible (with status="failed" of
    # its own), just incomplete. "failed" at the file level is reserved for
    # when nothing could be processed at all (e.g. Docling/boundary failure).
    assert result.status == "partial"
    assert result.papers[0].status == "failed"
    assert calls["questions"] == 0  # never reached -- structure failed first
    db.close()


def test_question_extraction_failure_still_preserves_structure(tmp_path, monkeypatch):
    install_fakes(monkeypatch, questions_side_effect=LLMCallFailed("malformed output, retries exhausted"))
    db = Database(tmp_path / "test.db")
    result = run_pipeline(db)

    assert result.status == "partial"
    assert result.papers[0].status == "partial"
    assert result.papers[0].metadata.institution.document_value == "Some College"
    db.close()


def test_one_paper_failing_does_not_block_other_papers(tmp_path, monkeypatch):
    boundary_result = PaperBoundaryDetectionResult(
        papers=[
            PaperBoundaryCandidate(start_page=1, end_page=3, confidence=0.8, evidence=[]),
            PaperBoundaryCandidate(start_page=4, end_page=6, confidence=0.9, evidence=[]),
        ]
    )
    install_fakes(monkeypatch, total_pages=6, boundary_result=boundary_result)
    call_count = {"n": 0}

    def flaky_structure(evidence, config):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise LLMCallFailed("first paper's structure call failed")
        return default_structure_result()

    monkeypatch.setattr(pipeline_module, "extract_paper_structure", flaky_structure)

    db = Database(tmp_path / "test.db")
    result = run_pipeline(db)

    assert result.status == "partial"
    assert [p.status for p in result.papers] == ["failed", "success"]
    db.close()


# --- database resume behavior / idempotent re-runs ---


def test_rerunning_a_fully_successful_file_does_not_recall_the_llm(tmp_path, monkeypatch):
    calls = install_fakes(monkeypatch)
    db = Database(tmp_path / "test.db")
    run_pipeline(db)
    assert calls["boundaries"] == 1 and calls["structure"] == 1 and calls["questions"] == 1

    result2 = run_pipeline(db)
    # No new LLM calls at all -- the whole file was already marked success.
    assert calls["boundaries"] == 1 and calls["structure"] == 1 and calls["questions"] == 1
    assert result2.status == "success"
    assert len(result2.papers) == 1
    db.close()


def test_resume_after_partial_failure_reuses_cached_boundary_and_retries_only_the_failed_paper(tmp_path, monkeypatch):
    boundary_result = PaperBoundaryDetectionResult(
        papers=[
            PaperBoundaryCandidate(start_page=1, end_page=3, confidence=0.8, evidence=[]),
            PaperBoundaryCandidate(start_page=4, end_page=6, confidence=0.9, evidence=[]),
        ]
    )
    fail_first_call = {"failed_once": False}

    def flaky_structure(evidence, config):
        if evidence.start_page == 1 and not fail_first_call["failed_once"]:
            fail_first_call["failed_once"] = True
            raise LLMCallFailed("transient failure")
        return default_structure_result()

    calls = install_fakes(monkeypatch, total_pages=6, boundary_result=boundary_result)
    monkeypatch.setattr(pipeline_module, "extract_paper_structure", flaky_structure)

    db = Database(tmp_path / "test.db")
    result1 = run_pipeline(db)
    assert result1.status == "partial"
    assert calls["boundaries"] == 1  # only ever called once

    result2 = run_pipeline(db)
    assert calls["boundaries"] == 1  # still not re-called on resume
    assert result2.status == "success"
    assert [p.status for p in result2.papers] == ["success", "success"]
    db.close()


# --- split-fallback notes are threaded through to the Paper record ---


def test_split_fallback_notes_are_recorded_on_the_paper(tmp_path, monkeypatch):
    calls = install_fakes(monkeypatch)

    def fake_extract_questions_for_paper(evidence, sections, config):
        calls["questions"] += 1
        return QuestionExtractionResult(questions=[]), ["whole-paper question extraction was truncated; recovered via split"]

    monkeypatch.setattr(pipeline_module, "extract_questions_for_paper", fake_extract_questions_for_paper)
    db = Database(tmp_path / "test.db")
    result = run_pipeline(db)

    assert result.papers[0].status == "success"
    assert any("truncated" in n for n in result.papers[0].notes)
    db.close()


# --- reload_per_paper policy wiring ---


def test_reload_per_paper_unloads_before_each_paper(tmp_path, monkeypatch):
    boundary_result = PaperBoundaryDetectionResult(
        papers=[
            PaperBoundaryCandidate(start_page=1, end_page=3, confidence=0.8, evidence=[]),
            PaperBoundaryCandidate(start_page=4, end_page=6, confidence=0.9, evidence=[]),
        ]
    )
    calls = install_fakes(monkeypatch, total_pages=6, boundary_result=boundary_result)
    config = replace(DEFAULT_CONFIG, llm_reload_policy="reload_per_paper")
    db = Database(tmp_path / "test.db")
    run_pipeline(db, config=config)
    assert calls["unload"] == 2
    db.close()


def test_persistent_policy_never_unloads_between_papers(tmp_path, monkeypatch):
    boundary_result = PaperBoundaryDetectionResult(
        papers=[
            PaperBoundaryCandidate(start_page=1, end_page=3, confidence=0.8, evidence=[]),
            PaperBoundaryCandidate(start_page=4, end_page=6, confidence=0.9, evidence=[]),
        ]
    )
    calls = install_fakes(monkeypatch, total_pages=6, boundary_result=boundary_result)
    db = Database(tmp_path / "test.db")
    run_pipeline(db)
    assert calls["unload"] == 0
    db.close()


# --- section_index wiring on extracted questions ---


def test_question_section_index_maps_to_persisted_section_id(tmp_path, monkeypatch):
    structure_result = PaperStructureExtraction(
        institution=MetadataFieldExtraction(value="Some College", evidence="Some College letterhead"),
        sections=[DetectedSection(raw_label="II. Long Essay", section_type="essay", start_page=1, end_page=6, confidence=0.9)],
    )
    questions_result = QuestionExtractionResult(
        questions=[
            ExtractedQuestion(
                question_number="1", question_text="Explain glycolysis.", question_type="essay",
                source_pages=[1], extraction_confidence=0.9, section_index=0,
            )
        ]
    )
    install_fakes(monkeypatch, structure_result=structure_result, questions_result=questions_result)
    db = Database(tmp_path / "test.db")
    result = run_pipeline(db)

    questions = db.get_questions_for_paper(result.papers[0].paper_id)
    assert len(questions) == 1
    assert questions[0].section_id == result.papers[0].sections[0].section_id
    assert questions[0].source_filename == "fake.pdf"
    assert questions[0].source_file_hash == FILE_HASH
    assert questions[0].institution == "Some College"
    db.close()


# --- deterministic question-number backfill ---


def test_question_number_is_backfilled_from_leading_digit_in_text(tmp_path, monkeypatch):
    questions_result = QuestionExtractionResult(
        questions=[
            ExtractedQuestion(
                question_number=None, question_text="16. Hypocalcaemia can occur in all except:",
                question_type="mcq", source_pages=[1], extraction_confidence=0.8,
            )
        ]
    )
    install_fakes(monkeypatch, questions_result=questions_result)
    db = Database(tmp_path / "test.db")
    result = run_pipeline(db)

    questions = db.get_questions_for_paper(result.papers[0].paper_id)
    assert questions[0].question_number == "16"
    assert questions[0].question_text == "Hypocalcaemia can occur in all except:"
    db.close()


# --- deterministic reconciliation (question_number / MCQ options) ---


def test_reconciliation_backfills_dropped_mcq_options_from_fresh_extraction(tmp_path, monkeypatch):
    mcq_blocks = [
        block("list_item", "1. Which enzyme catalyses step 1?"),
        block("list_item", "A. Hexokinase"),
        block("list_item", "B. Pepsin"),
        block("list_item", "C. Trypsin"),
    ]

    def fake_extract_page_range_evidence(path, start, end):
        return PageRangeEvidence(
            source_path=str(path), start_page=1, end_page=1, conversion_status="success", blocks=mcq_blocks,
        )

    questions_result = QuestionExtractionResult(
        questions=[
            ExtractedQuestion(
                question_number=None, question_text="Which enzyme catalyses step 1?",
                question_type="mcq", options=[], source_pages=[1], extraction_confidence=0.9,
            )
        ]
    )
    install_fakes(monkeypatch, total_pages=1, questions_result=questions_result)
    monkeypatch.setattr(pipeline_module, "extract_page_range_evidence", fake_extract_page_range_evidence)

    db = Database(tmp_path / "test.db")
    result = run_pipeline(db, total_pages=1)

    questions = db.get_questions_for_paper(result.papers[0].paper_id)
    assert questions[0].question_number == "1"
    assert questions[0].options == ["A. Hexokinase", "B. Pepsin", "C. Trypsin"]
    db.close()


def test_reconciliation_also_applies_to_a_cached_question_extraction_verdict(tmp_path, monkeypatch):
    """This is the whole reason reconciliation lives in pipeline.py rather
    than inside extract_questions: a resumed run that hits the
    question_extraction_log cache (e.g. built by an older version of this
    code, before reconciliation existed) must still benefit from it -- not
    just a freshly-made LLM call."""
    mcq_blocks = [
        block("list_item", "1. Which enzyme catalyses step 1?"),
        block("list_item", "A. Hexokinase"),
        block("list_item", "B. Pepsin"),
        block("list_item", "C. Trypsin"),
    ]

    def fake_extract_page_range_evidence(path, start, end):
        return PageRangeEvidence(
            source_path=str(path), start_page=1, end_page=1, conversion_status="success", blocks=mcq_blocks,
        )

    calls = install_fakes(monkeypatch, total_pages=1)
    monkeypatch.setattr(pipeline_module, "extract_page_range_evidence", fake_extract_page_range_evidence)

    db = Database(tmp_path / "test.db")
    stale_cached_result = QuestionExtractionResult(
        questions=[
            ExtractedQuestion(
                question_number=None, question_text="Which enzyme catalyses step 1?",
                question_type="mcq", options=[], source_pages=[1], extraction_confidence=0.9,
            )
        ]
    )
    db.save_question_extraction_verdict(FILE_HASH, 1, 1, DEFAULT_CONFIG.model_identifier, stale_cached_result)

    result = run_pipeline(db, total_pages=1)

    assert calls["questions"] == 0  # never called -- the cached verdict was used
    questions = db.get_questions_for_paper(result.papers[0].paper_id)
    assert questions[0].question_number == "1"
    assert questions[0].options == ["A. Hexokinase", "B. Pepsin", "C. Trypsin"]
    db.close()


# --- OCR correction audit trail, end to end ---


def test_ocr_correction_survives_the_full_pipeline_to_the_persisted_question(tmp_path, monkeypatch):
    """The schema-level validator (test_paper_extraction.py) already proves
    correction_applied=True requires original_text at the ExtractedQuestion
    level in isolation. This is the gap the real run left uncovered (see
    project notes: correction_applied fired 0 times in the real run, so this
    path was never exercised past unit level) -- that the audit trail
    actually survives assembly into a Question record AND a DB round trip,
    not just Pydantic construction."""
    questions_result = QuestionExtractionResult(
        questions=[
            ExtractedQuestion(
                question_number="1", question_text="Explain glycolysis.",
                original_text="Explain glyco1ysis.", correction_applied=True,
                question_type="short_answer", source_pages=[1], extraction_confidence=0.85,
            )
        ]
    )
    install_fakes(monkeypatch, questions_result=questions_result)
    db = Database(tmp_path / "test.db")
    result = run_pipeline(db)

    questions = db.get_questions_for_paper(result.papers[0].paper_id)
    assert len(questions) == 1
    assert questions[0].correction_applied is True
    assert questions[0].original_text == "Explain glyco1ysis."
    assert questions[0].question_text == "Explain glycolysis."
    db.close()


def test_model_supplied_question_number_is_not_overridden(tmp_path, monkeypatch):
    questions_result = QuestionExtractionResult(
        questions=[
            ExtractedQuestion(
                question_number="16a", question_text="16. Hypocalcaemia can occur in all except:",
                question_type="mcq", source_pages=[1], extraction_confidence=0.8,
            )
        ]
    )
    install_fakes(monkeypatch, questions_result=questions_result)
    db = Database(tmp_path / "test.db")
    result = run_pipeline(db)

    questions = db.get_questions_for_paper(result.papers[0].paper_id)
    assert questions[0].question_number == "16a"
    assert questions[0].question_text == "16. Hypocalcaemia can occur in all except:"
    db.close()
