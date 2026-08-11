import pytest
from pydantic import ValidationError

from exampapersorter.config import DEFAULT_CONFIG
from exampapersorter.llm_client import LLMCallFailed
from exampapersorter.question_extraction import paper_extraction as paper_extraction_module
from exampapersorter.question_extraction.paper_extraction import (
    extract_paper_structure,
    extract_questions,
    extract_questions_for_paper,
)
from exampapersorter.schemas import (
    DetectedSection,
    EvidenceBlock,
    ExtractedQuestion,
    MetadataFieldExtraction,
    PageRangeEvidence,
    PaperStructureExtraction,
    QuestionExtractionResult,
)


def block(block_type, text, pages=(1,)):
    return EvidenceBlock(page_numbers=list(pages), block_type=block_type, text=text)


def make_evidence(blocks, start=1, end=1):
    return PageRangeEvidence(source_path="fake.pdf", start_page=start, end_page=end, conversion_status="success", blocks=blocks)


def field(value=None, evidence=None, confidence=0.9):
    return MetadataFieldExtraction(value=value, evidence=evidence, confidence=confidence)


def test_extract_paper_structure_returns_model_result(monkeypatch):
    expected = PaperStructureExtraction(institution=field("Some College", "Some College evidence"), sections=[])

    def fake_call_structured(config, system_prompt, user_prompt, response_model, call_label=None):
        assert response_model is PaperStructureExtraction
        return expected

    monkeypatch.setattr(paper_extraction_module, "call_structured", fake_call_structured)
    result = extract_paper_structure(make_evidence([block("title", "Some College")]), DEFAULT_CONFIG)
    assert result is expected


def test_extract_questions_includes_section_list_in_prompt(monkeypatch):
    sections = [DetectedSection(raw_label="II. Long Essay", section_type="essay", start_page=1, end_page=1, confidence=0.9)]
    captured = {}

    def fake_call_structured(config, system_prompt, user_prompt, response_model, call_label=None):
        captured["user_prompt"] = user_prompt
        return QuestionExtractionResult(questions=[])

    monkeypatch.setattr(paper_extraction_module, "call_structured", fake_call_structured)
    extract_questions(make_evidence([block("list_item", "1. Explain glycolysis")]), sections, DEFAULT_CONFIG)
    assert "II. Long Essay" in captured["user_prompt"]


def test_extract_questions_handles_no_sections(monkeypatch):
    captured = {}

    def fake_call_structured(config, system_prompt, user_prompt, response_model, call_label=None):
        captured["user_prompt"] = user_prompt
        return QuestionExtractionResult(questions=[])

    monkeypatch.setattr(paper_extraction_module, "call_structured", fake_call_structured)
    result = extract_questions(make_evidence([block("list_item", "1. a question")]), [], DEFAULT_CONFIG)
    assert result.questions == []
    assert "no sections were detected" in captured["user_prompt"]


def test_extract_questions_bundles_mcq_stem_and_options_before_rendering(monkeypatch):
    """The evidence handed to the LLM should already show the stem+options
    as one mcq_group block, not four separate list_items -- this is what
    actually fixes the option-fragmentation problem, independent of
    whatever the (faked here) LLM does with it."""
    blocks = [
        block("list_item", "1. Which enzyme catalyses step 1?"),
        block("list_item", "A. Hexokinase"),
        block("list_item", "B. Pepsin"),
        block("list_item", "C. Trypsin"),
    ]
    captured = {}

    def fake_call_structured(config, system_prompt, user_prompt, response_model, call_label=None):
        captured["user_prompt"] = user_prompt
        return QuestionExtractionResult(questions=[])

    monkeypatch.setattr(paper_extraction_module, "call_structured", fake_call_structured)
    extract_questions(make_evidence(blocks), [], DEFAULT_CONFIG)
    assert "(mcq_group)" in captured["user_prompt"]
    assert captured["user_prompt"].count("(list_item)") == 0
    assert "OPTION: A. Hexokinase" in captured["user_prompt"]


# --- MCQ options field ---


def test_mcq_options_round_trip_through_schema():
    q = ExtractedQuestion(
        question_number="1", question_text="Which of the following...?", question_type="mcq",
        options=["Trypsin", "Pepsin", "Lipase"], source_pages=[1], extraction_confidence=0.9,
    )
    assert q.options == ["Trypsin", "Pepsin", "Lipase"]


def test_non_mcq_question_has_empty_options_by_default():
    q = ExtractedQuestion(
        question_number="1", question_text="Explain glycolysis.", question_type="short_answer",
        source_pages=[1], extraction_confidence=0.9,
    )
    assert q.options == []


# --- OCR correction audit trail ---


def test_ocr_correction_preserves_original_text():
    q = ExtractedQuestion(
        question_number="1", question_text="Explain glycolysis.", original_text="Explain glyco1ysis.",
        correction_applied=True, question_type="short_answer", source_pages=[1], extraction_confidence=0.8,
    )
    assert q.correction_applied is True
    assert q.original_text == "Explain glyco1ysis."
    assert q.question_text == "Explain glycolysis."


def test_no_correction_leaves_original_text_null():
    q = ExtractedQuestion(
        question_number="1", question_text="Explain glycolysis.", question_type="short_answer",
        source_pages=[1], extraction_confidence=0.8,
    )
    assert q.correction_applied is False
    assert q.original_text is None


def test_correction_applied_without_original_text_is_rejected():
    """An unauditable correction claim is a schema-validation failure (so
    call_structured retries it), not something that silently passes."""
    with pytest.raises(ValidationError):
        ExtractedQuestion(
            question_number="1", question_text="Explain glycolysis.", correction_applied=True,
            question_type="short_answer", source_pages=[1], extraction_confidence=0.8,
        )


# --- metadata field value/evidence coupling ---


def test_metadata_field_value_without_evidence_is_rejected():
    with pytest.raises(ValidationError):
        MetadataFieldExtraction(value="2025", evidence=None)


def test_metadata_field_evidence_without_value_is_rejected():
    with pytest.raises(ValidationError):
        MetadataFieldExtraction(value=None, evidence="Date: 04-07-2025")


def test_metadata_field_null_value_and_evidence_is_fine():
    f = MetadataFieldExtraction()
    assert f.value is None and f.evidence is None


# --- ambiguity is preserved, not silently resolved ---


def test_unusual_section_label_is_not_forced_into_a_hardcoded_vocabulary():
    """DetectedSection.raw_label is free text -- an exam label this project's
    prompts never mention (e.g. "PART IV - VIVA VOCE") must still round-trip
    verbatim, with section_type falling back to "other"/"unknown" rather than
    the code silently remapping it to something it recognizes."""
    section = DetectedSection(
        raw_label="PART IV - VIVA VOCE", section_type="other", start_page=2, end_page=2, confidence=0.6,
    )
    assert section.raw_label == "PART IV - VIVA VOCE"
    assert section.section_type == "other"


def test_config_question_types_to_process_can_exclude_mcq_downstream():
    """Extraction never filters MCQs out -- config.question_types_to_process
    is how a downstream stage would choose to exclude them. This just
    confirms mcq is excluded by the default filter set while the answer
    section types remain included, i.e. the mechanism the requirements
    describe ("mark them clearly so downstream stages can exclude them
    through configuration") actually works end to end on the schema."""
    from exampapersorter.config import DEFAULT_CONFIG as CFG

    questions = [
        ExtractedQuestion(question_number="1", question_text="A or B?", question_type="mcq", source_pages=[1], extraction_confidence=0.9),
        ExtractedQuestion(question_number="2", question_text="Explain X.", question_type="long_answer", source_pages=[1], extraction_confidence=0.9),
    ]
    included = [q for q in questions if q.question_type in CFG.question_types_to_process]
    assert [q.question_number for q in included] == ["2"]


def test_ambiguous_question_keeps_its_note():
    q = ExtractedQuestion(
        question_number=None, question_text="...(illegible)...", question_type="unknown",
        source_pages=[2], extraction_confidence=0.2, ambiguous=True, ambiguity_note="text is cut off mid-sentence",
    )
    assert q.ambiguous is True
    assert q.question_number is None  # never invented
    assert q.ambiguity_note


# --- truncation-driven split fallback ---


def test_extract_questions_for_paper_returns_whole_paper_result_on_success(monkeypatch):
    expected = QuestionExtractionResult(questions=[])

    def fake_extract_questions(evidence, sections, config):
        return expected

    monkeypatch.setattr(paper_extraction_module, "extract_questions", fake_extract_questions)
    result, notes = extract_questions_for_paper(make_evidence([block("list_item", "1. q")]), [], DEFAULT_CONFIG)
    assert result is expected
    assert notes == []


def test_extract_questions_for_paper_reraises_non_truncation_failure(monkeypatch):
    def fake_extract_questions(evidence, sections, config):
        raise LLMCallFailed("schema-invalid JSON", truncated=False)

    monkeypatch.setattr(paper_extraction_module, "extract_questions", fake_extract_questions)
    with pytest.raises(LLMCallFailed):
        extract_questions_for_paper(make_evidence([block("list_item", "1. q")]), [], DEFAULT_CONFIG)


def test_extract_questions_for_paper_splits_by_section_on_truncation(monkeypatch):
    sections = [
        DetectedSection(raw_label="I. MCQ", section_type="mcq", start_page=1, end_page=1, confidence=0.9),
        DetectedSection(raw_label="II. Long Essay", section_type="long_answer", start_page=2, end_page=2, confidence=0.9),
    ]
    evidence = make_evidence(
        [block("list_item", "1. mcq stem", pages=(1,)), block("list_item", "1. essay stem", pages=(2,))],
        start=1, end=2,
    )

    call_count = {"whole": 0, "chunks": 0}

    def fake_extract_questions(chunk_evidence, chunk_sections, config):
        if chunk_evidence.start_page == 1 and chunk_evidence.end_page == 2:
            call_count["whole"] += 1
            raise LLMCallFailed("truncated", truncated=True)
        call_count["chunks"] += 1
        q = ExtractedQuestion(
            question_number="1", question_text=f"q on page {chunk_evidence.start_page}", question_type="mcq",
            source_pages=[chunk_evidence.start_page], extraction_confidence=0.8, section_index=0,
        )
        return QuestionExtractionResult(questions=[q])

    monkeypatch.setattr(paper_extraction_module, "extract_questions", fake_extract_questions)
    result, notes = extract_questions_for_paper(evidence, sections, DEFAULT_CONFIG)

    assert call_count["whole"] == 1
    assert call_count["chunks"] == 2
    assert len(result.questions) == 2
    # section_index was remapped from the chunk-local 0 to the real index
    # for each of the paper's two sections.
    assert {q.section_index for q in result.questions} == {0, 1}
    assert any("truncated" in n for n in notes)


def test_extract_questions_for_paper_splits_by_page_when_only_one_section(monkeypatch):
    evidence = make_evidence(
        [block("list_item", "1. q", pages=(1,)), block("list_item", "2. q", pages=(2,))],
        start=1, end=2,
    )
    section = [DetectedSection(raw_label="MCQ", section_type="mcq", start_page=1, end_page=2, confidence=0.9)]

    call_count = {"whole": 0, "pages": []}

    def fake_extract_questions(chunk_evidence, chunk_sections, config):
        if chunk_evidence.start_page != chunk_evidence.end_page:
            call_count["whole"] += 1
            raise LLMCallFailed("truncated", truncated=True)
        call_count["pages"].append(chunk_evidence.start_page)
        q = ExtractedQuestion(
            question_number=str(chunk_evidence.start_page), question_text="q", question_type="mcq",
            source_pages=[chunk_evidence.start_page], extraction_confidence=0.8,
        )
        return QuestionExtractionResult(questions=[q])

    monkeypatch.setattr(paper_extraction_module, "extract_questions", fake_extract_questions)
    result, notes = extract_questions_for_paper(evidence, section, DEFAULT_CONFIG)

    assert call_count["whole"] == 1
    assert call_count["pages"] == [1, 2]
    assert len(result.questions) == 2


def test_extract_questions_for_paper_raises_if_split_recovers_nothing(monkeypatch):
    section = [DetectedSection(raw_label="MCQ", section_type="mcq", start_page=1, end_page=1, confidence=0.9)]
    evidence = make_evidence([block("list_item", "1. q", pages=(1,))], start=1, end=1)

    def fake_extract_questions(chunk_evidence, chunk_sections, config):
        raise LLMCallFailed("truncated", truncated=True)

    monkeypatch.setattr(paper_extraction_module, "extract_questions", fake_extract_questions)
    with pytest.raises(LLMCallFailed):
        extract_questions_for_paper(evidence, section, DEFAULT_CONFIG)
