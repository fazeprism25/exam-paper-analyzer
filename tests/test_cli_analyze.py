import argparse
import logging
from dataclasses import replace
from types import SimpleNamespace

import pymupdf
import pytest

import cli as cli_module
from exampapersorter import analyze_pipeline as analyze_pipeline_module
from exampapersorter.pdf_utils import compute_file_hash
from exampapersorter.schemas import (
    DataQualityNotes,
    ExecutiveSummary,
    FinalAnalysisReport,
    MetadataFieldValue,
    Paper,
    PaperMetadata,
    Question,
    QuestionPaperFileResult,
    QuestionTopicClassification,
    QuestionTypeDistributionReport,
    Topic,
)
from exampapersorter.topic_authority import TopicAuthorityResult
from exampapersorter.topic_classification.pipeline import PaperClassificationResult


def _base_args(**overrides):
    defaults = dict(
        textbook=None, index=None, question_papers=None, provider=None, openrouter_api_key=None,
        save_key=False, model=None, seed=None, reload_policy=None, debug=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# --- Input validation (must fail before any expensive work) ---


def test_missing_topic_source_exits_with_clear_error(tmp_path, caplog):
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    args = _base_args(question_papers=str(papers_dir))

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as exc_info:
            cli_module.cmd_analyze(args)
    assert exc_info.value.code == 1
    assert "topic source" in caplog.text.lower()


def test_missing_question_papers_directory_exits_with_clear_error(tmp_path, caplog):
    args = _base_args(textbook="somebook.pdf", question_papers=str(tmp_path / "does_not_exist"))

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as exc_info:
            cli_module.cmd_analyze(args)
    assert exc_info.value.code == 1
    assert "not found" in caplog.text.lower()


def test_empty_question_papers_directory_exits_with_clear_error(tmp_path, caplog):
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    args = _base_args(textbook="somebook.pdf", question_papers=str(papers_dir))

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as exc_info:
            cli_module.cmd_analyze(args)
    assert exc_info.value.code == 1
    assert "no pdf files found" in caplog.text.lower()


def test_missing_api_key_exits_before_any_pdf_processing(tmp_path, monkeypatch, caplog):
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "paper1.pdf").write_bytes(b"%PDF-1.4 fake")
    args = _base_args(textbook="somebook.pdf", question_papers=str(papers_dir))

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(cli_module.api_key_store, "load_saved_key", lambda: None)

    called = {"authority": False}
    monkeypatch.setattr(
        cli_module, "resolve_textbook_topic_authority",
        lambda *a, **k: called.__setitem__("authority", True),
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as exc_info:
            cli_module.cmd_analyze(args)
    assert exc_info.value.code == 1
    assert "no openrouter api key" in caplog.text.lower()
    assert called["authority"] is False  # must fail before touching the topic source / doing any PDF work


# --- Full orchestration, with Stage 2-6 internals mocked out (already
# covered by their own test suites -- this only tests that cmd_analyze
# wires them together correctly) ---


def test_cmd_analyze_happy_path_end_to_end_with_mocked_stages(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "t.db"
    output_dir = tmp_path / "output"
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()

    pdf_path = papers_dir / "paper1.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    file_hash = compute_file_hash(pdf_path)
    paper_id = f"{file_hash[:12]}_paper1"

    topic = Topic(id="t1", name="Topic 1", level=1, source_pages=[])
    authority = TopicAuthorityResult(
        file_hash="authorityhash1234567890abcdef", source_path=str(tmp_path / "syllabus.md"),
        source_type="index_outline", topics=[topic], validation=None, from_cache=False,
    )

    empty_meta = MetadataFieldValue()
    metadata = PaperMetadata(
        exam_name=empty_meta, institution=empty_meta, subject=empty_meta,
        date=empty_meta, year=empty_meta, paper_identifier=empty_meta,
    )

    def fake_resolve_index(index_path, config, db):
        return authority

    def fake_process_question_paper_file(pdf_path_, file_hash_, total_pages, config, db):
        question = Question(
            question_id=f"{paper_id}_q1", paper_id=paper_id, section_id=None,
            source_filename=pdf_path_.name, source_file_hash=file_hash_,
            question_number="1", question_text="What is metabolism?", question_type="short_answer",
            source_pages=[1], extraction_confidence=0.9,
        )
        paper = Paper(
            paper_id=paper_id, file_hash=file_hash_, start_page=1, end_page=1, metadata=metadata,
            sections=[], boundary_confidence=0.9, status="success",
        )
        db.register_question_paper_file(file_hash_, str(pdf_path_), total_pages)
        db.save_paper(paper)
        db.save_questions([question])
        return QuestionPaperFileResult(
            file_path=str(pdf_path_), file_hash=file_hash_, total_pages=total_pages,
            model_used="fake-model", papers=[paper], boundary_issues=[], status="success",
        )

    def fake_classify_paper(paper, topics, textbook_file_hash, config, db):
        classification = QuestionTopicClassification(
            question_id=f"{paper_id}_q1", status="classified", topic_id="t1", confidence=0.9,
        )
        db.save_question_classifications([classification], {t.id: t for t in topics})
        return PaperClassificationResult(
            paper_id=paper.paper_id, status="success", total_questions=1, classified_count=1,
        )

    dedup_result = SimpleNamespace(
        total_questions=1, exact_duplicate_groups=0, candidate_pairs_generated=0,
        llm_pairs_reused_from_cache=0, llm_pairs_newly_judged=0, llm_batches_failed=0,
        canonical_group_count=1, status_counts={"singleton": 1},
    )

    report = FinalAnalysisReport(
        textbook_file_hash=authority.file_hash,
        summary=ExecutiveSummary(
            textbook_file_hash=authority.file_hash, total_papers=1, total_question_occurrences=1,
            total_canonical_questions=1, total_topics=1, topics_tested=1, topics_untested=0,
            repeated_canonical_question_count=0, years_represented=[], no_match_count=0,
            ambiguous_count=0, unclassified_count=0, data_integrity_reconciled=True,
        ),
        question_type_distribution=QuestionTypeDistributionReport(total_occurrences=1, by_type=[]),
        most_repeated_questions=[], most_tested_topics_by_occurrences=[], most_tested_topics_by_unique_questions=[],
        topic_analysis=[], untested_topics=[], no_match_questions=[], ambiguous_questions=[],
        data_quality=DataQualityNotes(
            no_match_count=0, ambiguous_count=0, unclassified_count=0,
            year_conflict_paper_count=0, data_integrity_reconciled=True, notes=[],
        ),
    )

    monkeypatch.setattr(cli_module, "DEFAULT_CONFIG", replace(cli_module.DEFAULT_CONFIG, database_path=db_path, output_directory=output_dir))
    monkeypatch.setattr(cli_module, "validate_api_key", lambda config: (True, "ok"))
    monkeypatch.setattr(cli_module, "resolve_index_topic_authority", fake_resolve_index)
    monkeypatch.setattr(analyze_pipeline_module, "process_question_paper_file", fake_process_question_paper_file)
    monkeypatch.setattr(analyze_pipeline_module, "classify_paper", fake_classify_paper)
    monkeypatch.setattr(cli_module, "run_deduplication", lambda config, db: dedup_result)
    monkeypatch.setattr(cli_module, "run_final_analysis", lambda db, file_hash: report)

    args = _base_args(
        index=str(tmp_path / "syllabus.md"), question_papers=str(papers_dir), openrouter_api_key="sk-fake-key",
    )

    cli_module.cmd_analyze(args)

    out = capsys.readouterr().out
    assert "[OK] Topic authority loaded" in out
    assert "Questions extracted: 1 occurrence" in out
    assert "[OK] Questions classified" in out
    assert "[OK] Repeated questions detected" in out
    assert "[OK] Report generated" in out
    assert (output_dir / "final_analysis" / "report.md").exists()
    assert (output_dir / "final_analysis" / "analysis.json").exists()
