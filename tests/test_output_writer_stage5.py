import json

from exampapersorter.frequency_analysis.aggregate import build_frequency_analysis
from exampapersorter.output_writer import write_stage5_outputs
from exampapersorter.schemas import CanonicalQuestion, MetadataFieldValue, Paper, PaperMetadata, Question, Topic


def t(id, level=1, parent_id=None, name=None):
    return Topic(id=id, name=name or id, level=level, parent_id=parent_id, source_pages=[10])


def q(question_id, topic_id=None, status="unclassified", qtype="short_answer", paper_year=None):
    return Question(
        question_id=question_id, paper_id="p1", section_id=None,
        source_filename="paper.pdf", source_file_hash="hash1", paper_year=paper_year,
        question_number="1", question_text="Explain X.", question_type=qtype,
        source_pages=[1], extraction_confidence=0.9,
        topic_id=topic_id, classification_status=status,
    )


def paper():
    field = lambda **kw: MetadataFieldValue(**kw)
    return Paper(
        paper_id="p1", file_hash="h1", start_page=1, end_page=5,
        metadata=PaperMetadata(exam_name=field(), institution=field(), subject=field(), date=field(), year=field(), paper_identifier=field()),
        boundary_confidence=0.9,
    )


def _sample_summary():
    topics = [t("chapter_a"), t("chapter_b")]
    questions = [
        q("q1", topic_id="chapter_a", status="classified", qtype="mcq", paper_year="2022"),
        q("q2", topic_id="chapter_a", status="classified", qtype="mcq", paper_year="2022"),
        q("q3", status="no_match"),
    ]
    canonicals = [
        CanonicalQuestion(canonical_question_id="c1", canonical_question_text="Explain X.", question_type="mcq", representative_question_id="q1", dedup_status="exact_duplicate", dedup_confidence=1.0),
        CanonicalQuestion(canonical_question_id="c2", canonical_question_text="Explain X.", question_type="mcq", representative_question_id="q3", dedup_status="singleton"),
    ]
    return build_frequency_analysis(topics, questions, canonicals, [paper()], "hash1")


def test_write_stage5_outputs_creates_summary_json_and_report_md(tmp_path):
    summary = _sample_summary()
    summary_path, report_path = write_stage5_outputs(tmp_path, summary)

    assert summary_path == tmp_path / "frequency_analysis" / "summary.json"
    assert report_path == tmp_path / "frequency_analysis" / "report.md"
    assert summary_path.exists()
    assert report_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded["total_question_occurrences"] == 3
    assert loaded["total_canonical_questions"] == 2

    report_text = report_path.read_text(encoding="utf-8")
    assert "# Exam Question Analysis" in report_text
    assert "chapter_b" not in report_text or "chapter_b (0 occurrences" in report_text or "No Confident Topic Match" in report_text
    assert "No Confident Topic Match (1)" in report_text
    assert "q3" in report_text  # no_match question stays visible, traceable


def test_stage5_output_files_are_reproducible_across_runs(tmp_path):
    summary = _sample_summary()
    write_stage5_outputs(tmp_path, summary)
    first_json = (tmp_path / "frequency_analysis" / "summary.json").read_text(encoding="utf-8")
    first_md = (tmp_path / "frequency_analysis" / "report.md").read_text(encoding="utf-8")

    write_stage5_outputs(tmp_path, summary)
    second_json = (tmp_path / "frequency_analysis" / "summary.json").read_text(encoding="utf-8")
    second_md = (tmp_path / "frequency_analysis" / "report.md").read_text(encoding="utf-8")

    assert first_json == second_json
    assert first_md == second_md


def test_zero_question_topic_appears_in_chapters_section(tmp_path):
    summary = _sample_summary()
    _, report_path = write_stage5_outputs(tmp_path, summary)
    text = report_path.read_text(encoding="utf-8")
    assert "chapter_b" in text
    assert "(no questions tested)" in text
