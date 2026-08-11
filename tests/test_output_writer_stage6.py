import json

from exampapersorter.final_analysis.build import build_final_report
from exampapersorter.frequency_analysis.aggregate import build_frequency_analysis
from exampapersorter.output_writer import write_stage6_outputs
from exampapersorter.schemas import CanonicalQuestion, FinalAnalysisReport, MetadataFieldValue, Paper, PaperMetadata, Question, Topic


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


def _sample_report():
    topics = [t("chapter_a"), t("chapter_b")]
    questions = [
        q("q1", topic_id="chapter_a", status="classified", qtype="mcq", paper_year="2022"),
        q("q2", topic_id="chapter_a", status="classified", qtype="mcq", paper_year="2022"),
        q("q3", status="no_match"),
    ]
    canonicals = [
        CanonicalQuestion(
            canonical_question_id="c1", canonical_question_text="Explain X.", question_type="mcq",
            topic_id="chapter_a", topic_name="chapter_a", representative_question_id="q1",
            dedup_status="exact_duplicate", dedup_confidence=1.0,
            source_question_ids=["q1", "q2"], occurrence_count=2, unique_paper_count=1, years=["2022"],
        ),
        CanonicalQuestion(
            canonical_question_id="c2", canonical_question_text="Explain X.", question_type="mcq",
            representative_question_id="q3", dedup_status="singleton",
            source_question_ids=["q3"], occurrence_count=1, unique_paper_count=1, years=[],
        ),
    ]
    summary = build_frequency_analysis(topics, questions, canonicals, [paper()], "hash1")
    return build_final_report(summary, canonicals, questions)


def test_write_stage6_outputs_creates_analysis_json_and_report_md(tmp_path):
    report = _sample_report()
    json_path, report_path = write_stage6_outputs(tmp_path, report)

    assert json_path == tmp_path / "final_analysis" / "analysis.json"
    assert report_path == tmp_path / "final_analysis" / "report.md"
    assert json_path.exists()
    assert report_path.exists()

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["total_question_occurrences"] == 3
    assert loaded["summary"]["total_canonical_questions"] == 2
    # JSON validates against the schema it was dumped from
    FinalAnalysisReport.model_validate(loaded)

    text = report_path.read_text(encoding="utf-8")
    assert "# Exam Paper Analysis" in text
    assert "## Executive Summary" in text
    assert "## Most Repeated Questions" in text
    assert "## Most Tested Chapters" in text
    assert "## Question Type Distribution" in text
    assert "## Analysis by Chapter" in text
    assert "## Untested Topics" in text
    assert "## Questions With No Confident Topic Match" in text
    assert "## Ambiguous Classifications" in text
    assert "## Data Quality / Notes" in text
    assert "q3" not in text  # question_id itself isn't shown; full text is
    assert "Explain X." in text  # no_match question's text stays visible


def test_stage6_output_files_are_reproducible_across_runs(tmp_path):
    report = _sample_report()
    write_stage6_outputs(tmp_path, report)
    first_json = (tmp_path / "final_analysis" / "analysis.json").read_text(encoding="utf-8")
    first_md = (tmp_path / "final_analysis" / "report.md").read_text(encoding="utf-8")

    write_stage6_outputs(tmp_path, report)
    second_json = (tmp_path / "final_analysis" / "analysis.json").read_text(encoding="utf-8")
    second_md = (tmp_path / "final_analysis" / "report.md").read_text(encoding="utf-8")

    assert first_json == second_json
    assert first_md == second_md


def test_zero_question_topic_appears_in_untested_section(tmp_path):
    report = _sample_report()
    _, report_path = write_stage6_outputs(tmp_path, report)
    text = report_path.read_text(encoding="utf-8")
    assert "chapter_b" in text
    assert "## Untested Topics" in text


def test_repeated_question_shown_with_dedup_status_and_confidence(tmp_path):
    report = _sample_report()
    _, report_path = write_stage6_outputs(tmp_path, report)
    text = report_path.read_text(encoding="utf-8")
    assert "exact_duplicate" in text
    assert "1.00" in text


def test_markdown_heading_hierarchy_is_well_formed(tmp_path):
    report = _sample_report()
    _, report_path = write_stage6_outputs(tmp_path, report)
    lines = report_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# Exam Paper Analysis"
    top_level_sections = [ln for ln in lines if ln.startswith("## ")]
    expected = [
        "## Executive Summary", "## Most Repeated Questions", "## Most Tested Chapters",
        "## Question Type Distribution", "## Analysis by Chapter", "## Untested Topics",
    ]
    for e in expected:
        assert e in top_level_sections
