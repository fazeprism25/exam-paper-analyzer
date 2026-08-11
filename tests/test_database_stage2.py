from exampapersorter.database import Database
from exampapersorter.schemas import (
    EvidenceRef,
    MetadataFieldExtraction,
    MetadataFieldValue,
    Paper,
    PaperBoundaryCandidate,
    PaperBoundaryDetectionResult,
    PaperMetadata,
    PaperStructureExtraction,
    Question,
    QuestionExtractionResult,
    Section,
)


def empty_metadata():
    f = MetadataFieldValue()
    return PaperMetadata(exam_name=f, institution=f, subject=f, date=f, year=f, paper_identifier=f)


def test_question_paper_file_status_lifecycle(tmp_path):
    db = Database(tmp_path / "test.db")
    db.register_question_paper_file("hash1", "some.pdf", 15)
    assert db.get_question_paper_file_status("hash1") == "pending"

    db.update_question_paper_file_status("hash1", "success", "qwen3:8b", None)
    assert db.get_question_paper_file_status("hash1") == "success"
    db.close()


def test_register_is_idempotent_and_preserves_status(tmp_path):
    db = Database(tmp_path / "test.db")
    db.register_question_paper_file("hash1", "some.pdf", 15)
    db.update_question_paper_file_status("hash1", "success", "qwen3:8b", None)
    # Re-registering (e.g. a repeat CLI invocation) must not reset progress.
    db.register_question_paper_file("hash1", "some.pdf", 15)
    assert db.get_question_paper_file_status("hash1") == "success"
    db.close()


def test_unknown_file_hash_has_no_status(tmp_path):
    db = Database(tmp_path / "test.db")
    assert db.get_question_paper_file_status("nonexistent") is None
    db.close()


def test_paper_boundary_verdict_cache_round_trips(tmp_path):
    db = Database(tmp_path / "test.db")
    result = PaperBoundaryDetectionResult(
        papers=[PaperBoundaryCandidate(start_page=1, end_page=5, confidence=0.9, evidence=[EvidenceRef(page=1, text="hdr")])]
    )
    assert db.get_paper_boundary_verdict("hash1", "qwen3:8b") is None
    db.save_paper_boundary_verdict("hash1", "qwen3:8b", result)
    loaded = db.get_paper_boundary_verdict("hash1", "qwen3:8b")
    assert loaded == result
    db.close()


def test_paper_structure_verdict_cache_round_trips(tmp_path):
    db = Database(tmp_path / "test.db")
    result = PaperStructureExtraction(
        institution=MetadataFieldExtraction(value="Some College", evidence="Some College letterhead"), sections=[]
    )
    db.save_paper_structure_verdict("hash1", 1, 5, "qwen3:8b", result)
    assert db.get_paper_structure_verdict("hash1", 1, 5, "qwen3:8b") == result
    assert db.get_paper_structure_verdict("hash1", 1, 6, "qwen3:8b") is None  # different range -- no false hit
    db.close()


def test_question_extraction_verdict_cache_round_trips(tmp_path):
    db = Database(tmp_path / "test.db")
    result = QuestionExtractionResult(questions=[])
    db.save_question_extraction_verdict("hash1", 1, 5, "qwen3:8b", result)
    assert db.get_question_extraction_verdict("hash1", 1, 5, "qwen3:8b") == result
    db.close()


def test_paper_round_trips_and_resume_check(tmp_path):
    db = Database(tmp_path / "test.db")
    paper = Paper(
        paper_id="p1", file_hash="hash1", start_page=1, end_page=5, metadata=empty_metadata(),
        sections=[], boundary_confidence=0.9, boundary_evidence=[], status="success", notes=[],
    )
    assert db.get_paper("p1") is None
    db.save_paper(paper)
    loaded = db.get_paper("p1")
    assert loaded == paper
    assert loaded.status == "success"
    db.close()


def test_get_papers_for_file_orders_by_start_page(tmp_path):
    db = Database(tmp_path / "test.db")
    for idx, start in [(2, 6), (1, 1)]:
        db.save_paper(
            Paper(
                paper_id=f"p{idx}", file_hash="hash1", start_page=start, end_page=start + 4,
                metadata=empty_metadata(), sections=[], boundary_confidence=0.9, boundary_evidence=[],
                status="success", notes=[],
            )
        )
    papers = db.get_papers_for_file("hash1")
    assert [p.paper_id for p in papers] == ["p1", "p2"]
    db.close()


def test_questions_round_trip_including_flags(tmp_path):
    db = Database(tmp_path / "test.db")
    questions = [
        Question(
            question_id="p1_q1", paper_id="p1", section_id="p1_s1",
            source_filename="some.pdf", source_file_hash="hash1",
            exam_name="Sessional Exam", institution="Some College", subject="Biochemistry",
            paper_year="2025", paper_date="04-07-2025",
            question_number="1", question_text="Explain glycolysis.", original_text="Explain glyco1ysis.",
            correction_applied=True, question_type="short_answer", source_pages=[2],
            extraction_confidence=0.85, ambiguous=False,
        ),
        Question(
            question_id="p1_q2", paper_id="p1", section_id=None,
            source_filename="some.pdf", source_file_hash="hash1",
            question_number=None, question_text="Which enzyme?", question_type="mcq",
            options=["Hexokinase", "Pepsin", "Trypsin"], source_pages=[3],
            extraction_confidence=0.1, ambiguous=True, ambiguity_note="cut off",
        ),
    ]
    db.save_questions(questions)
    loaded = db.get_questions_for_paper("p1")
    assert len(loaded) == 2
    assert loaded[0].correction_applied is True
    assert loaded[0].original_text == "Explain glyco1ysis."
    assert loaded[0].exam_name == "Sessional Exam"
    assert loaded[0].paper_year == "2025"
    assert loaded[1].ambiguous is True
    assert loaded[1].question_number is None
    assert loaded[1].options == ["Hexokinase", "Pepsin", "Trypsin"]
    db.close()


def test_get_questions_for_paper_empty_is_not_an_error(tmp_path):
    db = Database(tmp_path / "test.db")
    assert db.get_questions_for_paper("no_such_paper") == []
    db.close()
