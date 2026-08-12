"""Tests the pause/resume detection seam in app.py (find_previous_analysis)
without constructing any Tk widgets -- this project has no existing GUI
test infrastructure, so full window automation would be both fragile and a
new dependency. find_previous_analysis is pure DB/filesystem logic; the
class method that calls it (AnalyzerApp._check_for_previous_analysis) is
only a thin adapter onto Tk widgets and is not covered here."""
from dataclasses import replace
from pathlib import Path

from exampapersorter.app import find_previous_analysis
from exampapersorter.config import DEFAULT_CONFIG
from exampapersorter.database import Database
from exampapersorter.pdf_utils import compute_file_hash


def make_pdf(path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(path))
    doc.close()


def test_no_database_means_no_previous_analysis(tmp_path):
    config = replace(DEFAULT_CONFIG, database_path=tmp_path / "does_not_exist.db")
    assert find_previous_analysis(config) is None


def test_no_unfinished_job_means_no_previous_analysis(tmp_path):
    db_path = tmp_path / "t.db"
    Database(db_path).close()  # creates the file with an empty analysis_jobs table
    config = replace(DEFAULT_CONFIG, database_path=db_path)
    assert find_previous_analysis(config) is None


def test_completed_job_is_not_reported_as_a_previous_analysis(tmp_path):
    db_path = tmp_path / "t.db"
    db = Database(db_path)
    db.upsert_analysis_job("job1", "textbook", "book.pdf", str(tmp_path / "papers"), "running")
    db.set_analysis_job_status("job1", "completed")
    db.close()

    config = replace(DEFAULT_CONFIG, database_path=db_path)
    assert find_previous_analysis(config) is None


def test_paused_job_is_reported_with_progress_and_reason(tmp_path):
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    pdf1, pdf2 = papers_dir / "p1.pdf", papers_dir / "p2.pdf"
    make_pdf(pdf1)
    make_pdf(pdf2)

    db_path = tmp_path / "t.db"
    db = Database(db_path)
    db.upsert_analysis_job("job1", "textbook", str(tmp_path / "book.pdf"), str(papers_dir), "running")
    db.register_question_paper_file(compute_file_hash(pdf1), str(pdf1), 1)
    db.update_question_paper_file_status(compute_file_hash(pdf1), "success", "model", None)
    db.set_analysis_job_status("job1", "paused", pause_reason="OpenRouter daily quota exhausted")
    db.close()

    config = replace(DEFAULT_CONFIG, database_path=db_path)
    info = find_previous_analysis(config)

    assert info is not None
    assert info.job.status == "paused"
    assert info.job.pause_reason == "OpenRouter daily quota exhausted"
    assert info.completed == 1
    assert info.total == 2


def test_previous_analysis_survives_a_deleted_papers_folder(tmp_path):
    """The folder may have been moved/deleted between sessions -- progress
    just reports 0/0 rather than crashing the startup check."""
    db_path = tmp_path / "t.db"
    db = Database(db_path)
    db.upsert_analysis_job("job1", "textbook", "book.pdf", str(tmp_path / "gone"), "running")
    db.close()

    config = replace(DEFAULT_CONFIG, database_path=db_path)
    info = find_previous_analysis(config)

    assert info is not None
    assert info.completed == 0
    assert info.total == 0


def test_running_job_left_by_a_crash_is_still_reported(tmp_path):
    """A crash never gets to mark the job "failed" or "paused" -- it's left
    at "running", and that must still be treated as resumable."""
    db_path = tmp_path / "t.db"
    db = Database(db_path)
    db.upsert_analysis_job("job1", "index", "syllabus.md", str(tmp_path / "papers"), "running")
    db.close()

    config = replace(DEFAULT_CONFIG, database_path=db_path)
    info = find_previous_analysis(config)

    assert info is not None
    assert info.job.status == "running"
