"""analysis_jobs is the minimal, additive piece of state that lets the GUI
(app.py) discover an unfinished analysis after being closed and reopened --
see schemas.AnalysisJob's docstring. These tests exercise the Database
methods directly, independent of the pipeline that calls them."""
from exampapersorter.database import Database


def test_upsert_creates_a_running_job(tmp_path):
    db = Database(tmp_path / "test.db")
    db.upsert_analysis_job("job1", "textbook", "book.pdf", "papers/", "running")

    job = db.get_analysis_job("job1")
    assert job.job_id == "job1"
    assert job.topic_source_type == "textbook"
    assert job.topic_source_path == "book.pdf"
    assert job.question_papers_dir == "papers/"
    assert job.status == "running"
    assert job.pause_reason is None
    db.close()


def test_get_analysis_job_returns_none_when_absent(tmp_path):
    db = Database(tmp_path / "test.db")
    assert db.get_analysis_job("does-not-exist") is None
    db.close()


def test_set_analysis_job_status_records_pause_reason(tmp_path):
    db = Database(tmp_path / "test.db")
    db.upsert_analysis_job("job1", "textbook", "book.pdf", "papers/", "running")

    db.set_analysis_job_status("job1", "paused", pause_reason="OpenRouter daily quota exhausted")

    job = db.get_analysis_job("job1")
    assert job.status == "paused"
    assert job.pause_reason == "OpenRouter daily quota exhausted"
    db.close()


def test_upsert_on_an_existing_job_id_clears_a_prior_pause_reason(tmp_path):
    """Re-running (a genuine resume) always starts a fresh "running" status
    -- whatever paused/failed it before no longer applies once it's running
    again."""
    db = Database(tmp_path / "test.db")
    db.upsert_analysis_job("job1", "textbook", "book.pdf", "papers/", "running")
    db.set_analysis_job_status("job1", "paused", pause_reason="quota exhausted")

    db.upsert_analysis_job("job1", "textbook", "book.pdf", "papers/", "running")

    job = db.get_analysis_job("job1")
    assert job.status == "running"
    assert job.pause_reason is None
    db.close()


def test_upsert_preserves_created_at_across_a_resume(tmp_path):
    db = Database(tmp_path / "test.db")
    db.upsert_analysis_job("job1", "textbook", "book.pdf", "papers/", "running")
    first_created_at = db.get_analysis_job("job1").created_at

    db.set_analysis_job_status("job1", "paused", pause_reason="quota exhausted")
    db.upsert_analysis_job("job1", "textbook", "book.pdf", "papers/", "running")

    assert db.get_analysis_job("job1").created_at == first_created_at
    db.close()


def test_get_latest_unfinished_analysis_job_ignores_completed_jobs(tmp_path):
    db = Database(tmp_path / "test.db")
    db.upsert_analysis_job("job1", "textbook", "book.pdf", "papers/", "running")
    db.set_analysis_job_status("job1", "completed")

    assert db.get_latest_unfinished_analysis_job() is None
    db.close()


def test_get_latest_unfinished_analysis_job_finds_a_paused_job(tmp_path):
    db = Database(tmp_path / "test.db")
    db.upsert_analysis_job("job1", "textbook", "book.pdf", "papers/", "running")
    db.set_analysis_job_status("job1", "paused", pause_reason="quota exhausted")

    job = db.get_latest_unfinished_analysis_job()
    assert job is not None
    assert job.job_id == "job1"
    assert job.status == "paused"
    db.close()


def test_get_latest_unfinished_analysis_job_finds_a_job_still_running(tmp_path):
    """A crash never gets to call set_analysis_job_status at all -- the row
    is left exactly as upsert_analysis_job wrote it, status="running". That
    must still be discovered as "something to resume"."""
    db = Database(tmp_path / "test.db")
    db.upsert_analysis_job("job1", "textbook", "book.pdf", "papers/", "running")

    job = db.get_latest_unfinished_analysis_job()
    assert job is not None
    assert job.status == "running"
    db.close()


def test_get_latest_unfinished_analysis_job_returns_the_most_recently_touched_one(tmp_path):
    db = Database(tmp_path / "test.db")
    db.upsert_analysis_job("job1", "textbook", "book1.pdf", "papers1/", "running")
    db.set_analysis_job_status("job1", "paused", pause_reason="quota exhausted")
    db.upsert_analysis_job("job2", "index", "syllabus.md", "papers2/", "running")

    job = db.get_latest_unfinished_analysis_job()
    assert job.job_id == "job2"
    db.close()


def test_a_second_completed_job_does_not_hide_an_earlier_unfinished_one(tmp_path):
    """Two independent analyses (different textbook/folder): one paused,
    one later started and completed. The paused one must still surface as
    the resumable job -- starting and finishing something else must not
    silently lose track of it."""
    db = Database(tmp_path / "test.db")
    db.upsert_analysis_job("job1", "textbook", "book1.pdf", "papers1/", "running")
    db.set_analysis_job_status("job1", "paused", pause_reason="quota exhausted")
    db.upsert_analysis_job("job2", "index", "syllabus.md", "papers2/", "running")
    db.set_analysis_job_status("job2", "completed")

    job = db.get_latest_unfinished_analysis_job()
    assert job.job_id == "job1"
    db.close()
