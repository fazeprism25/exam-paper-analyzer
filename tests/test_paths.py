from pathlib import Path

from exampapersorter import paths


def test_user_data_dir_honors_explicit_override(monkeypatch, tmp_path):
    monkeypatch.setenv("EXAMPAPERANALYZER_DATA_DIR", str(tmp_path / "custom"))
    assert paths.user_data_dir() == tmp_path / "custom"


def test_user_data_dir_is_platform_specific_and_absolute(monkeypatch):
    monkeypatch.delenv("EXAMPAPERANALYZER_DATA_DIR", raising=False)
    result = paths.user_data_dir()
    assert result.is_absolute()
    assert result.name == "ExamPaperAnalyzer"


def test_is_frozen_false_under_pytest():
    assert paths.is_frozen() is False
