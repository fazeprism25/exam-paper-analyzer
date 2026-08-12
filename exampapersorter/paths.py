"""Per-user data directory for the packaged desktop app.

`Config`'s own defaults (database_path="data/pipeline.db",
output_directory="output", embedding_cache_dir="data/embedding_cache") are
resolved relative to the current working directory -- correct for running
from source (`python cli.py ...`) or from the test suite, but wrong once
the app is installed: on Windows the install directory is typically
`Program Files` (not writable by a normal user) and on macOS it's inside
the read-only `.app` bundle. Nothing in `cli.py` or `config.py` changes
here -- this module is only consulted by `exampapersorter/app.py` (the GUI
entry point), so source/CLI/test behavior is untouched.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_APP_DIR_NAME = "ExamPaperAnalyzer"


def is_frozen() -> bool:
    """True when running from a PyInstaller-built executable."""
    return bool(getattr(sys, "frozen", False))


def user_data_dir() -> Path:
    """Per-user, per-platform writable directory for this app's database,
    output reports, and embedding-model cache.

    Windows:  %LOCALAPPDATA%\\ExamPaperAnalyzer
    macOS:    ~/Library/Application Support/ExamPaperAnalyzer
    Linux:    $XDG_DATA_HOME/ExamPaperAnalyzer or ~/.local/share/ExamPaperAnalyzer

    An `EXAMPAPERANALYZER_DATA_DIR` environment variable overrides this
    unconditionally (used by packaging smoke tests and by anyone who wants
    the GUI to use a non-default location).
    """
    override = os.environ.get("EXAMPAPERANALYZER_DATA_DIR")
    if override:
        return Path(override)

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / _APP_DIR_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / _APP_DIR_NAME
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / _APP_DIR_NAME
