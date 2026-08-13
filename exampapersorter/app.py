"""Desktop GUI entry point.

A thin tkinter form over the existing `analyze` workflow -- it performs no
extraction/classification/deduplication/frequency-analysis itself. Every
piece of real work is delegated to the same functions `cli.py`'s `analyze`
command already uses (topic_authority.resolve_*, analyze_pipeline.
extract_questions_for_files/classify_papers_for_files, deduplication.
pipeline.run_deduplication, final_analysis.pipeline.run_final_analysis) so
Stages 1-6 have exactly one implementation, called from two entry points.

Runs the pipeline on a background thread so the Tk event loop never
blocks; progress reaches the UI via a logging.Handler that pushes
formatted records onto a queue.Queue, polled from the main thread with
`root.after(...)` (the only Tk-safe way to move data across threads).
"""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from exampapersorter import api_key_store, paths
from exampapersorter.analyze_pipeline import (
    classify_papers_for_files,
    compute_job_id,
    compute_job_progress,
    extract_questions_for_files,
)
from exampapersorter.config import DEFAULT_CONFIG, Config
from exampapersorter.database import Database
from exampapersorter.deduplication.pipeline import run_deduplication
from exampapersorter.final_analysis.pipeline import run_final_analysis
from exampapersorter.frequency_analysis.pipeline import FrequencyAnalysisPrerequisiteError
from exampapersorter.llm_client import LLMCallFailed
from exampapersorter.llm_providers.openrouter import validate_api_key
from exampapersorter.logging_setup import redact_secret_from_logs, setup_logging
from exampapersorter.output_writer import write_stage3_outputs, write_stage4_outputs, write_stage6_outputs
from exampapersorter.schemas import AnalysisJob
from exampapersorter.topic_authority import TopicAuthorityError, resolve_index_topic_authority, resolve_textbook_topic_authority

logger = logging.getLogger(__name__)

APP_TITLE = "Exam Paper Analyzer"


@dataclass
class PreviousAnalysisInfo:
    job: AnalysisJob
    completed: int
    total: int


def find_previous_analysis(config: Config) -> PreviousAnalysisInfo | None:
    """Detects an unfinished analysis from a prior session -- a quota pause,
    an ordinary crash, or the app simply being closed mid-run (see
    database.py's analysis_jobs table; all three leave status != "completed"
    and are treated identically: resumable). Pure DB/filesystem logic with
    no Tk involved, so it's testable without constructing a GUI (see
    tests/test_app.py) -- the seam between "is there something to resume"
    and "how the window displays that"."""
    if not config.database_path.exists():
        return None
    db = Database(config.database_path)
    try:
        job = db.get_latest_unfinished_analysis_job()
        if job is None:
            return None
        papers_dir = Path(job.question_papers_dir)
        pdf_paths = sorted(papers_dir.glob("*.pdf")) if papers_dir.is_dir() else []
        completed, total = compute_job_progress(pdf_paths, db)
        return PreviousAnalysisInfo(job=job, completed=completed, total=total)
    finally:
        db.close()


class _QueueLogHandler(logging.Handler):
    """Forwards formatted log records to a thread-safe queue for the Tk
    main loop to drain -- the only supported way to get data from the
    worker thread into Tk widgets."""

    def __init__(self, q: "queue.Queue[str]") -> None:
        super().__init__(level=logging.INFO)
        self._queue = q
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._queue.put(self.format(record))
        except Exception:
            pass


def _packaged_config() -> Config:
    """The Config the GUI runs with: same tunables as the CLI, but with
    storage paths redirected to a per-user data directory instead of the
    current working directory (see paths.py)."""
    data_dir = paths.user_data_dir()
    return replace(
        DEFAULT_CONFIG,
        database_path=data_dir / "pipeline.db",
        output_directory=data_dir / "output",
        embedding_cache_dir=data_dir / "embedding_cache",
    )


def _open_path(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


class AnalyzerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title(APP_TITLE)
        root.geometry("560x620")
        root.minsize(480, 560)

        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._worker: threading.Thread | None = None
        self._report_path: Path | None = None
        self._output_dir: Path | None = None

        self.topic_source_var = tk.StringVar(value="textbook")
        self.topic_source_path_var = tk.StringVar(value="")
        self.question_papers_var = tk.StringVar(value="")
        self.api_key_var = tk.StringVar(value="")
        self.save_key_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready.")

        self._build_widgets()
        self._prefill_saved_key()
        self._check_for_previous_analysis()
        self.root.after(100, self._drain_log_queue)

    # ---- UI construction ----

    def _build_widgets(self) -> None:
        pad = {"padx": 12, "pady": 6}

        header = ttk.Label(self.root, text=APP_TITLE, font=("Segoe UI", 14, "bold"))
        header.pack(anchor="w", **pad)

        source_frame = ttk.LabelFrame(self.root, text="Topic source")
        source_frame.pack(fill="x", **pad)
        ttk.Radiobutton(
            source_frame, text="Textbook (has its own table of contents)",
            variable=self.topic_source_var, value="textbook", command=self._clear_source_path,
        ).pack(anchor="w", padx=8, pady=2)
        ttk.Radiobutton(
            source_frame, text="Index / syllabus (.pdf, .txt, .md, .json)",
            variable=self.topic_source_var, value="index", command=self._clear_source_path,
        ).pack(anchor="w", padx=8, pady=2)

        source_row = ttk.Frame(source_frame)
        source_row.pack(fill="x", padx=8, pady=(2, 8))
        ttk.Entry(source_row, textvariable=self.topic_source_path_var, state="readonly").pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(source_row, text="Select File...", command=self._pick_source_file).pack(side="left", padx=(8, 0))

        papers_frame = ttk.LabelFrame(self.root, text="Exam papers folder")
        papers_frame.pack(fill="x", **pad)
        papers_row = ttk.Frame(papers_frame)
        papers_row.pack(fill="x", padx=8, pady=8)
        ttk.Entry(papers_row, textvariable=self.question_papers_var, state="readonly").pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(papers_row, text="Select Folder...", command=self._pick_papers_folder).pack(side="left", padx=(8, 0))

        key_frame = ttk.LabelFrame(self.root, text="OpenRouter API key")
        key_frame.pack(fill="x", **pad)
        ttk.Entry(key_frame, textvariable=self.api_key_var, show="•").pack(fill="x", padx=8, pady=(8, 2))
        ttk.Checkbutton(
            key_frame, text="Save API key securely on this computer", variable=self.save_key_var,
        ).pack(anchor="w", padx=8, pady=(0, 8))

        self.analyze_button = ttk.Button(self.root, text="Analyze", command=self._on_analyze_clicked)
        self.analyze_button.pack(**pad)

        progress_frame = ttk.LabelFrame(self.root, text="Progress")
        progress_frame.pack(fill="both", expand=True, **pad)
        ttk.Label(progress_frame, textvariable=self.status_var).pack(anchor="w", padx=8, pady=(8, 0))
        self.progress_bar = ttk.Progressbar(progress_frame, mode="indeterminate")
        self.progress_bar.pack(fill="x", padx=8, pady=8)

        log_container = ttk.Frame(progress_frame)
        log_container.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log_text = tk.Text(log_container, height=10, state="disabled", wrap="word")
        scrollbar = ttk.Scrollbar(log_container, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        result_row = ttk.Frame(self.root)
        result_row.pack(fill="x", **pad)
        self.open_report_button = ttk.Button(
            result_row, text="Open Report", command=self._open_report, state="disabled"
        )
        self.open_report_button.pack(side="left")
        self.open_folder_button = ttk.Button(
            result_row, text="Open Output Folder", command=self._open_output_folder, state="disabled"
        )
        self.open_folder_button.pack(side="left", padx=(8, 0))

    def _prefill_saved_key(self) -> None:
        saved = api_key_store.load_saved_key()
        if saved:
            self.save_key_var.set(True)
            self.status_var.set("Using a previously saved API key (leave the field blank to reuse it).")

    def _check_for_previous_analysis(self) -> None:
        """"Previous Analysis Found" -- prefills the form from the most
        recent unfinished job and relabels the button, rather than a
        separate screen (see find_previous_analysis)."""
        try:
            info = find_previous_analysis(_packaged_config())
        except Exception:
            logger.exception("Could not check for a previous analysis")
            return
        if info is None:
            return

        job = info.job
        self.topic_source_var.set(job.topic_source_type)
        self.topic_source_path_var.set(job.topic_source_path)
        self.question_papers_var.set(job.question_papers_dir)

        if job.status == "paused" and job.pause_reason:
            reason = f" Paused because: {job.pause_reason}"
        elif job.status == "failed" and job.pause_reason:
            reason = f" It stopped due to an error: {job.pause_reason}"
        else:
            reason = " The app was closed before it finished."
        self.status_var.set(
            f"Previous analysis found: {info.completed} of {info.total} paper(s) already processed.{reason} "
            "Enter your API key and click Resume Analysis to continue, or change the textbook/folder to start "
            "a new analysis."
        )
        self.analyze_button.configure(text="Resume Analysis")

    # ---- File pickers ----

    def _clear_source_path(self) -> None:
        self.topic_source_path_var.set("")

    def _pick_source_file(self) -> None:
        if self.topic_source_var.get() == "textbook":
            filetypes = [("PDF files", "*.pdf")]
        else:
            filetypes = [("Index/syllabus files", "*.pdf *.txt *.md *.markdown *.json"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select file", filetypes=filetypes)
        if path:
            self.topic_source_path_var.set(path)

    def _pick_papers_folder(self) -> None:
        path = filedialog.askdirectory(title="Select exam papers folder")
        if path:
            self.question_papers_var.set(path)

    # ---- Analyze flow ----

    def _on_analyze_clicked(self) -> None:
        source_path = self.topic_source_path_var.get().strip()
        papers_dir = self.question_papers_var.get().strip()
        api_key = self.api_key_var.get().strip()

        if not source_path:
            messagebox.showerror(APP_TITLE, "Select a textbook or index/syllabus file first.")
            return
        if not papers_dir:
            messagebox.showerror(APP_TITLE, "Select a folder of exam-paper PDFs first.")
            return

        explicit_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        saved_key = api_key_store.load_saved_key()
        resolved_key = explicit_key or saved_key
        if not resolved_key:
            messagebox.showerror(
                APP_TITLE,
                "No OpenRouter API key supplied and none is saved. Get a free key at "
                "https://openrouter.ai/keys and paste it in.",
            )
            return

        self.analyze_button.configure(state="disabled")
        self.open_report_button.configure(state="disabled")
        self.open_folder_button.configure(state="disabled")
        self._clear_log()
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start(12)
        self.status_var.set("Starting analysis...")

        is_textbook = self.topic_source_var.get() == "textbook"
        keep_key_saved = self.save_key_var.get()

        self._worker = threading.Thread(
            target=self._run_analysis,
            args=(is_textbook, Path(source_path), Path(papers_dir), resolved_key, keep_key_saved),
            daemon=True,
        )
        self._worker.start()

    def _run_analysis(
        self, is_textbook: bool, source_path: Path, papers_dir: Path, api_key: str, keep_key_saved: bool,
    ) -> None:
        try:
            self._run_pipeline(is_textbook, source_path, papers_dir, api_key, keep_key_saved)
        except TopicAuthorityError as exc:
            self._finish(error=str(exc))
        except FrequencyAnalysisPrerequisiteError as exc:
            self._finish(error=str(exc))
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            logger.exception("Analysis failed")
            self._finish(error=f"Unexpected error: {exc}")

    def _run_pipeline(
        self, is_textbook: bool, source_path: Path, papers_dir: Path, api_key: str, keep_key_saved: bool,
    ) -> None:
        redact_secret_from_logs(api_key)
        config = replace(_packaged_config(), llm_provider="openrouter", openrouter_api_key=api_key)

        ok, message = validate_api_key(config)
        if not ok:
            self._finish(error=f"OpenRouter API key check failed: {message}")
            return
        if keep_key_saved:
            api_key_store.save_key(api_key)
        else:
            api_key_store.clear_saved_key()

        pdf_paths = sorted(papers_dir.glob("*.pdf"))
        if not pdf_paths:
            self._finish(error=f"No PDF files found in {papers_dir}")
            return

        topic_source_type = "textbook" if is_textbook else "index"
        job_id = compute_job_id(topic_source_type, source_path, papers_dir)

        db = Database(config.database_path)
        try:
            db.upsert_analysis_job(job_id, topic_source_type, str(source_path), str(papers_dir), "running")
            try:
                self._post_status("Loading topic authority...")
                if is_textbook:
                    authority = resolve_textbook_topic_authority(source_path, config, db)
                else:
                    authority = resolve_index_topic_authority(source_path, config, db)
                self._post_status(f"Topic authority loaded: {len(authority.topics)} topics")

                self._post_status(f"Found {len(pdf_paths)} exam paper(s)")
                self._post_status("Processing questions...")
                extract_questions_for_files(pdf_paths, config, db)

                if not db.get_all_questions():
                    db.set_analysis_job_status(job_id, "failed")
                    self._finish(error="No questions could be extracted from the supplied exam papers.")
                    return

                self._post_status("Classifying questions...")
                file_summaries, questions_by_paper, totals, _skipped = classify_papers_for_files(
                    pdf_paths, authority.topics, authority.file_hash, config, db
                )
                write_stage3_outputs(
                    config.output_directory, source_path, config.model_identifier, authority.topics,
                    file_summaries, questions_by_paper,
                )

                self._post_status("Finding repeated questions...")
                dedup_result = run_deduplication(config, db)
                run_stats = {
                    "provider": config.llm_provider,
                    "model_identifier": config.model_identifier,
                    "embedding_model": config.embedding_model,
                    "embedding_model_version": config.embedding_model_version,
                    "embedding_similarity_threshold": config.embedding_similarity_threshold,
                    "embedding_cross_type_similarity_threshold": config.embedding_cross_type_similarity_threshold,
                    "dedup_min_merge_confidence": config.dedup_min_merge_confidence,
                    "total_questions": dedup_result.total_questions,
                    "exact_duplicate_groups": dedup_result.exact_duplicate_groups,
                    "candidate_pairs_generated": dedup_result.candidate_pairs_generated,
                    "llm_pairs_reused_from_cache": dedup_result.llm_pairs_reused_from_cache,
                    "llm_pairs_newly_judged": dedup_result.llm_pairs_newly_judged,
                    "llm_batches_failed": dedup_result.llm_batches_failed,
                    "canonical_group_count": dedup_result.canonical_group_count,
                    "status_counts": dedup_result.status_counts,
                }
                write_stage4_outputs(
                    config.output_directory, run_stats, db.list_canonical_questions(),
                    db.list_dedup_pair_decisions(verdict="uncertain"),
                )

                self._post_status("Generating report...")
                report = run_final_analysis(db, authority.file_hash)
                json_path, report_path = write_stage6_outputs(config.output_directory, report)
            except (TopicAuthorityError, LLMCallFailed) as exc:
                # An account-level OpenRouter quota/credit exhaustion (see
                # LLMCallFailed.quota_exhausted / TopicAuthorityError.
                # quota_exhausted) pauses cleanly instead of surfacing as a
                # generic error -- everything already processed is already
                # committed to the database, so there's nothing left to save
                # here beyond the job's own paused status.
                if getattr(exc, "quota_exhausted", False):
                    completed, total = compute_job_progress(pdf_paths, db)
                    db.set_analysis_job_status(job_id, "paused", pause_reason=str(exc))
                    self._finish_paused(str(exc), completed, total)
                    return
                db.set_analysis_job_status(job_id, "failed")
                raise
            except Exception:
                db.set_analysis_job_status(job_id, "failed")
                raise
            db.set_analysis_job_status(job_id, "completed")
        finally:
            db.close()

        self._finish(report=report, report_path=report_path, output_dir=config.output_directory)

    # ---- Thread-safe UI updates ----

    def _post_status(self, message: str) -> None:
        self._log_queue.put(f"[status] {message}")

    def _finish(self, *, report=None, report_path: Path | None = None, output_dir: Path | None = None, error: str | None = None) -> None:
        self._log_queue.put(("__DONE__", report, report_path, output_dir, error))

    def _finish_paused(self, message: str, completed: int, total: int) -> None:
        self._log_queue.put(("__PAUSED__", message, completed, total))

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _append_log(self, line: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _drain_log_queue(self) -> None:
        try:
            while True:
                item = self._log_queue.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "__DONE__":
                    _, report, report_path, output_dir, error = item
                    self._on_finished(report, report_path, output_dir, error)
                    continue
                if isinstance(item, tuple) and item and item[0] == "__PAUSED__":
                    _, message, completed, total = item
                    self._on_paused(message, completed, total)
                    continue
                line = str(item)
                if line.startswith("[status] "):
                    self.status_var.set(line[len("[status] "):])
                self._append_log(line)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log_queue)

    def _on_paused(self, message: str, completed: int, total: int) -> None:
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate", value=0)
        self.analyze_button.configure(state="normal", text="Resume Analysis")
        self.status_var.set(
            f"Analysis paused: {message} Completed {completed} of {total} paper(s) -- your progress has been "
            "saved locally. You can close this app now and resume later."
        )
        self._append_log("")
        self._append_log(f"PAUSED: {message}")
        self._append_log(f"Completed {completed} of {total} paper(s). Progress saved locally.")
        messagebox.showinfo(
            APP_TITLE,
            f"Analysis paused.\n\n{message}\n\nCompleted {completed} of {total} paper(s). Your work has been "
            "saved locally -- close the app any time and click 'Resume Analysis' to continue.",
        )

    def _on_finished(self, report, report_path: Path | None, output_dir: Path | None, error: str | None) -> None:
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate", value=100 if not error else 0)
        self.analyze_button.configure(state="normal", text="Analyze")

        if error:
            self.status_var.set("Analysis failed.")
            self._append_log(f"ERROR: {error}")
            messagebox.showerror(APP_TITLE, error)
            return

        self._report_path = report_path
        self._output_dir = output_dir
        self.open_report_button.configure(state="normal")
        self.open_folder_button.configure(state="normal")

        s = report.summary
        self.status_var.set("Analysis complete.")
        self._append_log("")
        self._append_log("Analysis complete.")
        self._append_log(f"{s.total_question_occurrences} questions analyzed")
        self._append_log(f"{s.total_canonical_questions} unique questions")
        self._append_log(f"{s.repeated_canonical_question_count} repeated question groups")

    def _open_report(self) -> None:
        if self._report_path:
            _open_path(self._report_path)

    def _open_output_folder(self) -> None:
        if self._output_dir:
            _open_path(self._output_dir)


def _install_file_logging() -> Path:
    """The packaged app has no console (windowed build), so a crash or an
    error before the GUI's own log pane is visible would otherwise leave
    no trace at all. Mirrors setup_logging()'s format into a per-user log
    file in addition to stderr."""
    log_dir = paths.user_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)
    return log_path


def main() -> None:
    setup_logging(DEFAULT_CONFIG.log_level)
    log_path = _install_file_logging()
    logger.info("Exam Paper Analyzer starting (log file: %s)", log_path)

    root = tk.Tk()
    app = AnalyzerApp(root)

    handler = _QueueLogHandler(app._log_queue)
    logging.getLogger().addHandler(handler)

    def _report_tk_callback_exception(exc, val, tb) -> None:
        logger.error("Unhandled error in GUI callback", exc_info=(exc, val, tb))
        messagebox.showerror(APP_TITLE, f"An unexpected error occurred:\n\n{val}\n\nSee {log_path} for details.")

    root.report_callback_exception = _report_tk_callback_exception

    root.mainloop()


if __name__ == "__main__":
    main()
