"""SQLite persistence for Stage 1.

Why a database for what's conceptually a one-shot script? Two reasons
directly tied to the resumability requirement:

1. Docling conversion of a page range is the expensive step (tens of
   seconds per range on this machine). If the process crashes or the
   search has to continue past the first window, we don't want to
   re-run Docling on ranges we've already extracted -- `page_range_evidence`
   caches that.
2. `toc_search_log` records every range we've already asked the LLM about
   and its verdict, so re-running the search after a crash resumes from
   where it left off instead of re-asking (and re-billing local compute
   for) ranges already classified as "neither".

Both tables are keyed by the textbook's content hash, not its filename or
path -- so renaming the file, or moving it, doesn't invalidate the cache,
and two different files that happen to share a name never collide.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from exampapersorter.schemas import (
    AnalysisJob,
    CanonicalQuestion,
    DedupPairDecision,
    PageRangeEvidence,
    Paper,
    PaperBoundaryDetectionResult,
    PaperStructureExtraction,
    Question,
    QuestionEmbedding,
    QuestionExtractionResult,
    QuestionTopicClassification,
    TOCDetectionResult,
    Topic,
    TopicClassificationResult,
    TopicExtractionResult,
    ValidationReport,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS textbook_files (
    file_hash TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    total_pages INTEGER,
    first_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS page_range_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT NOT NULL,
    start_page INTEGER NOT NULL,
    end_page INTEGER NOT NULL,
    conversion_status TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(file_hash, start_page, end_page)
);

CREATE TABLE IF NOT EXISTS toc_search_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT NOT NULL,
    start_page INTEGER NOT NULL,
    end_page INTEGER NOT NULL,
    model_used TEXT NOT NULL,
    document_structure TEXT NOT NULL,
    confidence REAL NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(file_hash, start_page, end_page, model_used)
);

CREATE TABLE IF NOT EXISTS topic_extraction_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT NOT NULL,
    model_used TEXT NOT NULL,
    source_start_page INTEGER NOT NULL,
    source_end_page INTEGER NOT NULL,
    status TEXT NOT NULL,
    topics_json TEXT,
    validation_json TEXT,
    created_at TEXT NOT NULL
);

-- Columns added by _migrate() below (ALTER TABLE, not here) so opening an
-- existing pre-migration database doesn't lose its cached data:
--   toc_search_log.call_metadata_json, topic_extraction_runs.call_metadata_json,
--   paper_boundary_log.call_metadata_json, paper_structure_log.call_metadata_json,
--   question_extraction_log.call_metadata_json
-- Each holds llm_client.CallAttemptMetadata (requested/actual model,
-- fallback_level, attempt_count, provider) for the call that produced that
-- row, JSON-encoded; NULL for rows written before this column existed.

-- --- Stage 2: question-paper extraction ---

CREATE TABLE IF NOT EXISTS question_paper_files (
    file_hash TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    total_pages INTEGER,
    status TEXT NOT NULL,           -- pending | success | partial | failed
    model_used TEXT,
    error_message TEXT,
    first_seen_at TEXT NOT NULL,
    last_processed_at TEXT
);

CREATE TABLE IF NOT EXISTS paper_boundary_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT NOT NULL,
    model_used TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(file_hash, model_used)
);

CREATE TABLE IF NOT EXISTS paper_structure_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT NOT NULL,
    start_page INTEGER NOT NULL,
    end_page INTEGER NOT NULL,
    model_used TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(file_hash, start_page, end_page, model_used)
);

CREATE TABLE IF NOT EXISTS question_extraction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT NOT NULL,
    start_page INTEGER NOT NULL,
    end_page INTEGER NOT NULL,
    model_used TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(file_hash, start_page, end_page, model_used)
);

CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL,
    start_page INTEGER NOT NULL,
    end_page INTEGER NOT NULL,
    status TEXT NOT NULL,           -- success | partial | failed
    paper_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS questions (
    question_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    section_id TEXT,
    source_filename TEXT NOT NULL,
    source_file_hash TEXT NOT NULL,
    exam_name TEXT,
    institution TEXT,
    subject TEXT,
    paper_year TEXT,
    paper_date TEXT,
    question_number TEXT,
    question_text TEXT NOT NULL,
    options_json TEXT NOT NULL,
    original_text TEXT,
    correction_applied INTEGER NOT NULL,
    question_type TEXT NOT NULL,
    source_pages_json TEXT NOT NULL,
    extraction_confidence REAL,
    ambiguous INTEGER NOT NULL,
    ambiguity_note TEXT,
    created_at TEXT NOT NULL
);

-- Columns added by _migrate() below (ALTER TABLE, not here) so an existing
-- pre-Stage-3 database doesn't lose its already-extracted questions:
--   questions.topic_id, questions.topic_name, questions.topic_confidence,
--   questions.classification_status (see _QUESTION_CLASSIFICATION_COLUMNS)

-- --- Stage 3: topic classification ---
-- Brand new table (unlike the ALTER-TABLE columns above) so it can just
-- declare call_metadata_json directly -- no pre-Stage-3 database has this
-- table to retrofit.
CREATE TABLE IF NOT EXISTS topic_classification_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    textbook_file_hash TEXT NOT NULL,
    model_used TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    call_metadata_json TEXT,
    UNIQUE(paper_id, model_used)
);

-- --- Stage 4: semantic deduplication ---
-- All three tables are brand new (no ALTER TABLE needed, no existing table
-- touched) so opening a pre-Stage-4 database is purely additive -- every
-- Stage 1/2/3 row is untouched. `questions` itself gains no new column;
-- question_canonical_links is the sole place a question occurrence is
-- associated with a canonical group (see schemas.CanonicalQuestion's
-- docstring for why occurrence-level aggregates are never stored directly
-- on canonical_questions).

CREATE TABLE IF NOT EXISTS question_embeddings (
    question_id TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (question_id, embedding_model, embedding_version)
);

-- Doubles as both the LLM-call cache (a resumed run never re-asks about a
-- pair it already has a verdict for) and the audit trail (decision_source
-- distinguishes a deterministic exact-duplicate match from an LLM
-- judgment). question_id_a/question_id_b are always stored with a < b
-- (lexicographic) by the caller, so the same unordered pair always hits the
-- same cache row regardless of which order it was generated in.
-- model_used is a fixed sentinel ("deterministic") for decision_source=
-- exact_duplicate rather than NULL, so the UNIQUE constraint below behaves
-- the same way for both decision sources (SQLite treats NULLs as mutually
-- distinct, which would silently defeat a NULL-valued uniqueness check).
CREATE TABLE IF NOT EXISTS dedup_pair_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id_a TEXT NOT NULL,
    question_id_b TEXT NOT NULL,
    decision_source TEXT NOT NULL,
    model_used TEXT NOT NULL,
    verdict TEXT NOT NULL,
    confidence REAL NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL,
    call_metadata_json TEXT,
    UNIQUE(question_id_a, question_id_b, model_used)
);

CREATE TABLE IF NOT EXISTS canonical_questions (
    canonical_question_id TEXT PRIMARY KEY,
    canonical_question_text TEXT NOT NULL,
    question_type TEXT NOT NULL,
    topic_id TEXT,
    topic_name TEXT,
    representative_question_id TEXT NOT NULL,
    dedup_confidence REAL,
    dedup_status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Many-to-one: multiple question_id rows may point at the same
-- canonical_question_id. A question_id is the PRIMARY KEY here (not part of
-- a composite), enforcing that one occurrence belongs to at most one
-- canonical group at a time.
CREATE TABLE IF NOT EXISTS question_canonical_links (
    question_id TEXT PRIMARY KEY,
    canonical_question_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- --- GUI pause/resume job tracking (see schemas.AnalysisJob) ---
-- Deliberately holds only run identity + lifecycle state, never Stage 1-6
-- data -- progress ("12 of 20 papers done") is always recomputed live from
-- question_paper_files/papers so this table can never drift stale against
-- them. Brand new table, no ALTER TABLE needed.
CREATE TABLE IF NOT EXISTS analysis_jobs (
    job_id TEXT PRIMARY KEY,
    topic_source_type TEXT NOT NULL,
    topic_source_path TEXT NOT NULL,
    question_papers_dir TEXT NOT NULL,
    status TEXT NOT NULL,           -- running | paused | failed | completed
    pause_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


# Additive migrations for databases created before call-metadata tracking
# existed. CREATE TABLE IF NOT EXISTS in SCHEMA above only affects brand-new
# tables, so an existing on-disk DB needs these columns added explicitly --
# ALTER TABLE ADD COLUMN, never a destructive CREATE/DROP, so previously
# cached evidence/verdicts survive opening an old DB with this code.
_CALL_METADATA_COLUMNS = [
    ("toc_search_log", "call_metadata_json"),
    ("topic_extraction_runs", "call_metadata_json"),
    ("paper_boundary_log", "call_metadata_json"),
    ("paper_structure_log", "call_metadata_json"),
    ("question_extraction_log", "call_metadata_json"),
]

# Additive migration for Stage 3 (topic classification): a pre-existing
# `questions` table (from a database created before Stage 3 existed) gets
# these columns bolted on rather than the table being recreated, so
# already-extracted Stage 2 questions are never lost. classification_status
# gets a literal default so pre-existing rows read back as "unclassified"
# (never resolved) rather than NULL/empty -- SQLite backfills a column's
# DEFAULT into already-existing rows for ALTER TABLE ADD COLUMN.
_QUESTION_CLASSIFICATION_COLUMNS = [
    ("questions", "topic_id", "TEXT"),
    ("questions", "topic_name", "TEXT"),
    ("questions", "topic_confidence", "REAL"),
    ("questions", "classification_status", "TEXT DEFAULT 'unclassified'"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        for table, column in _CALL_METADATA_COLUMNS:
            try:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
        for table, column, coltype in _QUESTION_CLASSIFICATION_COLUMNS:
            try:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _cursor(self):
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # --- textbook_files ---

    def register_textbook(self, file_hash: str, file_path: str, total_pages: int) -> None:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO textbook_files (file_hash, file_path, total_pages, first_seen_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(file_hash) DO UPDATE SET file_path=excluded.file_path, total_pages=excluded.total_pages""",
                (file_hash, file_path, total_pages, _now()),
            )

    # --- page_range_evidence cache ---

    def get_cached_evidence(self, file_hash: str, start_page: int, end_page: int) -> PageRangeEvidence | None:
        with self._cursor() as cur:
            row = cur.execute(
                """SELECT evidence_json FROM page_range_evidence
                   WHERE file_hash=? AND start_page=? AND end_page=?""",
                (file_hash, start_page, end_page),
            ).fetchone()
        if row is None:
            return None
        return PageRangeEvidence.model_validate_json(row["evidence_json"])

    def save_evidence(self, file_hash: str, evidence: PageRangeEvidence) -> None:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO page_range_evidence
                   (file_hash, start_page, end_page, conversion_status, evidence_json, error_message, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(file_hash, start_page, end_page) DO UPDATE SET
                       conversion_status=excluded.conversion_status,
                       evidence_json=excluded.evidence_json,
                       error_message=excluded.error_message,
                       created_at=excluded.created_at""",
                (
                    file_hash,
                    evidence.start_page,
                    evidence.end_page,
                    evidence.conversion_status,
                    evidence.model_dump_json(),
                    evidence.error_message,
                    _now(),
                ),
            )

    # --- toc_search_log ---

    def get_toc_verdict(
        self, file_hash: str, start_page: int, end_page: int, model_used: str
    ) -> TOCDetectionResult | None:
        with self._cursor() as cur:
            row = cur.execute(
                """SELECT response_json FROM toc_search_log
                   WHERE file_hash=? AND start_page=? AND end_page=? AND model_used=?""",
                (file_hash, start_page, end_page, model_used),
            ).fetchone()
        if row is None:
            return None
        return TOCDetectionResult.model_validate_json(row["response_json"])

    def log_toc_verdict(
        self,
        file_hash: str,
        start_page: int,
        end_page: int,
        model_used: str,
        result: TOCDetectionResult,
        call_metadata: dict | None = None,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO toc_search_log
                   (file_hash, start_page, end_page, model_used, document_structure, confidence, response_json, created_at, call_metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(file_hash, start_page, end_page, model_used) DO UPDATE SET
                       document_structure=excluded.document_structure,
                       confidence=excluded.confidence,
                       response_json=excluded.response_json,
                       created_at=excluded.created_at,
                       call_metadata_json=excluded.call_metadata_json""",
                (
                    file_hash,
                    start_page,
                    end_page,
                    model_used,
                    result.document_structure,
                    result.confidence,
                    result.model_dump_json(),
                    _now(),
                    json.dumps(call_metadata) if call_metadata is not None else None,
                ),
            )

    # --- topic_extraction_runs ---

    def save_topic_extraction_run(
        self,
        file_hash: str,
        model_used: str,
        source_start_page: int,
        source_end_page: int,
        status: str,
        topics: TopicExtractionResult | None,
        validation: ValidationReport | None,
        call_metadata: dict | None = None,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO topic_extraction_runs
                   (file_hash, model_used, source_start_page, source_end_page, status, topics_json, validation_json, created_at, call_metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    file_hash,
                    model_used,
                    source_start_page,
                    source_end_page,
                    status,
                    topics.model_dump_json() if topics else None,
                    validation.model_dump_json() if validation else None,
                    _now(),
                    json.dumps(call_metadata) if call_metadata is not None else None,
                ),
            )

    # --- question_paper_files ---

    def register_question_paper_file(self, file_hash: str, file_path: str, total_pages: int) -> None:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO question_paper_files (file_hash, file_path, total_pages, status, first_seen_at)
                   VALUES (?, ?, ?, 'pending', ?)
                   ON CONFLICT(file_hash) DO UPDATE SET file_path=excluded.file_path, total_pages=excluded.total_pages""",
                (file_hash, file_path, total_pages, _now()),
            )

    def get_question_paper_file_status(self, file_hash: str) -> str | None:
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT status FROM question_paper_files WHERE file_hash=?", (file_hash,)
            ).fetchone()
        return row["status"] if row else None

    def update_question_paper_file_status(
        self, file_hash: str, status: str, model_used: str, error_message: str | None
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """UPDATE question_paper_files SET status=?, model_used=?, error_message=?, last_processed_at=?
                   WHERE file_hash=?""",
                (status, model_used, error_message, _now(), file_hash),
            )

    # --- paper_boundary_log ---

    def get_paper_boundary_verdict(self, file_hash: str, model_used: str) -> PaperBoundaryDetectionResult | None:
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT result_json FROM paper_boundary_log WHERE file_hash=? AND model_used=?",
                (file_hash, model_used),
            ).fetchone()
        if row is None:
            return None
        return PaperBoundaryDetectionResult.model_validate_json(row["result_json"])

    def save_paper_boundary_verdict(
        self, file_hash: str, model_used: str, result: PaperBoundaryDetectionResult, call_metadata: dict | None = None
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO paper_boundary_log (file_hash, model_used, result_json, created_at, call_metadata_json)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(file_hash, model_used) DO UPDATE SET
                       result_json=excluded.result_json, created_at=excluded.created_at,
                       call_metadata_json=excluded.call_metadata_json""",
                (file_hash, model_used, result.model_dump_json(), _now(), json.dumps(call_metadata) if call_metadata is not None else None),
            )

    # --- paper_structure_log ---

    def get_paper_structure_verdict(
        self, file_hash: str, start_page: int, end_page: int, model_used: str
    ) -> PaperStructureExtraction | None:
        with self._cursor() as cur:
            row = cur.execute(
                """SELECT result_json FROM paper_structure_log
                   WHERE file_hash=? AND start_page=? AND end_page=? AND model_used=?""",
                (file_hash, start_page, end_page, model_used),
            ).fetchone()
        if row is None:
            return None
        return PaperStructureExtraction.model_validate_json(row["result_json"])

    def save_paper_structure_verdict(
        self,
        file_hash: str,
        start_page: int,
        end_page: int,
        model_used: str,
        result: PaperStructureExtraction,
        call_metadata: dict | None = None,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO paper_structure_log (file_hash, start_page, end_page, model_used, result_json, created_at, call_metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(file_hash, start_page, end_page, model_used) DO UPDATE SET
                       result_json=excluded.result_json, created_at=excluded.created_at,
                       call_metadata_json=excluded.call_metadata_json""",
                (
                    file_hash, start_page, end_page, model_used, result.model_dump_json(), _now(),
                    json.dumps(call_metadata) if call_metadata is not None else None,
                ),
            )

    # --- question_extraction_log ---

    def get_question_extraction_verdict(
        self, file_hash: str, start_page: int, end_page: int, model_used: str
    ) -> QuestionExtractionResult | None:
        with self._cursor() as cur:
            row = cur.execute(
                """SELECT result_json FROM question_extraction_log
                   WHERE file_hash=? AND start_page=? AND end_page=? AND model_used=?""",
                (file_hash, start_page, end_page, model_used),
            ).fetchone()
        if row is None:
            return None
        return QuestionExtractionResult.model_validate_json(row["result_json"])

    def save_question_extraction_verdict(
        self,
        file_hash: str,
        start_page: int,
        end_page: int,
        model_used: str,
        result: QuestionExtractionResult,
        call_metadata: dict | None = None,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO question_extraction_log (file_hash, start_page, end_page, model_used, result_json, created_at, call_metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(file_hash, start_page, end_page, model_used) DO UPDATE SET
                       result_json=excluded.result_json, created_at=excluded.created_at,
                       call_metadata_json=excluded.call_metadata_json""",
                (
                    file_hash, start_page, end_page, model_used, result.model_dump_json(), _now(),
                    json.dumps(call_metadata) if call_metadata is not None else None,
                ),
            )

    # --- papers ---

    def save_paper(self, paper: Paper) -> None:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO papers (paper_id, file_hash, start_page, end_page, status, paper_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(paper_id) DO UPDATE SET
                       status=excluded.status, paper_json=excluded.paper_json, created_at=excluded.created_at""",
                (paper.paper_id, paper.file_hash, paper.start_page, paper.end_page, paper.status,
                 paper.model_dump_json(), _now()),
            )

    def get_paper(self, paper_id: str) -> Paper | None:
        with self._cursor() as cur:
            row = cur.execute("SELECT paper_json FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
        if row is None:
            return None
        return Paper.model_validate_json(row["paper_json"])

    def get_papers_for_file(self, file_hash: str) -> list[Paper]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT paper_json FROM papers WHERE file_hash=? ORDER BY start_page", (file_hash,)
            ).fetchall()
        return [Paper.model_validate_json(row["paper_json"]) for row in rows]

    def get_all_papers(self) -> list[Paper]:
        """Every detected exam paper across every processed question-paper
        file -- Stage 5 operates corpus-wide (like Stage 4), not scoped to
        one question-papers directory or textbook."""
        with self._cursor() as cur:
            rows = cur.execute("SELECT paper_json FROM papers ORDER BY paper_id").fetchall()
        return [Paper.model_validate_json(row["paper_json"]) for row in rows]

    # --- questions ---

    def save_questions(self, questions: list[Question]) -> None:
        # topic_id/topic_name/topic_confidence/classification_status are
        # written here from `q` too (Stage 2 always constructs a fresh
        # Question with classification_status="unclassified" -- see
        # schemas.Question) so a genuinely re-extracted paper's questions
        # correctly start over unclassified. This is safe against
        # clobbering Stage 3's own work: pipeline.py only ever reaches
        # save_questions for a paper that was NOT already status="success",
        # so an already-classified paper's rows are never re-written here.
        with self._cursor() as cur:
            for q in questions:
                cur.execute(
                    """INSERT INTO questions
                       (question_id, paper_id, section_id, source_filename, source_file_hash, exam_name,
                        institution, subject, paper_year, paper_date, question_number, question_text,
                        options_json, original_text, correction_applied, question_type, source_pages_json,
                        extraction_confidence, ambiguous, ambiguity_note, topic_id, topic_name,
                        topic_confidence, classification_status, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(question_id) DO UPDATE SET
                           section_id=excluded.section_id, source_filename=excluded.source_filename,
                           source_file_hash=excluded.source_file_hash, exam_name=excluded.exam_name,
                           institution=excluded.institution, subject=excluded.subject,
                           paper_year=excluded.paper_year, paper_date=excluded.paper_date,
                           question_number=excluded.question_number,
                           question_text=excluded.question_text, options_json=excluded.options_json,
                           original_text=excluded.original_text,
                           correction_applied=excluded.correction_applied, question_type=excluded.question_type,
                           source_pages_json=excluded.source_pages_json,
                           extraction_confidence=excluded.extraction_confidence,
                           ambiguous=excluded.ambiguous, ambiguity_note=excluded.ambiguity_note,
                           topic_id=excluded.topic_id, topic_name=excluded.topic_name,
                           topic_confidence=excluded.topic_confidence,
                           classification_status=excluded.classification_status""",
                    (
                        q.question_id, q.paper_id, q.section_id, q.source_filename, q.source_file_hash,
                        q.exam_name, q.institution, q.subject, q.paper_year, q.paper_date,
                        q.question_number, q.question_text, json.dumps(q.options),
                        q.original_text, int(q.correction_applied), q.question_type,
                        json.dumps(q.source_pages), q.extraction_confidence, int(q.ambiguous),
                        q.ambiguity_note, q.topic_id, q.topic_name, q.topic_confidence,
                        q.classification_status, _now(),
                    ),
                )

    @staticmethod
    def _question_from_row(row: sqlite3.Row) -> Question:
        return Question(
            question_id=row["question_id"],
            paper_id=row["paper_id"],
            section_id=row["section_id"],
            source_filename=row["source_filename"],
            source_file_hash=row["source_file_hash"],
            exam_name=row["exam_name"],
            institution=row["institution"],
            subject=row["subject"],
            paper_year=row["paper_year"],
            paper_date=row["paper_date"],
            question_number=row["question_number"],
            question_text=row["question_text"],
            options=json.loads(row["options_json"]),
            original_text=row["original_text"],
            correction_applied=bool(row["correction_applied"]),
            question_type=row["question_type"],
            source_pages=json.loads(row["source_pages_json"]),
            extraction_confidence=row["extraction_confidence"],
            ambiguous=bool(row["ambiguous"]),
            ambiguity_note=row["ambiguity_note"],
            topic_id=row["topic_id"],
            topic_name=row["topic_name"],
            topic_confidence=row["topic_confidence"],
            classification_status=row["classification_status"] or "unclassified",
        )

    def get_questions_for_paper(self, paper_id: str) -> list[Question]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM questions WHERE paper_id=? ORDER BY question_id", (paper_id,)
            ).fetchall()
        return [self._question_from_row(row) for row in rows]

    def get_all_questions(self) -> list[Question]:
        """Every question occurrence across every paper/textbook -- Stage 4
        operates corpus-wide, unlike Stage 3's per-paper granularity."""
        with self._cursor() as cur:
            rows = cur.execute("SELECT * FROM questions ORDER BY question_id").fetchall()
        return [self._question_from_row(row) for row in rows]

    def _questions_by_ids(self, question_ids: list[str]) -> dict[str, Question]:
        if not question_ids:
            return {}
        with self._cursor() as cur:
            placeholders = ",".join("?" for _ in question_ids)
            rows = cur.execute(
                f"SELECT * FROM questions WHERE question_id IN ({placeholders})", question_ids
            ).fetchall()
        return {row["question_id"]: self._question_from_row(row) for row in rows}

    # --- Stage 3: topic classification ---

    def get_latest_topic_extraction(self, file_hash: str) -> TopicExtractionResult | None:
        """The most recently saved non-failed topic list for this textbook
        (topics_json is only null for a "failed" run -- see
        save_topic_extraction_run). Stage 3 reads this as an already-
        completed Stage 1 prerequisite; it never re-runs Stage 1 itself."""
        with self._cursor() as cur:
            row = cur.execute(
                """SELECT topics_json FROM topic_extraction_runs
                   WHERE file_hash=? AND topics_json IS NOT NULL
                   ORDER BY id DESC LIMIT 1""",
                (file_hash,),
            ).fetchone()
        if row is None:
            return None
        return TopicExtractionResult.model_validate_json(row["topics_json"])

    def get_topic_classification_verdict(self, paper_id: str, model_used: str) -> TopicClassificationResult | None:
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT result_json FROM topic_classification_log WHERE paper_id=? AND model_used=?",
                (paper_id, model_used),
            ).fetchone()
        if row is None:
            return None
        return TopicClassificationResult.model_validate_json(row["result_json"])

    def save_topic_classification_verdict(
        self,
        paper_id: str,
        textbook_file_hash: str,
        model_used: str,
        result: TopicClassificationResult,
        call_metadata: dict | None = None,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO topic_classification_log
                   (paper_id, textbook_file_hash, model_used, result_json, created_at, call_metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(paper_id, model_used) DO UPDATE SET
                       textbook_file_hash=excluded.textbook_file_hash, result_json=excluded.result_json,
                       created_at=excluded.created_at, call_metadata_json=excluded.call_metadata_json""",
                (
                    paper_id, textbook_file_hash, model_used, result.model_dump_json(), _now(),
                    json.dumps(call_metadata) if call_metadata is not None else None,
                ),
            )

    def save_question_classifications(
        self, classifications: list[QuestionTopicClassification], topics_by_id: dict[str, Topic]
    ) -> None:
        """Writes each classification's resolved topic_id/topic_name/
        topic_confidence/classification_status onto its already-persisted
        `questions` row. Never inserts a new row -- a question_id that
        doesn't already exist (shouldn't happen; classifications are only
        ever built from questions read out of this same table) is simply a
        no-op UPDATE affecting zero rows, not an error."""
        with self._cursor() as cur:
            for c in classifications:
                topic_name = topics_by_id[c.topic_id].name if c.topic_id in topics_by_id else None
                topic_confidence = c.confidence if c.status != "unclassified" else None
                cur.execute(
                    """UPDATE questions SET topic_id=?, topic_name=?, topic_confidence=?, classification_status=?
                       WHERE question_id=?""",
                    (c.topic_id, topic_name, topic_confidence, c.status, c.question_id),
                )

    # --- Stage 4: semantic deduplication ---

    def get_question_embedding(self, question_id: str, model: str, version: str) -> QuestionEmbedding | None:
        with self._cursor() as cur:
            row = cur.execute(
                """SELECT content_hash, vector_json FROM question_embeddings
                   WHERE question_id=? AND embedding_model=? AND embedding_version=?""",
                (question_id, model, version),
            ).fetchone()
        if row is None:
            return None
        return QuestionEmbedding(
            question_id=question_id, model=model, version=version,
            content_hash=row["content_hash"], vector=json.loads(row["vector_json"]),
        )

    def save_question_embedding(self, embedding: QuestionEmbedding) -> None:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO question_embeddings
                   (question_id, embedding_model, embedding_version, content_hash, dim, vector_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(question_id, embedding_model, embedding_version) DO UPDATE SET
                       content_hash=excluded.content_hash, dim=excluded.dim,
                       vector_json=excluded.vector_json, created_at=excluded.created_at""",
                (
                    embedding.question_id, embedding.model, embedding.version, embedding.content_hash,
                    len(embedding.vector), json.dumps(embedding.vector), _now(),
                ),
            )

    # --- dedup_pair_decisions (cache + audit trail) ---

    _DETERMINISTIC_MODEL_SENTINEL = "deterministic"

    def get_dedup_pair_decision(
        self, question_id_a: str, question_id_b: str, model_used: str
    ) -> DedupPairDecision | None:
        """Caller must pass question_id_a < question_id_b (lexicographic) --
        this is a raw cache lookup, it does not reorder for you. model_used
        should be Database._DETERMINISTIC_MODEL_SENTINEL when looking up an
        exact-duplicate decision (see save_dedup_pair_decision)."""
        with self._cursor() as cur:
            row = cur.execute(
                """SELECT decision_source, verdict, confidence, reason FROM dedup_pair_decisions
                   WHERE question_id_a=? AND question_id_b=? AND model_used=?""",
                (question_id_a, question_id_b, model_used),
            ).fetchone()
        if row is None:
            return None
        return DedupPairDecision(
            question_id_a=question_id_a, question_id_b=question_id_b,
            decision_source=row["decision_source"], verdict=row["verdict"],
            confidence=row["confidence"], reason=row["reason"],
        )

    def save_dedup_pair_decision(
        self, decision: DedupPairDecision, model_used: str | None = None, call_metadata: dict | None = None
    ) -> None:
        """model_used identifies which LLM verdict this is cached against
        (config.model_identifier) for decision_source="llm_judge"; omit (or
        pass None) for decision_source="exact_duplicate", which is stored
        under a fixed sentinel instead of a real model name since no LLM was
        involved. question_id_a/b must already be in a<b order -- callers
        build DedupPairDecision that way (see deduplication/candidates.py)."""
        effective_model = model_used if decision.decision_source == "llm_judge" else self._DETERMINISTIC_MODEL_SENTINEL
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO dedup_pair_decisions
                   (question_id_a, question_id_b, decision_source, model_used, verdict, confidence, reason, created_at, call_metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(question_id_a, question_id_b, model_used) DO UPDATE SET
                       decision_source=excluded.decision_source, verdict=excluded.verdict,
                       confidence=excluded.confidence, reason=excluded.reason,
                       created_at=excluded.created_at, call_metadata_json=excluded.call_metadata_json""",
                (
                    decision.question_id_a, decision.question_id_b, decision.decision_source, effective_model,
                    decision.verdict, decision.confidence, decision.reason, _now(),
                    json.dumps(call_metadata) if call_metadata is not None else None,
                ),
            )

    def list_dedup_pair_decisions(self, verdict: str | None = None) -> list[DedupPairDecision]:
        """Every persisted pair decision (optionally filtered by verdict) --
        the full audit trail, used by output_writer.py to surface
        needs_review pairs and by tests to assert on what got cached."""
        with self._cursor() as cur:
            if verdict is None:
                rows = cur.execute(
                    "SELECT question_id_a, question_id_b, decision_source, verdict, confidence, reason "
                    "FROM dedup_pair_decisions ORDER BY question_id_a, question_id_b"
                ).fetchall()
            else:
                rows = cur.execute(
                    "SELECT question_id_a, question_id_b, decision_source, verdict, confidence, reason "
                    "FROM dedup_pair_decisions WHERE verdict=? ORDER BY question_id_a, question_id_b",
                    (verdict,),
                ).fetchall()
        return [
            DedupPairDecision(
                question_id_a=row["question_id_a"], question_id_b=row["question_id_b"],
                decision_source=row["decision_source"], verdict=row["verdict"],
                confidence=row["confidence"], reason=row["reason"],
            )
            for row in rows
        ]

    # --- canonical_questions / question_canonical_links ---

    def replace_canonical_state(self, canonical_questions: list[CanonicalQuestion], links: dict[str, str]) -> None:
        """Full replace, in one transaction: Stage 4 is deterministic given
        its cached embeddings/pair decisions, so each run recomputes the
        entire canonical grouping from scratch rather than trying to
        incrementally patch a prior run's groups -- see
        deduplication/pipeline.py. This never touches `questions` itself,
        only these two additive tables. `links` maps question_id ->
        canonical_question_id; every question_id present gets exactly one
        row (question_id is this table's PRIMARY KEY)."""
        now = _now()
        with self._cursor() as cur:
            cur.execute("DELETE FROM question_canonical_links")
            cur.execute("DELETE FROM canonical_questions")
            for cq in canonical_questions:
                cur.execute(
                    """INSERT INTO canonical_questions
                       (canonical_question_id, canonical_question_text, question_type, topic_id, topic_name,
                        representative_question_id, dedup_confidence, dedup_status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        cq.canonical_question_id, cq.canonical_question_text, cq.question_type,
                        cq.topic_id, cq.topic_name, cq.representative_question_id,
                        cq.dedup_confidence, cq.dedup_status, now, now,
                    ),
                )
            for question_id, canonical_question_id in links.items():
                cur.execute(
                    "INSERT INTO question_canonical_links (question_id, canonical_question_id, created_at) VALUES (?, ?, ?)",
                    (question_id, canonical_question_id, now),
                )

    def _hydrate_canonical(self, base_rows: list[sqlite3.Row]) -> list[CanonicalQuestion]:
        """Fills in the derived fields (occurrence_count, unique_paper_count,
        years, source_question_ids) by joining against question_canonical_links
        + questions -- these are never stored directly on canonical_questions
        (see CanonicalQuestion's docstring)."""
        if not base_rows:
            return []
        canonical_ids = [row["canonical_question_id"] for row in base_rows]
        with self._cursor() as cur:
            placeholders = ",".join("?" for _ in canonical_ids)
            link_rows = cur.execute(
                f"""SELECT question_id, canonical_question_id FROM question_canonical_links
                    WHERE canonical_question_id IN ({placeholders})""",
                canonical_ids,
            ).fetchall()
        question_ids_by_canonical: dict[str, list[str]] = {}
        for lr in link_rows:
            question_ids_by_canonical.setdefault(lr["canonical_question_id"], []).append(lr["question_id"])

        all_question_ids = [qid for ids in question_ids_by_canonical.values() for qid in ids]
        questions_by_id = self._questions_by_ids(all_question_ids)

        results: list[CanonicalQuestion] = []
        for row in base_rows:
            question_ids = sorted(question_ids_by_canonical.get(row["canonical_question_id"], []))
            member_questions = [questions_by_id[qid] for qid in question_ids if qid in questions_by_id]
            years = sorted({q.paper_year for q in member_questions if q.paper_year})
            unique_papers = {q.paper_id for q in member_questions}
            results.append(
                CanonicalQuestion(
                    canonical_question_id=row["canonical_question_id"],
                    canonical_question_text=row["canonical_question_text"],
                    question_type=row["question_type"],
                    topic_id=row["topic_id"],
                    topic_name=row["topic_name"],
                    representative_question_id=row["representative_question_id"],
                    dedup_confidence=row["dedup_confidence"],
                    dedup_status=row["dedup_status"],
                    occurrence_count=len(question_ids),
                    unique_paper_count=len(unique_papers),
                    years=years,
                    source_question_ids=question_ids,
                )
            )
        return results

    def get_canonical_question_detail(self, canonical_question_id: str) -> CanonicalQuestion | None:
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT * FROM canonical_questions WHERE canonical_question_id=?", (canonical_question_id,)
            ).fetchone()
        if row is None:
            return None
        return self._hydrate_canonical([row])[0]

    def list_canonical_questions(self) -> list[CanonicalQuestion]:
        with self._cursor() as cur:
            rows = cur.execute("SELECT * FROM canonical_questions ORDER BY canonical_question_id").fetchall()
        return self._hydrate_canonical(rows)

    def get_canonical_question_id_for_question(self, question_id: str) -> str | None:
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT canonical_question_id FROM question_canonical_links WHERE question_id=?", (question_id,)
            ).fetchone()
        return row["canonical_question_id"] if row else None

    # --- analysis_jobs (GUI pause/resume tracking, see schemas.AnalysisJob) ---

    @staticmethod
    def _analysis_job_from_row(row: sqlite3.Row) -> AnalysisJob:
        return AnalysisJob(
            job_id=row["job_id"], topic_source_type=row["topic_source_type"],
            topic_source_path=row["topic_source_path"], question_papers_dir=row["question_papers_dir"],
            status=row["status"], pause_reason=row["pause_reason"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def upsert_analysis_job(
        self, job_id: str, topic_source_type: str, topic_source_path: str, question_papers_dir: str, status: str
    ) -> None:
        """Called at the start of an `analyze` run (both cli.py and app.py)
        to record/resume this run's identity. Always clears any previous
        pause_reason -- a fresh "running" status means whatever paused it
        before no longer applies. created_at is preserved across an
        already-existing job_id (a genuine resume), never reset."""
        now = _now()
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO analysis_jobs
                   (job_id, topic_source_type, topic_source_path, question_papers_dir, status, pause_reason, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
                   ON CONFLICT(job_id) DO UPDATE SET
                       topic_source_type=excluded.topic_source_type,
                       topic_source_path=excluded.topic_source_path,
                       question_papers_dir=excluded.question_papers_dir,
                       status=excluded.status,
                       pause_reason=NULL,
                       updated_at=excluded.updated_at""",
                (job_id, topic_source_type, topic_source_path, question_papers_dir, status, now, now),
            )

    def set_analysis_job_status(self, job_id: str, status: str, pause_reason: str | None = None) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE analysis_jobs SET status=?, pause_reason=?, updated_at=? WHERE job_id=?",
                (status, pause_reason, _now(), job_id),
            )

    def get_analysis_job(self, job_id: str) -> AnalysisJob | None:
        with self._cursor() as cur:
            row = cur.execute("SELECT * FROM analysis_jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._analysis_job_from_row(row) if row else None

    def get_latest_unfinished_analysis_job(self) -> AnalysisJob | None:
        """The most recently touched job that hasn't reached status
        "completed" -- running (including one a crash left stuck there),
        paused, or failed all count as "not done, safe to offer resuming"
        (see schemas.AnalysisJob's docstring). Used at GUI startup to detect
        an interrupted analysis from a prior session."""
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT * FROM analysis_jobs WHERE status != 'completed' ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        return self._analysis_job_from_row(row) if row else None
