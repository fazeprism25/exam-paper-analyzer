#!/usr/bin/env python
"""Command-line entry point.

    python cli.py extract-topics --textbook <path to any textbook PDF>
    python cli.py extract-questions --question-papers <dir of exam-paper PDFs>
    python cli.py classify-topics --textbook <same textbook PDF> --question-papers <same dir>
    python cli.py deduplicate
    python cli.py analyze-frequency --textbook <same textbook PDF>
    python cli.py generate-report --textbook <same textbook PDF>

classify-topics (Stage 3) reads both prior stages' already-persisted
results from the database -- it does not re-run either of them, so
extract-topics and extract-questions must each have completed successfully
first for the given textbook/papers.

deduplicate (Stage 4) reads every already-extracted question occurrence
straight from the database, corpus-wide across every paper/textbook run so
far -- it takes no --textbook/--question-papers arguments at all, and does
not require Stage 3 (topic classification) to have run, though it uses
topic_id/topic_name as contextual evidence when available.

analyze-frequency (Stage 5) is a deterministic aggregation/reporting stage
over already-persisted Stage 1-4 data -- it makes zero LLM calls and zero
embedding calls. It takes --textbook (not --question-papers) because it
needs Stage 1's full topic hierarchy, including topics with zero questions,
which extract-topics/classify-topics already keyed to the textbook's file
hash. It requires extract-topics, extract-questions, and deduplicate to
have already run; classify-topics is not strictly required (unclassified
questions are still reported, just not attributed to a topic).

generate-report (Stage 6) is the final user-facing presentation layer over
Stage 5's own aggregation -- it makes zero LLM calls and zero embedding
calls, and requires no API key. It runs Stage 5's aggregation itself
(cheap, deterministic, no cost) rather than requiring analyze-frequency to
have been run first, so it takes the same --textbook argument and the same
extract-topics/extract-questions/deduplicate prerequisites. It writes its
own output under output/final_analysis/ and never touches Stage 1-5 output.

Every command that makes LLM calls (extract-topics, extract-questions,
classify-topics, deduplicate) defaults to OpenRouter (--provider openrouter,
the default) -- a normal user just needs their own OpenRouter API key
(--openrouter-api-key or the OPENROUTER_API_KEY environment variable), no
local model install. Ollama remains available as an optional local
dev/testing backend via --provider ollama. analyze-frequency and
generate-report make no LLM calls at all, so neither takes --provider or an
API key.

Nothing about the textbook or exam papers is hard-coded here or anywhere in
the pipeline -- the paths are the only required arguments.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

from exampapersorter import api_key_store
from exampapersorter.analyze_pipeline import classify_papers_for_files, extract_questions_for_files
from exampapersorter.config import DEFAULT_CONFIG, Config
from exampapersorter.database import Database
from exampapersorter.deduplication.pipeline import run_deduplication
from exampapersorter.final_analysis.pipeline import run_final_analysis
from exampapersorter.frequency_analysis.pipeline import FrequencyAnalysisPrerequisiteError, run_frequency_analysis
from exampapersorter.llm_client import LLMCallFailed, get_call_metrics, get_last_call_metadata, reset_call_metrics
from exampapersorter.llm_providers.openrouter import validate_api_key
from exampapersorter.logging_setup import redact_secret_from_logs, setup_logging
from exampapersorter.output_writer import (
    write_stage1_outputs,
    write_stage3_outputs,
    write_stage4_outputs,
    write_stage5_outputs,
    write_stage6_outputs,
)
from exampapersorter.pdf_utils import compute_file_hash, get_page_count
from exampapersorter.schemas import TopicExtractionResult
from exampapersorter.topic_authority import TopicAuthorityError, resolve_index_topic_authority, resolve_textbook_topic_authority
from exampapersorter.topic_extraction.extract import extract_topics
from exampapersorter.topic_extraction.hierarchy import derive_parent_ids
from exampapersorter.topic_extraction.name_recovery import recover_names
from exampapersorter.topic_extraction.search import find_table_of_contents
from exampapersorter.validation import reconcile_resolution_status, validate_topics

logger = logging.getLogger(__name__)


def _call_metadata_dict() -> dict | None:
    meta = get_last_call_metadata()
    return meta.to_dict() if meta else None


def _resolve_llm_config(args: argparse.Namespace, base: Config) -> Config:
    """Applies --provider/--model/--openrouter-api-key on top of `base`,
    and for OpenRouter (the default), validates the key up front with a
    cheap, non-quota-spending check (see llm_providers.openrouter.
    validate_api_key) so a bad/missing key fails clearly before any PDF
    processing starts rather than surfacing as an opaque LLM call failure
    many minutes into a run."""
    provider = args.provider or DEFAULT_CONFIG.llm_provider

    if provider == "ollama":
        if args.openrouter_api_key:
            logger.warning("--openrouter-api-key was given but --provider is ollama; ignoring it")
        return replace(base, llm_provider="ollama", ollama_model=args.model or DEFAULT_CONFIG.ollama_model)

    if provider != "openrouter":
        logger.error("Unknown --provider %r (expected 'openrouter' or 'ollama')", provider)
        sys.exit(1)

    if args.model:
        logger.warning("--model has no effect with --provider openrouter (the model pool is centrally configured); ignoring it")

    explicit_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    # Falls back to a previously-saved key (see api_key_store.py) only when
    # neither the CLI flag nor the env var supplied one -- an explicit
    # source always wins over whatever was saved before, so a user testing
    # a different key never gets silently overridden by an old saved one.
    api_key = explicit_key or api_key_store.load_saved_key()
    if not api_key:
        logger.error(
            "No OpenRouter API key supplied. Pass --openrouter-api-key or set the "
            "OPENROUTER_API_KEY environment variable. Get a key at https://openrouter.ai/keys"
        )
        sys.exit(1)
    redact_secret_from_logs(api_key)

    config = replace(base, llm_provider="openrouter", openrouter_api_key=api_key)
    ok, message = validate_api_key(config)
    if not ok:
        logger.error("OpenRouter API key check failed: %s", message)
        sys.exit(1)
    logger.info(
        "OpenRouter API key OK. Model pool (%d models, tried in order with automatic fallback): %s",
        len(config.openrouter_models), ", ".join(config.openrouter_models),
    )

    if explicit_key and getattr(args, "save_key", False):
        if api_key_store.save_key(explicit_key):
            logger.info("Saved OpenRouter API key to the OS credential store for future runs (--openrouter-api-key/env var won't be needed next time)")
        else:
            logger.warning("Could not save the OpenRouter API key locally (no OS credential backend available) -- you'll need to supply it again next run")

    return config


def cmd_extract_topics(args: argparse.Namespace) -> None:
    config = replace(
        DEFAULT_CONFIG,
        maximum_textbook_search_pages=args.max_search_pages or DEFAULT_CONFIG.maximum_textbook_search_pages,
        initial_textbook_search_range=args.search_window or DEFAULT_CONFIG.initial_textbook_search_range,
        llm_seed=args.seed if args.seed is not None else DEFAULT_CONFIG.llm_seed,
        log_level="DEBUG" if args.debug else DEFAULT_CONFIG.log_level,
    )
    setup_logging(config.log_level)
    config = _resolve_llm_config(args, config)

    textbook_path = Path(args.textbook)
    if not textbook_path.exists():
        logger.error("Textbook not found: %s", textbook_path)
        sys.exit(1)

    total_pages = get_page_count(textbook_path)
    file_hash = compute_file_hash(textbook_path)
    logger.info("Textbook: %s (%d pages, hash=%s)", textbook_path.name, total_pages, file_hash[:12])

    db = Database(config.database_path)
    db.register_textbook(file_hash, str(textbook_path), total_pages)

    search_cap = min(config.maximum_textbook_search_pages, total_pages)
    logger.info(
        "Searching pages 1-%d (window=%d, step=%d) for table of contents / index using model=%s",
        search_cap, config.initial_textbook_search_range, config.textbook_search_step, config.model_identifier,
    )
    search_result = find_table_of_contents(textbook_path, file_hash, total_pages, db, config)

    if not search_result.found:
        print("\n=== Stage 1 FAILED: no table of contents/index found ===")
        print(f"Searched pages 1-{search_cap}")
        print("Ranges checked:")
        for a in search_result.attempts:
            print(
                f"  pages {a.start_page}-{a.end_page}: status={a.conversion_status} "
                f"structure={a.document_structure} confidence={a.confidence} note={a.note}"
            )
        db.close()
        sys.exit(1)

    logger.info(
        "Extracting topics from pages %d-%d", search_result.evidence.start_page, search_result.evidence.end_page
    )
    try:
        topics = extract_topics(search_result.evidence, config)
    except LLMCallFailed as exc:
        logger.error("Topic extraction failed: %s", exc)
        db.save_topic_extraction_run(
            file_hash, config.model_identifier, search_result.evidence.start_page, search_result.evidence.end_page,
            status="failed", topics=None, validation=None,
        )
        db.close()
        sys.exit(1)

    topics = TopicExtractionResult(topics=reconcile_resolution_status(topics.topics))

    unresolved_count = sum(1 for t in topics.topics if t.resolution_status == "name_unresolved")
    if unresolved_count:
        logger.info("Attempting name recovery for %d topics via chapter-opening pages", unresolved_count)
        recovered = recover_names(topics.topics, search_result.evidence, textbook_path, file_hash, db, config)
        topics = TopicExtractionResult(topics=recovered)

    topics = TopicExtractionResult(topics=derive_parent_ids(topics.topics))

    valid_page_range = (search_result.evidence.start_page, search_result.evidence.end_page)
    validation = validate_topics(topics, config, valid_page_range=valid_page_range)
    status = "success" if validation.passed else "needs_review"
    db.save_topic_extraction_run(
        file_hash, config.model_identifier, search_result.evidence.start_page, search_result.evidence.end_page,
        status=status, topics=topics, validation=validation, call_metadata=_call_metadata_dict(),
    )

    topics_path, evidence_path = write_stage1_outputs(
        config.output_directory, textbook_path, topics, validation,
        search_result.evidence, search_result.verdict, search_result.attempts, config.model_identifier,
    )
    db.close()

    print("\n=== Stage 1 complete ===")
    print(f"Textbook: {textbook_path.name} ({total_pages} pages)")
    print(f"Model: {config.model_identifier}")
    print(
        f"TOC/index found on pages {search_result.evidence.start_page}-{search_result.evidence.end_page} "
        f"(structure={search_result.verdict.document_structure}, confidence={search_result.verdict.confidence:.2f})"
    )
    print("Ranges searched: " + ", ".join(f"{a.start_page}-{a.end_page}" for a in search_result.attempts))
    print(f"Topics extracted: {len(topics.topics)}")
    print(f"Validation: {'PASSED' if validation.passed else 'ISSUES FOUND'}")
    for issue in validation.issues:
        print(f"  [{issue.severity}] {issue.code}: {issue.message}")
    print(f"\nSaved: {topics_path}")
    print(f"Saved: {evidence_path}")
    print("\nTopic list:")
    _print_topic_tree(topics.topics)


def _print_topic_tree(topics: list) -> None:
    """Render by actual parent_id linkage, not the raw `level` field --
    `level` alone is misleading when a child topic's level isn't exactly
    parent_level + 1 (e.g. a corrective title node one level removed)."""
    by_parent: dict[str | None, list] = {}
    for t in topics:
        by_parent.setdefault(t.parent_id, []).append(t)

    def walk(parent_id: str | None, depth: int) -> None:
        for t in by_parent.get(parent_id, []):
            print(f"{'  ' * depth}- [{t.id}] {t.name} (pages: {t.source_pages})")
            walk(t.id, depth + 1)

    walk(None, 0)


def cmd_extract_questions(args: argparse.Namespace) -> None:
    config = replace(
        DEFAULT_CONFIG,
        llm_seed=args.seed if args.seed is not None else DEFAULT_CONFIG.llm_seed,
        # Stage 2 makes many more LLM calls per run than Stage 1 did, so the
        # "never reload" default that keeps Stage 1's tested behavior isn't
        # the right choice here -- see config.py's llm_reload_policy comment
        # for the reasoning (grounded in the Stage 1 reproducibility finding).
        # Only consulted for --provider ollama; OpenRouter has nothing to reload.
        llm_reload_policy=args.reload_policy or "reload_on_failure",
        log_level="DEBUG" if args.debug else DEFAULT_CONFIG.log_level,
    )
    setup_logging(config.log_level)
    config = _resolve_llm_config(args, config)

    question_papers_dir = Path(args.question_papers)
    if not question_papers_dir.is_dir():
        logger.error("Not a directory: %s", question_papers_dir)
        sys.exit(1)

    pdf_paths = sorted(question_papers_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.error("No PDF files found in %s", question_papers_dir)
        sys.exit(1)

    reset_call_metrics()
    db = Database(config.database_path)
    file_results, question_counts_by_paper, question_type_counts = extract_questions_for_files(pdf_paths, config, db)
    db.close()
    metrics = get_call_metrics()

    total_papers = sum(len(r.papers) for r in file_results)
    total_sections = sum(len(p.sections) for r in file_results for p in r.papers)
    total_questions = sum(question_counts_by_paper.values())

    print("\n=== Stage 2 complete ===")
    print(f"Question-paper PDFs processed: {len(pdf_paths)}")
    print(f"Papers detected: {total_papers}")
    print(f"Sections detected: {total_sections}")
    print(f"Questions extracted: {total_questions}")
    print(f"Question types: {question_type_counts}")
    for r in file_results:
        status_line = f"  {Path(r.file_path).name}: status={r.status}, papers={len(r.papers)}"
        print(status_line)
        for issue in r.boundary_issues:
            print(f"    [boundary:{issue.code}] {issue.message}")
        for p in r.papers:
            print(f"    - {p.paper_id}: pages {p.start_page}-{p.end_page}, status={p.status}, "
                  f"sections={len(p.sections)}, questions={question_counts_by_paper.get(p.paper_id, 0)}")
    print(
        f"\nLLM calls: {metrics.total_calls} (attempts={metrics.total_attempts}, "
        f"retries={metrics.total_retries}, reloads={metrics.total_reloads}, "
        f"failures={metrics.total_failures}, total_elapsed={metrics.total_elapsed_seconds:.1f}s)"
    )
    print(f"Provider: {config.llm_provider} (model_identifier={config.model_identifier})")
    print(f"Reload policy: {config.llm_reload_policy}")
    print(f"\nOutputs written under: {config.output_directory / 'question_extraction'}")


def cmd_classify_topics(args: argparse.Namespace) -> None:
    config = replace(
        DEFAULT_CONFIG,
        llm_seed=args.seed if args.seed is not None else DEFAULT_CONFIG.llm_seed,
        log_level="DEBUG" if args.debug else DEFAULT_CONFIG.log_level,
    )
    setup_logging(config.log_level)
    config = _resolve_llm_config(args, config)

    textbook_path = Path(args.textbook)
    if not textbook_path.exists():
        logger.error("Textbook not found: %s", textbook_path)
        sys.exit(1)

    question_papers_dir = Path(args.question_papers)
    if not question_papers_dir.is_dir():
        logger.error("Not a directory: %s", question_papers_dir)
        sys.exit(1)

    pdf_paths = sorted(question_papers_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.error("No PDF files found in %s", question_papers_dir)
        sys.exit(1)

    db = Database(config.database_path)

    textbook_file_hash = compute_file_hash(textbook_path)
    topics_result = db.get_latest_topic_extraction(textbook_file_hash)
    if topics_result is None:
        logger.error(
            "No topic list found for %s. Run 'extract-topics --textbook %s' first.",
            textbook_path.name, textbook_path,
        )
        db.close()
        sys.exit(1)
    topics = topics_result.topics
    logger.info("Loaded %d topics for %s (hash=%s)", len(topics), textbook_path.name, textbook_file_hash[:12])

    file_summaries, questions_by_paper, totals, skipped = classify_papers_for_files(
        pdf_paths, topics, textbook_file_hash, config, db
    )
    db.close()

    summary_path, by_topic_path = write_stage3_outputs(
        config.output_directory, textbook_path, config.model_identifier, topics, file_summaries, questions_by_paper
    )

    total_questions = sum(totals.values())
    print("\n=== Stage 3 complete ===")
    print(f"Textbook: {textbook_path.name} ({len(topics)} topics)")
    print(f"Provider: {config.llm_provider} (model_identifier={config.model_identifier})")
    print(f"Questions classified: {total_questions}")
    print(f"  classified={totals['classified']}, no_match={totals['no_match']}, "
          f"ambiguous={totals['ambiguous']}, unclassified={totals['unclassified']}")
    if skipped:
        print(f"Skipped (Stage 2 not yet run successfully): {', '.join(skipped)}")
    for fs in file_summaries:
        print(f"  {fs['source_pdf']}:")
        for p in fs["papers"]:
            print(
                f"    - {p['paper_id']}: status={p['status']}, questions={p['total_questions']}, "
                f"classified={p['classified_count']}, no_match={p['no_match_count']}, "
                f"ambiguous={p['ambiguous_count']}, unclassified={p['unclassified_count']}"
                + (f" -- {p['error_message']}" if p["error_message"] else "")
            )
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {by_topic_path}")


def cmd_deduplicate(args: argparse.Namespace) -> None:
    config = replace(
        DEFAULT_CONFIG,
        llm_seed=args.seed if args.seed is not None else DEFAULT_CONFIG.llm_seed,
        log_level="DEBUG" if args.debug else DEFAULT_CONFIG.log_level,
    )
    setup_logging(config.log_level)
    config = _resolve_llm_config(args, config)

    db = Database(config.database_path)

    if not db.get_all_questions():
        logger.error(
            "No questions found in the database. Run 'extract-questions' (Stage 2) first -- "
            "Stage 4 deduplicates already-extracted question occurrences, it does not extract from PDFs itself."
        )
        db.close()
        sys.exit(1)

    result = run_deduplication(config, db)

    canonical_questions = db.list_canonical_questions()
    needs_review_pairs = db.list_dedup_pair_decisions(verdict="uncertain")

    run_stats = {
        "provider": config.llm_provider,
        "model_identifier": config.model_identifier,
        "embedding_model": config.embedding_model,
        "embedding_model_version": config.embedding_model_version,
        "embedding_similarity_threshold": config.embedding_similarity_threshold,
        "embedding_cross_type_similarity_threshold": config.embedding_cross_type_similarity_threshold,
        "dedup_min_merge_confidence": config.dedup_min_merge_confidence,
        "total_questions": result.total_questions,
        "exact_duplicate_groups": result.exact_duplicate_groups,
        "candidate_pairs_generated": result.candidate_pairs_generated,
        "llm_pairs_reused_from_cache": result.llm_pairs_reused_from_cache,
        "llm_pairs_newly_judged": result.llm_pairs_newly_judged,
        "llm_batches_failed": result.llm_batches_failed,
        "canonical_group_count": result.canonical_group_count,
        "status_counts": result.status_counts,
    }
    summary_path, canonical_path, duplicate_groups_path = write_stage4_outputs(
        config.output_directory, run_stats, canonical_questions, needs_review_pairs,
    )
    db.close()

    print("\n=== Stage 4 complete ===")
    print(f"Provider: {config.llm_provider} (model_identifier={config.model_identifier})")
    print(f"Embedding model: {config.embedding_model} (version={config.embedding_model_version})")
    print(f"Questions: {result.total_questions}")
    print(
        f"Exact-duplicate groups: {result.exact_duplicate_groups} | "
        f"Candidate pairs: {result.candidate_pairs_generated} "
        f"(reused={result.llm_pairs_reused_from_cache}, newly judged={result.llm_pairs_newly_judged}, "
        f"failed batches={result.llm_batches_failed})"
    )
    print(f"Canonical groups: {result.canonical_group_count} -- {result.status_counts}")
    if needs_review_pairs:
        print(f"Pairs flagged needs_review: {len(needs_review_pairs)}")
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {canonical_path}")
    print(f"Saved: {duplicate_groups_path}")


def cmd_analyze_frequency(args: argparse.Namespace) -> None:
    config = replace(
        DEFAULT_CONFIG,
        log_level="DEBUG" if args.debug else DEFAULT_CONFIG.log_level,
    )
    setup_logging(config.log_level)

    textbook_path = Path(args.textbook)
    if not textbook_path.exists():
        logger.error("Textbook not found: %s", textbook_path)
        sys.exit(1)

    textbook_file_hash = compute_file_hash(textbook_path)
    db = Database(config.database_path)

    try:
        summary = run_frequency_analysis(db, textbook_file_hash)
    except FrequencyAnalysisPrerequisiteError as exc:
        logger.error(str(exc))
        db.close()
        sys.exit(1)
    db.close()

    summary_path, report_path = write_stage5_outputs(config.output_directory, summary)

    print("\n=== Stage 5 complete ===")
    print(f"Textbook: {textbook_path.name} ({summary.topic_count} topics)")
    print(
        f"Occurrences: {summary.total_question_occurrences} | "
        f"Canonical questions: {summary.total_canonical_questions} | Papers: {summary.total_papers}"
    )
    print(f"Question types: {summary.question_type_distribution.model_dump()}")
    integrity = summary.data_integrity
    print(f"Data integrity: {'OK' if integrity.reconciled else 'ISSUES FOUND'}")
    if not integrity.reconciled:
        print(f"  orphaned occurrences: {len(integrity.orphaned_occurrences)}")
        print(f"  duplicate links: {len(integrity.duplicate_links)}")
    print(f"Most repeated questions (occurrence_count > 1): {len(summary.most_repeated_questions)}")
    print("Most tested chapters (top 5):")
    for r in summary.most_tested_topics[:5]:
        print(f"  {r.rank}. {r.topic_name} -- {r.total_occurrences} occurrences ({r.unique_canonical_questions} unique)")
    if summary.no_match_questions or summary.ambiguous_questions or summary.unclassified_questions:
        print(
            f"Not topic-assigned: no_match={len(summary.no_match_questions)}, "
            f"ambiguous={len(summary.ambiguous_questions)}, unclassified={len(summary.unclassified_questions)}"
        )
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {report_path}")


def cmd_generate_report(args: argparse.Namespace) -> None:
    # No _resolve_llm_config call, deliberately -- Stage 6 makes zero LLM
    # calls and needs no API key (mirrors cmd_analyze_frequency).
    config = replace(
        DEFAULT_CONFIG,
        log_level="DEBUG" if args.debug else DEFAULT_CONFIG.log_level,
    )
    setup_logging(config.log_level)

    textbook_path = Path(args.textbook)
    if not textbook_path.exists():
        logger.error("Textbook not found: %s", textbook_path)
        sys.exit(1)

    textbook_file_hash = compute_file_hash(textbook_path)
    db = Database(config.database_path)

    try:
        report = run_final_analysis(db, textbook_file_hash)
    except FrequencyAnalysisPrerequisiteError as exc:
        logger.error(str(exc))
        db.close()
        sys.exit(1)
    db.close()

    json_path, report_path = write_stage6_outputs(config.output_directory, report)

    s = report.summary
    print("\n=== Stage 6 complete ===")
    print(f"Textbook: {textbook_path.name}")
    print(
        f"Papers: {s.total_papers} | Occurrences: {s.total_question_occurrences} | "
        f"Canonical questions: {s.total_canonical_questions}"
    )
    print(f"Topics: {s.total_topics} ({s.topics_tested} tested, {s.topics_untested} untested)")
    print(f"Repeated canonical question groups: {s.repeated_canonical_question_count}")
    if s.no_match_count or s.ambiguous_count or s.unclassified_count:
        print(
            f"Not topic-assigned: no_match={s.no_match_count}, ambiguous={s.ambiguous_count}, "
            f"unclassified={s.unclassified_count}"
        )
    print(f"Data integrity: {'OK' if s.data_integrity_reconciled else 'ISSUES FOUND'}")
    print(f"\nSaved: {json_path}")
    print(f"Saved: {report_path}")


def _ok(message: str) -> None:
    print(f"[OK] {message}")


def cmd_analyze(args: argparse.Namespace) -> None:
    """The single end-user entry point: topic source + exam-paper folder +
    API key in, final report out. Internally this is nothing more than
    topic-authority resolution (topic_authority.py) followed by Stages
    2/3/4/6 in sequence, using the exact same functions their own CLI
    commands use (extract_questions_for_files, classify_papers_for_files,
    run_deduplication, run_final_analysis) -- every one of those already
    checks the database before doing expensive work, so re-running `analyze`
    against a folder that's gained a few new papers only processes what's
    new (project notes section 10). No new stage logic lives here, only
    orchestration, validation, and user-facing progress output."""
    config = replace(
        DEFAULT_CONFIG,
        llm_reload_policy=args.reload_policy or "reload_on_failure",
        log_level="DEBUG" if args.debug else DEFAULT_CONFIG.log_level,
    )
    setup_logging(config.log_level)

    if not args.textbook and not args.index:
        logger.error(
            "You must provide a topic source: either --textbook <PDF> (a textbook with its own table of "
            "contents) or --index <PDF/TXT/MD/JSON> (an index, syllabus, or topic list you already have)."
        )
        sys.exit(1)

    question_papers_dir = Path(args.question_papers)
    if not question_papers_dir.is_dir():
        logger.error("Exam-paper folder not found (or not a directory): %s", question_papers_dir)
        sys.exit(1)
    pdf_paths = sorted(question_papers_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.error(
            "No PDF files found in %s -- add exam-paper PDFs to this folder and try again.", question_papers_dir
        )
        sys.exit(1)

    config = _resolve_llm_config(args, config)

    print("Exam Paper Analyzer")
    print("====================")

    db = Database(config.database_path)
    reset_call_metrics()

    try:
        if args.textbook:
            authority = resolve_textbook_topic_authority(Path(args.textbook), config, db)
        else:
            authority = resolve_index_topic_authority(Path(args.index), config, db)
    except TopicAuthorityError as exc:
        logger.error(str(exc))
        db.close()
        sys.exit(1)

    cache_note = " (reused from a previous run)" if authority.from_cache else ""
    _ok(f"Topic authority loaded: {len(authority.topics)} topics from {Path(authority.source_path).name}{cache_note}")
    if authority.validation is not None and not authority.validation.passed:
        logger.warning(
            "The topic list has %d issue(s) worth reviewing (see topic_extraction_runs / topics.json) -- "
            "continuing anyway, since these are advisory, not fatal.",
            sum(1 for i in authority.validation.issues if i.severity == "error"),
        )
    _ok(f"{len(pdf_paths)} exam paper(s) found in {question_papers_dir}")

    file_results, question_counts_by_paper, _ = extract_questions_for_files(pdf_paths, config, db)
    total_questions = sum(question_counts_by_paper.values())
    failed_files = [Path(r.file_path).name for r in file_results if r.status == "failed"]
    _ok(f"Questions extracted: {total_questions} occurrence(s) across {len(pdf_paths)} paper(s)")
    if failed_files:
        print(f"     Note: {len(failed_files)} file(s) could not be processed and were skipped: {', '.join(failed_files)}")

    if not db.get_all_questions():
        logger.error("No questions could be extracted from the supplied exam papers -- nothing to analyze.")
        db.close()
        sys.exit(1)

    file_summaries, questions_by_paper, totals, skipped = classify_papers_for_files(
        pdf_paths, authority.topics, authority.file_hash, config, db
    )
    write_stage3_outputs(
        config.output_directory, Path(authority.source_path), config.model_identifier, authority.topics,
        file_summaries, questions_by_paper,
    )
    _ok(
        f"Questions classified: {totals['classified']} matched to a topic, {totals['no_match']} no confident match, "
        f"{totals['ambiguous']} ambiguous, {totals['unclassified']} not yet classified"
    )
    if skipped:
        logger.warning("Skipped (question extraction did not succeed for these files): %s", ", ".join(skipped))

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
        config.output_directory, run_stats, db.list_canonical_questions(), db.list_dedup_pair_decisions(verdict="uncertain"),
    )
    _ok(f"Repeated questions detected: {dedup_result.canonical_group_count} canonical question group(s)")

    try:
        report = run_final_analysis(db, authority.file_hash)
    except FrequencyAnalysisPrerequisiteError as exc:
        logger.error(str(exc))
        db.close()
        sys.exit(1)
    _ok("Frequency analysis complete")

    json_path, report_path = write_stage6_outputs(config.output_directory, report)
    _ok("Report generated")
    db.close()

    s = report.summary
    print()
    print("SUMMARY")
    print(f"  Papers analyzed: {s.total_papers}")
    print(f"  Question occurrences: {s.total_question_occurrences}")
    print(f"  Canonical (unique) questions: {s.total_canonical_questions}")
    print(f"  Topics: {s.total_topics} ({s.topics_tested} tested, {s.topics_untested} untested)")
    print(f"  Repeated question groups: {s.repeated_canonical_question_count}")
    if s.no_match_count or s.ambiguous_count or s.unclassified_count:
        print(
            f"  Not topic-assigned: no_match={s.no_match_count}, ambiguous={s.ambiguous_count}, "
            f"unclassified={s.unclassified_count}"
        )
    print()
    print("Output:")
    print(f"  {report_path}")
    print(f"  {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exam paper sorter pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_extract = subparsers.add_parser("extract-topics", help="Stage 1: extract textbook topic list")
    p_extract.add_argument("--textbook", required=True, help="Path to any textbook PDF")
    p_extract.add_argument(
        "--provider", default=None, choices=["openrouter", "ollama"],
        help="LLM backend (default: openrouter). Ollama is an optional local dev/testing backend.",
    )
    p_extract.add_argument(
        "--openrouter-api-key", default=None,
        help="OpenRouter API key (default: OPENROUTER_API_KEY environment variable). Required for --provider openrouter.",
    )
    p_extract.add_argument("--model", default=None, help="Ollama model name -- only used with --provider ollama")
    p_extract.add_argument("--max-search-pages", type=int, default=None)
    p_extract.add_argument("--search-window", type=int, default=None)
    p_extract.add_argument("--seed", type=int, default=None, help="Fixed LLM sampling seed (default: config)")
    p_extract.add_argument("--debug", action="store_true", help="Verbose logging incl. raw LLM I/O")
    p_extract.set_defaults(func=cmd_extract_topics)

    p_questions = subparsers.add_parser("extract-questions", help="Stage 2: extract questions from exam-paper PDFs")
    p_questions.add_argument("--question-papers", required=True, help="Directory containing exam-paper PDFs")
    p_questions.add_argument(
        "--provider", default=None, choices=["openrouter", "ollama"],
        help="LLM backend (default: openrouter). Ollama is an optional local dev/testing backend.",
    )
    p_questions.add_argument(
        "--openrouter-api-key", default=None,
        help="OpenRouter API key (default: OPENROUTER_API_KEY environment variable). Required for --provider openrouter.",
    )
    p_questions.add_argument("--model", default=None, help="Ollama model name -- only used with --provider ollama")
    p_questions.add_argument("--seed", type=int, default=None, help="Fixed LLM sampling seed (default: config)")
    p_questions.add_argument(
        "--reload-policy", default=None,
        choices=["persistent", "reload_per_paper", "reload_every_n_calls", "reload_on_failure"],
        help="Ollama model unload/reload policy -- only used with --provider ollama (default: reload_on_failure)",
    )
    p_questions.add_argument("--debug", action="store_true", help="Verbose logging incl. raw LLM I/O")
    p_questions.set_defaults(func=cmd_extract_questions)

    p_classify = subparsers.add_parser(
        "classify-topics", help="Stage 3: classify extracted questions against the textbook's topic list"
    )
    p_classify.add_argument("--textbook", required=True, help="Path to the textbook PDF (must already have run extract-topics)")
    p_classify.add_argument(
        "--question-papers", required=True,
        help="Directory containing exam-paper PDFs (must already have run extract-questions)",
    )
    p_classify.add_argument(
        "--provider", default=None, choices=["openrouter", "ollama"],
        help="LLM backend (default: openrouter). Ollama is an optional local dev/testing backend.",
    )
    p_classify.add_argument(
        "--openrouter-api-key", default=None,
        help="OpenRouter API key (default: OPENROUTER_API_KEY environment variable). Required for --provider openrouter.",
    )
    p_classify.add_argument("--model", default=None, help="Ollama model name -- only used with --provider ollama")
    p_classify.add_argument("--seed", type=int, default=None, help="Fixed LLM sampling seed (default: config)")
    p_classify.add_argument("--debug", action="store_true", help="Verbose logging incl. raw LLM I/O")
    p_classify.set_defaults(func=cmd_classify_topics)

    p_dedup = subparsers.add_parser(
        "deduplicate",
        help="Stage 4: semantic deduplication over already-extracted question occurrences",
    )
    p_dedup.add_argument(
        "--provider", default=None, choices=["openrouter", "ollama"],
        help="LLM backend for semantic-equivalence judging (default: openrouter). Ollama is an optional local dev/testing backend.",
    )
    p_dedup.add_argument(
        "--openrouter-api-key", default=None,
        help="OpenRouter API key (default: OPENROUTER_API_KEY environment variable). Required for --provider openrouter.",
    )
    p_dedup.add_argument("--model", default=None, help="Ollama model name -- only used with --provider ollama")
    p_dedup.add_argument("--seed", type=int, default=None, help="Fixed LLM sampling seed (default: config)")
    p_dedup.add_argument("--debug", action="store_true", help="Verbose logging incl. raw LLM I/O")
    p_dedup.set_defaults(func=cmd_deduplicate)

    p_freq = subparsers.add_parser(
        "analyze-frequency",
        help="Stage 5: deterministic frequency/aggregation analysis over already-classified, deduplicated questions",
    )
    p_freq.add_argument(
        "--textbook", required=True,
        help="Path to the textbook PDF (must already have run extract-topics -- Stage 5 reads its topic hierarchy)",
    )
    p_freq.add_argument("--debug", action="store_true", help="Verbose logging")
    p_freq.set_defaults(func=cmd_analyze_frequency)

    p_report = subparsers.add_parser(
        "generate-report",
        help="Stage 6: final user-facing report over already-computed Stage 1-5 data (zero LLM/embedding calls)",
    )
    p_report.add_argument(
        "--textbook", required=True,
        help="Path to the textbook PDF (must already have run extract-topics, extract-questions, and deduplicate "
             "-- generate-report runs Stage 5's own aggregation itself, so a prior analyze-frequency run is not "
             "required)",
    )
    p_report.add_argument("--debug", action="store_true", help="Verbose logging")
    p_report.set_defaults(func=cmd_generate_report)

    p_analyze = subparsers.add_parser(
        "analyze",
        help="The single end-user command: topic source + exam-paper folder in, final report out "
             "(runs Stages 2-6 internally; no need to know or run individual stages)",
    )
    topic_source = p_analyze.add_mutually_exclusive_group(required=False)  # validated by hand for a clearer message
    topic_source.add_argument("--textbook", default=None, help="Path to a textbook PDF with its own table of contents")
    topic_source.add_argument(
        "--index", default=None,
        help="Path to an index/syllabus/topic list you already have (.pdf, .txt, .md, or .json) -- "
             "skips textbook topic discovery",
    )
    p_analyze.add_argument("--question-papers", required=True, help="Directory containing exam-paper PDFs")
    p_analyze.add_argument(
        "--provider", default=None, choices=["openrouter", "ollama"],
        help="LLM backend (default: openrouter). Ollama is an optional local dev/testing backend.",
    )
    p_analyze.add_argument(
        "--openrouter-api-key", default=None,
        help="OpenRouter API key (default: OPENROUTER_API_KEY environment variable, then a previously-saved key "
             "if any -- see --save-key). Required for --provider openrouter.",
    )
    p_analyze.add_argument(
        "--save-key", action="store_true",
        help="Save the supplied OpenRouter API key to the OS credential store (Windows Credential Manager / "
             "macOS Keychain / Linux Secret Service) so future runs don't need --openrouter-api-key or the "
             "environment variable again. Requires the optional 'keyring' package.",
    )
    p_analyze.add_argument("--model", default=None, help="Ollama model name -- only used with --provider ollama")
    p_analyze.add_argument("--seed", type=int, default=None, help="Fixed LLM sampling seed (default: config)")
    p_analyze.add_argument(
        "--reload-policy", default=None,
        choices=["persistent", "reload_per_paper", "reload_every_n_calls", "reload_on_failure"],
        help="Ollama model unload/reload policy -- only used with --provider ollama (default: reload_on_failure)",
    )
    p_analyze.add_argument("--debug", action="store_true", help="Verbose logging incl. raw LLM I/O")
    p_analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
