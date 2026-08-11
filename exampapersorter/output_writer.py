"""Writes pipeline outputs.

Stage 1: topics.json and topics_evidence.json. Kept as two files rather
than one so `topics.json` stays a clean, small artifact you'd hand to
Stage 2, while `topics_evidence.json` carries everything needed to audit
*why* those topics were extracted (which pages were searched, the TOC
detection verdict, the raw evidence text).

Stage 2: one JSON+Markdown pair per detected paper under
output/question_extraction/, plus a per-file summary JSON. The JSON is the
authoritative record (same shape persisted to SQLite); the Markdown is a
debug/inspection view only -- built specifically to answer the questions a
human reviewer needs to check: were papers split correctly, were
dates/metadata associated correctly (and any conflicts visible), were
sections detected correctly, were individual questions extracted
correctly, were MCQs identified.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from exampapersorter.schemas import (
    CanonicalQuestion,
    DedupPairDecision,
    FinalAnalysisReport,
    FrequencyAnalysisSummary,
    PageRangeEvidence,
    Paper,
    Question,
    QuestionPaperFileResult,
    TOCDetectionResult,
    Topic,
    TopicExtractionResult,
    ValidationReport,
)
from exampapersorter.topic_extraction.search import SearchAttempt


def write_stage1_outputs(
    output_dir: Path,
    textbook_path: Path,
    topics: TopicExtractionResult,
    validation: ValidationReport,
    toc_evidence: PageRangeEvidence,
    toc_verdict: TOCDetectionResult,
    search_attempts: list[SearchAttempt],
    model_used: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    topics_path = output_dir / "topics.json"
    topics_path.write_text(
        json.dumps(
            {
                "textbook_path": str(textbook_path),
                "model_used": model_used,
                "topics": [t.model_dump() for t in topics.topics],
                "validation": validation.model_dump(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    evidence_path = output_dir / "topics_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "textbook_path": str(textbook_path),
                "model_used": model_used,
                "search_attempts": [a.model_dump() for a in search_attempts],
                "accepted_page_range": {
                    "start_page": toc_evidence.start_page,
                    "end_page": toc_evidence.end_page,
                },
                "toc_verdict": toc_verdict.model_dump(),
                "evidence_blocks": [b.model_dump() for b in toc_evidence.blocks],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return topics_path, evidence_path


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def _field_dict(field) -> dict:
    return {
        "document": field.document_value,
        "document_evidence": field.document_evidence,
        "document_confidence": field.document_confidence,
        "filename": field.filename_value,
        "conflict": field.conflict,
    }


def _paper_json(paper: Paper, questions: list[Question], source_pdf: str) -> dict:
    return {
        "paper_id": paper.paper_id,
        "source_pdf": source_pdf,
        "page_range": [paper.start_page, paper.end_page],
        "status": paper.status,
        "notes": paper.notes,
        "boundary_confidence": paper.boundary_confidence,
        "boundary_evidence": [e.model_dump() for e in paper.boundary_evidence],
        "metadata": paper.metadata.model_dump(),
        "sections": [s.model_dump() for s in paper.sections],
        "questions": [q.model_dump() for q in questions],
    }


def _paper_markdown(paper: Paper, questions: list[Question], source_pdf: str) -> str:
    lines = [
        f"# {paper.paper_id}",
        "",
        f"Source: `{source_pdf}`, PDF pages {paper.start_page}-{paper.end_page} "
        f"(boundary confidence {paper.boundary_confidence:.2f})",
        f"Status: **{paper.status}**" + (f" -- {'; '.join(paper.notes)}" if paper.notes else ""),
        "",
        "## Metadata",
        "",
        "| Field | Document | Confidence | Grounding evidence | Filename | Conflict |",
        "|---|---|---|---|---|---|",
    ]
    m = paper.metadata
    for label, field in (
        ("exam_name", m.exam_name), ("institution", m.institution), ("subject", m.subject),
        ("date", m.date), ("year", m.year), ("paper_identifier", m.paper_identifier),
    ):
        d = _field_dict(field)
        conflict_marker = "**CONFLICT**" if d["conflict"] else ""
        confidence = f"{d['document_confidence']:.2f}" if d["document"] else ""
        lines.append(
            f"| {label} | {d['document'] or ''} | {confidence} | {d['document_evidence'] or ''} | "
            f"{d['filename'] or ''} | {conflict_marker} |"
        )
    if m.other_identifiers_document:
        lines.append("")
        lines.append(f"Other identifiers (document): {', '.join(m.other_identifiers_document)}")
    if m.other_identifiers_filename:
        lines.append(f"Other identifiers (filename tokens): {', '.join(m.other_identifiers_filename)}")

    lines.append("")
    lines.append("## Sections")
    lines.append("")
    if not paper.sections:
        lines.append("(none detected)")
    for s in paper.sections:
        lines.append(f"- `{s.section_id}` **{s.section_type}** -- {s.raw_label or '(no heading text)'} "
                      f"(pages {s.start_page}-{s.end_page}, confidence {s.confidence:.2f})")

    lines.append("")
    lines.append(f"## Questions ({len(questions)})")
    lines.append("")
    by_section: dict[str | None, list[Question]] = {}
    for q in questions:
        by_section.setdefault(q.section_id, []).append(q)

    section_order = [s.section_id for s in paper.sections] + [
        sid for sid in by_section if sid not in {s.section_id for s in paper.sections}
    ]
    for section_id in section_order:
        qs = by_section.get(section_id)
        if not qs:
            continue
        header = section_id or "(unassigned to a section)"
        lines.append(f"### {header}")
        lines.append("")
        for q in qs:
            flags = []
            if q.correction_applied:
                flags.append("OCR-corrected")
            if q.ambiguous:
                flags.append("AMBIGUOUS")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            num = q.question_number or "?"
            lines.append(f"**{num}.** ({q.question_type}, confidence {q.extraction_confidence:.2f}){flag_str}")
            lines.append(f"> {q.question_text}")
            for i, opt in enumerate(q.options):
                lines.append(f"> {chr(ord('A') + i)}. {opt}")
            if q.correction_applied and q.original_text:
                lines.append(f">\n> _original OCR: {q.original_text}_")
            if q.ambiguous and q.ambiguity_note:
                lines.append(f">\n> _ambiguity: {q.ambiguity_note}_")
            lines.append(f"> _source pages: {q.source_pages}_")
            lines.append("")

    return "\n".join(lines)


def write_stage2_outputs(
    output_dir: Path,
    file_result: QuestionPaperFileResult,
    questions_by_paper: dict[str, list[Question]],
) -> list[Path]:
    out_dir = output_dir / "question_extraction"
    out_dir.mkdir(parents=True, exist_ok=True)

    source_pdf = Path(file_result.file_path).name
    stem = _slug(Path(file_result.file_path).stem)
    written: list[Path] = []

    for paper in file_result.papers:
        questions = questions_by_paper.get(paper.paper_id, [])
        idx = paper.paper_id.split("_paper")[-1]
        base = out_dir / f"{stem}_paper{idx}"

        json_path = base.with_suffix(".json")
        json_path.write_text(json.dumps(_paper_json(paper, questions, source_pdf), indent=2), encoding="utf-8")
        written.append(json_path)

        md_path = base.with_suffix(".md")
        md_path.write_text(_paper_markdown(paper, questions, source_pdf), encoding="utf-8")
        written.append(md_path)

    summary_path = out_dir / f"{stem}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source_pdf": source_pdf,
                "file_hash": file_result.file_hash,
                "total_pages": file_result.total_pages,
                "model_used": file_result.model_used,
                "status": file_result.status,
                "papers_detected": len(file_result.papers),
                "boundary_issues": [i.model_dump() for i in file_result.boundary_issues],
                "papers": [
                    {
                        "paper_id": p.paper_id,
                        "page_range": [p.start_page, p.end_page],
                        "status": p.status,
                        "section_count": len(p.sections),
                        "question_count": len(questions_by_paper.get(p.paper_id, [])),
                    }
                    for p in file_result.papers
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    written.append(summary_path)

    return written


# --- Stage 3: topic classification ---
#
# Deliberately does NOT rewrite Stage 2's own output/question_extraction
# files -- those are an already-produced, human-reviewed artifact set (see
# project notes: Stage 2's 215 questions across 9 papers are treated as
# approved), and silently mutating them as a side effect of running Stage 3
# would be a regression risk Stage 3 has no need to take. The DB's
# `questions` rows are the live source of truth for classification;
# these files are a separate, additive report over that same data.


def write_stage3_outputs(
    output_dir: Path,
    textbook_path: Path,
    model_used: str,
    topics: list[Topic],
    file_summaries: list[dict],
    questions_by_paper: dict[str, list[Question]],
) -> tuple[Path, Path]:
    """file_summaries: one dict per processed question-paper PDF, each
    shaped {"source_pdf", "file_hash", "papers": [{"paper_id", "status",
    "total_questions", "classified_count", "no_match_count",
    "ambiguous_count", "unclassified_count", "error_message"}, ...]} --
    built by cli.py from topic_classification.pipeline.PaperClassificationResult,
    kept as plain dicts here so this module has no dependency on that
    pipeline module (mirrors write_stage1/2_outputs, which likewise only
    depend on schemas.py + their own stage's search/pipeline result types
    they're actually handed)."""
    out_dir = output_dir / "topic_classification"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_questions = [q for qs in questions_by_paper.values() for q in qs]
    totals = {"classified": 0, "no_match": 0, "ambiguous": 0, "unclassified": 0}
    for q in all_questions:
        totals[q.classification_status] = totals.get(q.classification_status, 0) + 1

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "textbook_path": str(textbook_path),
                "model_used": model_used,
                "topic_count": len(topics),
                "total_questions": len(all_questions),
                "totals_by_status": totals,
                "files": file_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    by_topic: dict[str, list[Question]] = {}
    no_match: list[Question] = []
    ambiguous: list[Question] = []
    unclassified: list[Question] = []
    for q in all_questions:
        if q.classification_status == "classified" and q.topic_id:
            by_topic.setdefault(q.topic_id, []).append(q)
        elif q.classification_status == "no_match":
            no_match.append(q)
        elif q.classification_status == "ambiguous":
            ambiguous.append(q)
        else:
            unclassified.append(q)

    lines = [
        "# Questions by topic",
        "",
        f"Textbook: `{textbook_path.name}` -- {len(topics)} topics, {len(all_questions)} questions classified "
        f"(model: {model_used})",
        "",
    ]

    def render_question_line(q: Question) -> str:
        num = q.question_number or "?"
        confidence = f"{q.topic_confidence:.2f}" if q.topic_confidence is not None else "n/a"
        return f"- **{num}.** ({q.paper_id}, confidence {confidence}) {q.question_text}"

    def walk(parent_id: str | None, depth: int) -> None:
        for t in [t for t in topics if t.parent_id == parent_id]:
            questions = by_topic.get(t.id, [])
            lines.append(f"{'#' * min(depth + 2, 6)} [{t.id}] {t.name} ({len(questions)})")
            lines.append("")
            for q in questions:
                lines.append(render_question_line(q))
            if questions:
                lines.append("")
            walk(t.id, depth + 1)

    walk(None, 0)

    for label, bucket in (("No confident topic match", no_match), ("Ambiguous", ambiguous), ("Unclassified / needs review", unclassified)):
        lines.append(f"## {label} ({len(bucket)})")
        lines.append("")
        if not bucket:
            lines.append("(none)")
            lines.append("")
            continue
        for q in bucket:
            lines.append(render_question_line(q))
        lines.append("")

    by_topic_path = out_dir / "by_topic.md"
    by_topic_path.write_text("\n".join(lines), encoding="utf-8")

    return summary_path, by_topic_path


# --- Stage 4: semantic deduplication ---
#
# Deliberately does NOT touch output/question_extraction or
# output/topic_classification -- Stage 4 output is purely additive under
# output/deduplication/. The `questions` DB rows and Stage 2/3 output files
# remain the source of truth for each raw occurrence; these files are a
# separate report over the canonical groups Stage 4 derived from them.


def write_stage4_outputs(
    output_dir: Path,
    run_stats: dict,
    canonical_questions: list[CanonicalQuestion],
    needs_review_pairs: list[DedupPairDecision],
) -> tuple[Path, Path, Path]:
    """`run_stats`: plain dict built by cli.py from
    deduplication.pipeline.DeduplicationResult (kept as a dict here rather
    than importing that dataclass, so this module has no dependency on the
    pipeline module -- mirrors write_stage3_outputs' file_summaries
    convention). `canonical_questions`: DB-hydrated records (occurrence_
    count/unique_paper_count/years/source_question_ids populated), e.g.
    from Database.list_canonical_questions(). `needs_review_pairs`: every
    verdict="uncertain" DedupPairDecision, listed independently of which
    canonical group (if any) each pair's questions ended up in -- a
    same_question verdict downgraded below the confidence floor also
    surfaces here (see deduplication/reconciliation.py)."""
    out_dir = output_dir / "deduplication"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(run_stats, indent=2), encoding="utf-8")

    canonical_path = out_dir / "canonical_questions.json"
    canonical_path.write_text(
        json.dumps([cq.model_dump() for cq in canonical_questions], indent=2), encoding="utf-8"
    )

    duplicate_groups = [cq.model_dump() for cq in canonical_questions if cq.occurrence_count > 1]
    duplicate_groups_path = out_dir / "duplicate_groups.json"
    duplicate_groups_path.write_text(
        json.dumps(
            {
                "duplicate_groups": duplicate_groups,
                "needs_review_pairs": [p.model_dump() for p in needs_review_pairs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return summary_path, canonical_path, duplicate_groups_path


# --- Stage 5: frequency analysis / aggregation ---
#
# Purely additive under output/frequency_analysis/ -- does not touch any
# Stage 1-4 output. summary.json is the authoritative, structured artifact
# (same shape as the FrequencyAnalysisSummary this module is handed);
# report.md is a human-readable rendering of that same data, not a second
# source of truth (project notes section 26).

_TYPE_LABELS: tuple[tuple[str, str], ...] = (
    ("mcq", "MCQ"),
    ("short_answer", "Short Answer"),
    ("short_essay", "Short Essay"),
    ("long_answer", "Long Answer"),
    ("essay", "Essay"),
    ("other", "Other"),
    ("unknown", "Unknown"),
)


def _type_counts_line(type_counts) -> str:
    return " | ".join(f"{label}: {getattr(type_counts, key)}" for key, label in _TYPE_LABELS)


def _render_frequency_report(summary: FrequencyAnalysisSummary) -> str:
    lines = ["# Exam Question Analysis", "", "## Overall Summary", ""]
    lines.append(f"- Total question occurrences: {summary.total_question_occurrences}")
    lines.append(f"- Total canonical (unique) questions: {summary.total_canonical_questions}")
    lines.append(f"- Total papers: {summary.total_papers}")
    lines.append(f"- Topics in textbook: {summary.topic_count}")
    lines.append(f"- Years represented: {', '.join(summary.years_represented) or '(none)'}")
    if summary.year_conflict_paper_ids:
        lines.append(
            f"- Papers with conflicting year metadata (see Stage 2 audit): {', '.join(summary.year_conflict_paper_ids)}"
        )
    integrity = summary.data_integrity
    lines.append(
        f"- Data integrity: {'OK' if integrity.reconciled else 'ISSUES FOUND'} "
        f"({integrity.total_raw_occurrences} raw occurrences -> {integrity.total_canonical_links} canonical links, "
        f"{integrity.distinct_canonical_question_ids_in_links} distinct canonical questions)"
    )
    if integrity.orphaned_occurrences:
        lines.append(f"  - orphaned occurrences: {', '.join(integrity.orphaned_occurrences)}")
    if integrity.duplicate_links:
        lines.append(f"  - duplicate links: {', '.join(integrity.duplicate_links)}")
    status = summary.classification_status_distribution
    lines.append(
        f"- Classification status: classified={status.get('classified', 0)}, "
        f"no_match={status.get('no_match', 0)}, ambiguous={status.get('ambiguous', 0)}, "
        f"unclassified={status.get('unclassified', 0)}"
    )
    lines.append("")

    lines.append("## Question Type Distribution")
    lines.append("")
    total = summary.total_question_occurrences
    tc = summary.question_type_distribution
    for key, label in _TYPE_LABELS:
        value = getattr(tc, key)
        pct = f"{(value / total * 100):.1f}%" if total else "0.0%"
        lines.append(f"- {label}: {value} ({pct})")
    lines.append("")

    lines.append("## Repetition by Question Type")
    lines.append("")
    for key, label in _TYPE_LABELS:
        stats = summary.repetition_by_question_type.get(key)
        if stats is None:
            continue
        lines.append(
            f"- {label}: {stats.total_occurrences} occurrences, {stats.unique_canonical_questions} unique, "
            f"{stats.repeated_canonical_questions} repeated"
        )
    lines.append("")

    lines.append("## Most Repeated Questions")
    lines.append("")
    if not summary.most_repeated_questions:
        lines.append("(none)")
    for r in summary.most_repeated_questions:
        lines.append(f"{r.rank}. **{r.canonical_question_text}**")
        lines.append(
            f"   - Occurrences: {r.occurrence_count} | Unique papers: {r.unique_paper_count} | "
            f"Types: {', '.join(r.question_types)}"
        )
        lines.append(f"   - Years: {', '.join(r.years) or '(unknown)'} | Topic: {r.topic_name or '(unassigned)'}")
    lines.append("")

    lines.append("## Most Tested Chapters")
    lines.append("")
    if not summary.most_tested_topics:
        lines.append("(none)")
    for r in summary.most_tested_topics:
        lines.append(f"{r.rank}. **{r.topic_name}** -- {r.total_occurrences} occurrences, {r.unique_canonical_questions} unique questions")
    lines.append("")

    lines.append("## Chapters")
    lines.append("")
    by_parent: dict[str | None, list] = {}
    for s in summary.topic_summaries:
        by_parent.setdefault(s.parent_id, []).append(s)
    for children in by_parent.values():
        children.sort(key=lambda s: s.topic_id)

    def walk(parent_id: str | None, depth: int) -> None:
        for s in by_parent.get(parent_id, []):
            header_level = min(depth + 2, 6)
            lines.append(
                f"{'#' * header_level} {s.topic_name} ({s.total_occurrences} occurrences, "
                f"{s.unique_canonical_questions} unique)"
            )
            lines.append("")
            if s.total_occurrences == 0:
                lines.append("(no questions tested)")
                lines.append("")
            else:
                lines.append(_type_counts_line(s.type_counts))
                lines.append(f"Papers represented: {s.paper_count} | Years: {', '.join(s.years) or '(unknown)'}")
                lines.append("")
                if s.repeated_canonical_questions:
                    lines.append("Repeated questions:")
                    for rq in s.repeated_canonical_questions:
                        lines.append(
                            f"- {rq.canonical_question_text} -- repeated {rq.occurrences_in_topic}x "
                            f"(years: {', '.join(rq.years) or '(unknown)'})"
                        )
                    lines.append("")
            walk(s.topic_id, depth + 1)

    walk(None, 0)

    for label, ids in (
        ("No Confident Topic Match", summary.no_match_questions),
        ("Ambiguous Topic Classification", summary.ambiguous_questions),
        ("Unclassified", summary.unclassified_questions),
    ):
        lines.append(f"## {label} ({len(ids)})")
        lines.append("")
        if not ids:
            lines.append("(none)")
        else:
            for qid in ids:
                lines.append(f"- {qid}")
        lines.append("")

    return "\n".join(lines)


def write_stage5_outputs(output_dir: Path, summary: FrequencyAnalysisSummary) -> tuple[Path, Path]:
    out_dir = output_dir / "frequency_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary.model_dump(), indent=2), encoding="utf-8")

    report_path = out_dir / "report.md"
    report_path.write_text(_render_frequency_report(summary), encoding="utf-8")

    return summary_path, report_path


# --- Stage 6: final user-facing report ---
#
# Purely additive under output/final_analysis/ -- does not touch any Stage
# 1-5 output. analysis.json is the authoritative, structured artifact (same
# shape as the FinalAnalysisReport this module is handed, GUI-ready);
# report.md is a human-readable rendering of that same data, not a second
# source of truth (mirrors write_stage5_outputs' own summary.json/report.md
# split).


def _render_final_analysis_report(report: FinalAnalysisReport) -> str:
    s = report.summary
    lines = ["# Exam Paper Analysis", "", "## Executive Summary", ""]
    lines.append(f"- Exam papers analyzed: {s.total_papers}")
    lines.append(f"- Question occurrences extracted: {s.total_question_occurrences}")
    lines.append(f"- Canonical (unique) questions: {s.total_canonical_questions}")
    lines.append(f"- Textbook topics: {s.total_topics} ({s.topics_tested} tested, {s.topics_untested} untested)")
    lines.append(f"- Repeated canonical question groups: {s.repeated_canonical_question_count}")
    lines.append(f"- Years represented: {', '.join(s.years_represented) or '(none)'}")
    if s.no_match_count or s.ambiguous_count or s.unclassified_count:
        lines.append(
            f"- Not attributed to a topic: no_match={s.no_match_count}, ambiguous={s.ambiguous_count}, "
            f"unclassified={s.unclassified_count}"
        )
    lines.append(
        f"- Data integrity: {'OK' if s.data_integrity_reconciled else 'ISSUES FOUND -- see Data Quality / Notes'}"
    )
    lines.append("")
    lines.append(
        "The sections below show the most frequently repeated questions and most tested chapters in the analyzed "
        "papers -- not a prediction of what will appear on a future exam."
    )
    lines.append("")

    lines.append("## Most Repeated Questions")
    lines.append("")
    if not report.most_repeated_questions:
        lines.append("(none -- no canonical question occurred more than once)")
    for r in report.most_repeated_questions:
        lines.append(f"### {r.rank}. {r.canonical_question_text}")
        lines.append("")
        lines.append(f"- Occurrences: {r.occurrence_count} | Unique papers: {r.unique_paper_count}")
        lines.append(f"- Years: {', '.join(r.years) or '(unknown)'}")
        lines.append(f"- Type(s): {', '.join(r.question_types)}")
        lines.append(f"- Topic: {r.topic_name or '(unassigned)'}")
        confidence_str = f" (confidence {r.dedup_confidence:.2f})" if r.dedup_confidence is not None else ""
        lines.append(f"- Deduplication: {r.dedup_status}{confidence_str}")
        lines.append(f"- Source papers: {', '.join(r.source_filenames) or '(unknown)'}")
        lines.append("")

    lines.append("## Most Tested Chapters")
    lines.append("")
    lines.append("### By total question occurrences")
    lines.append("")
    if not report.most_tested_topics_by_occurrences:
        lines.append("(none)")
    for r in report.most_tested_topics_by_occurrences:
        lines.append(
            f"{r.rank}. **{r.topic_name}** -- {r.total_occurrences} occurrences "
            f"({r.unique_canonical_questions} unique questions)"
        )
    lines.append("")
    lines.append("### By unique canonical questions")
    lines.append("")
    if not report.most_tested_topics_by_unique_questions:
        lines.append("(none)")
    for r in report.most_tested_topics_by_unique_questions:
        lines.append(
            f"{r.rank}. **{r.topic_name}** -- {r.unique_canonical_questions} unique questions "
            f"({r.total_occurrences} occurrences)"
        )
    lines.append("")

    lines.append("## Question Type Distribution")
    lines.append("")
    lines.append(f"{report.question_type_distribution.total_occurrences} total question occurrences:")
    lines.append("")
    for share in report.question_type_distribution.by_type:
        lines.append(f"- {share.label}: {share.occurrence_count} ({share.percentage:.1f}%)")
    lines.append("")
    lines.append("(Percentages are of question occurrences, not unique canonical questions.)")
    lines.append("")

    lines.append("## Analysis by Chapter")
    lines.append("")
    if not report.topic_analysis:
        lines.append("(no topics had classified questions)")
        lines.append("")
    by_parent: dict[str | None, list] = {}
    for t in report.topic_analysis:
        by_parent.setdefault(t.parent_id, []).append(t)
    for children in by_parent.values():
        children.sort(key=lambda t: t.topic_id)

    def walk(parent_id: str | None, depth: int) -> None:
        for t in by_parent.get(parent_id, []):
            header_level = min(depth + 3, 6)
            lines.append(f"{'#' * header_level} {t.topic_name}")
            lines.append("")
            if t.total_occurrences == 0:
                # A structural Section/Group header kept in the tree only
                # because a descendant leaf has questions (see build.
                # split_topic_analysis) -- no stats of its own to show.
                lines.append("(no questions classified directly into this chapter -- see subtopics below)")
                lines.append("")
            else:
                lines.append(
                    f"- Occurrences: {t.total_occurrences} | Unique canonical questions: "
                    f"{t.unique_canonical_questions} | Repeated: {len(t.repeated_canonical_questions)}"
                )
                lines.append(f"- Papers represented: {t.paper_count} | Years: {', '.join(t.years) or '(unknown)'}")
                lines.append("")
                lines.append(_type_counts_line(t.type_counts))
                lines.append("")
                if t.repeated_canonical_questions:
                    lines.append("Repeated questions:")
                    for rq in t.repeated_canonical_questions:
                        lines.append(
                            f"- {rq.canonical_question_text} -- {rq.occurrences_in_topic}x "
                            f"(years: {', '.join(rq.years) or '(unknown)'})"
                        )
                    lines.append("")
            walk(t.topic_id, depth + 1)

    walk(None, 0)

    lines.append("## Untested Topics")
    lines.append("")
    if not report.untested_topics:
        lines.append("(none -- every textbook topic had at least one classified question)")
    else:
        lines.append(
            f"{len(report.untested_topics)} textbook topics had no classified questions in the analyzed papers "
            "(this does not mean topic extraction failed to find them):"
        )
        lines.append("")
        for u in report.untested_topics:
            lines.append(f"- {u.topic_name}")
    lines.append("")

    for label, bucket in (
        ("Questions With No Confident Topic Match", report.no_match_questions),
        ("Ambiguous Classifications", report.ambiguous_questions),
    ):
        lines.append(f"## {label} ({len(bucket)})")
        lines.append("")
        if not bucket:
            lines.append("(none)")
        else:
            for item in bucket:
                lines.append(f"- **{item.question_text}**")
                lines.append(
                    f"  - Type: {item.question_type} | Paper: {item.source_filename} | "
                    f"Year: {item.paper_year or '(unknown)'}"
                )
        lines.append("")

    dq = report.data_quality
    lines.append("## Data Quality / Notes")
    lines.append("")
    lines.append(f"- Questions with no confident topic match: {dq.no_match_count}")
    lines.append(f"- Questions with ambiguous topic classification: {dq.ambiguous_count}")
    lines.append(f"- Questions not yet classified: {dq.unclassified_count}")
    lines.append(f"- Papers with conflicting year metadata: {dq.year_conflict_paper_count}")
    lines.append(f"- Data integrity check: {'OK' if dq.data_integrity_reconciled else 'ISSUES FOUND'}")
    lines.append("")
    for note in dq.notes:
        lines.append(f"- {note}")
    lines.append("")

    return "\n".join(lines)


def write_stage6_outputs(output_dir: Path, report: FinalAnalysisReport) -> tuple[Path, Path]:
    out_dir = output_dir / "final_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "analysis.json"
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")

    report_path = out_dir / "report.md"
    report_path.write_text(_render_final_analysis_report(report), encoding="utf-8")

    return json_path, report_path
