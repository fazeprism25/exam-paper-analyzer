"""Pure, deterministic functions that transform an already-computed Stage 5
FrequencyAnalysisSummary (plus a small amount of read-only Stage 2/4
hydration) into Stage 6's user-facing FinalAnalysisReport. No database
access, no LLM calls, no embedding calls -- kept pure specifically so this
module is testable with synthetic in-memory data (see
tests/test_final_analysis_build.py), mirroring frequency_analysis/
aggregate.py's own design. final_analysis/pipeline.py is the only caller
that touches a Database.

Every number here is either copied straight from FrequencyAnalysisSummary,
or a different SORT/GROUPING of data Stage 5 already computed (e.g. ranking
topics by unique_canonical_questions instead of total_occurrences) -- this
module introduces no new count, classification, or deduplication decision
of its own.
"""
from __future__ import annotations

from exampapersorter.frequency_analysis.aggregate import build_canonical_frequency
from exampapersorter.schemas import (
    CanonicalQuestion,
    DataQualityNotes,
    ExecutiveSummary,
    FinalAnalysisReport,
    FrequencyAnalysisSummary,
    Question,
    QuestionTypeCounts,
    QuestionTypeDistributionReport,
    QuestionTypeShare,
    RankedCanonicalQuestion,
    RankedTopic,
    RepeatedQuestionReportItem,
    TopicFrequencySummary,
    TopicRankingItem,
    UnclassifiedQuestionReportItem,
    UntestedTopicItem,
)

# Same key/label pairing as output_writer._TYPE_LABELS (Stage 5's Markdown
# renderer) -- kept as an independent copy rather than importing from
# output_writer, since output_writer imports schemas/this module's outputs,
# not the other way around.
_TYPE_LABELS: tuple[tuple[str, str], ...] = (
    ("mcq", "MCQ"),
    ("short_answer", "Short Answer"),
    ("short_essay", "Short Essay"),
    ("long_answer", "Long Answer"),
    ("essay", "Essay"),
    ("other", "Other"),
    ("unknown", "Unknown"),
)

# Static, honest caveats about known Stage 2-4 limitations -- fixed text,
# not derived from the data (project notes section 17/27). Deliberately
# avoids naming internal implementation details (union-find, embedding
# thresholds, SQLite tables, Pydantic).
_DATA_QUALITY_NOTES: tuple[str, ...] = (
    "Questions with visible OCR corruption are preserved as extracted, not silently cleaned up.",
    "A canonical question represents a group of semantically similar occurrences, not necessarily identical wording.",
    "Deduplication merges are conservative and confidence-based; borderline pairs are left ungrouped rather than merged.",
    "Year metadata is shown only where the source paper carried it; missing years are not inferred.",
    "Some questions could not be confidently matched to a textbook topic, or matched more than one topic ambiguously "
    "-- see the dedicated sections for those questions.",
)


def build_executive_summary(summary: FrequencyAnalysisSummary) -> ExecutiveSummary:
    tested_topics = sum(1 for s in summary.topic_summaries if s.total_occurrences > 0)
    return ExecutiveSummary(
        textbook_file_hash=summary.textbook_file_hash,
        total_papers=summary.total_papers,
        total_question_occurrences=summary.total_question_occurrences,
        total_canonical_questions=summary.total_canonical_questions,
        total_topics=summary.topic_count,
        topics_tested=tested_topics,
        topics_untested=summary.topic_count - tested_topics,
        repeated_canonical_question_count=len(summary.most_repeated_questions),
        years_represented=summary.years_represented,
        no_match_count=len(summary.no_match_questions),
        ambiguous_count=len(summary.ambiguous_questions),
        unclassified_count=len(summary.unclassified_questions),
        data_integrity_reconciled=summary.data_integrity.reconciled,
    )


def build_question_type_distribution(
    type_counts: QuestionTypeCounts, total_occurrences: int
) -> QuestionTypeDistributionReport:
    shares = []
    for key, label in _TYPE_LABELS:
        count = getattr(type_counts, key)
        pct = round(count / total_occurrences * 100, 1) if total_occurrences else 0.0
        shares.append(QuestionTypeShare(question_type=key, label=label, occurrence_count=count, percentage=pct))
    return QuestionTypeDistributionReport(total_occurrences=total_occurrences, by_type=shares)


def topics_by_occurrences(ranked_topics: list[RankedTopic]) -> list[TopicRankingItem]:
    """Reshapes Stage 5's already-ranked most_tested_topics (occurrence_count
    desc, see aggregate.rank_topics) -- no re-sorting, no new computation."""
    return [
        TopicRankingItem(
            rank=r.rank,
            topic_id=r.topic_id,
            topic_name=r.topic_name,
            total_occurrences=r.total_occurrences,
            unique_canonical_questions=r.unique_canonical_questions,
        )
        for r in ranked_topics
    ]


def rank_topics_by_unique_questions(topic_summaries: list[TopicFrequencySummary]) -> list[TopicRankingItem]:
    """A second ranking over the SAME already-computed TopicFrequencySummary
    rows Stage 5 produced, sorted by unique_canonical_questions instead of
    total_occurrences -- deliberately kept as a separate list from
    topics_by_occurrences rather than blended into one ranking, since the
    two orders can disagree (project notes section 10)."""
    tested = [s for s in topic_summaries if s.total_occurrences > 0]
    ordered = sorted(tested, key=lambda s: (-s.unique_canonical_questions, -s.total_occurrences, s.topic_id))
    return [
        TopicRankingItem(
            rank=i + 1,
            topic_id=s.topic_id,
            topic_name=s.topic_name,
            total_occurrences=s.total_occurrences,
            unique_canonical_questions=s.unique_canonical_questions,
        )
        for i, s in enumerate(ordered)
    ]


def split_topic_analysis(
    topic_summaries: list[TopicFrequencySummary],
) -> tuple[list[TopicFrequencySummary], list[UntestedTopicItem]]:
    """Splits Stage 5's complete (including zero-question) topic list into
    the Analysis-by-Chapter tree and the flat Untested Topics list (project
    notes section 12).

    A topic with zero DIRECT occurrences of its own (total_occurrences==0,
    still reported honestly as such) is nonetheless kept in the tree if it
    has at least one tested descendant. This matters in practice, not just
    in theory: confirmed directly against the real fixture's topic
    hierarchy that every one of its 28 tested topics is a level-3 leaf --
    every level-1/level-2 Section/Group node has total_occurrences==0 of
    its own, purely because Stage 3 only ever classifies a question onto a
    leaf topic_id. Using total_occurrences==0 alone to decide tree
    membership would silently exclude EVERY Section/Group header, which
    breaks the tree walk for every leaf beneath it (project notes section
    11: "preserve the textbook hierarchy, do not flatten it"). Untested
    Topics is reserved for a topic whose entire subtree -- itself and every
    descendant -- has zero occurrences (confirmed against the real fixture:
    2 of its 16 Section/Group nodes qualify; the other 14 have at least one
    tested descendant and belong in the tree instead).

    This is a presentation-only membership decision made by walking
    already-computed parent_id links -- it does not change
    ExecutiveSummary.topics_tested/topics_untested, which stay tied to
    Stage 5's own total_occurrences>0 definition (the authoritative
    28-tested/36-untested-style count) precisely so that number isn't
    silently redefined here."""
    by_id = {s.topic_id: s for s in topic_summaries}
    children_by_parent: dict[str | None, list[str]] = {}
    for s in topic_summaries:
        children_by_parent.setdefault(s.parent_id, []).append(s.topic_id)

    memo: dict[str, bool] = {}

    def subtree_has_questions(topic_id: str) -> bool:
        if topic_id in memo:
            return memo[topic_id]
        memo[topic_id] = False  # guard against a malformed cyclic parent chain
        result = by_id[topic_id].total_occurrences > 0 or any(
            subtree_has_questions(child_id) for child_id in children_by_parent.get(topic_id, [])
        )
        memo[topic_id] = result
        return result

    tested = [s for s in topic_summaries if subtree_has_questions(s.topic_id)]
    untested = [
        UntestedTopicItem(topic_id=s.topic_id, topic_name=s.topic_name, level=s.level, parent_id=s.parent_id)
        for s in topic_summaries
        if not subtree_has_questions(s.topic_id)
    ]
    return tested, untested


def build_repeated_questions(
    ranked_canonical: list[RankedCanonicalQuestion],
    canonical_questions: list[CanonicalQuestion],
    questions_by_id: dict[str, Question],
) -> list[RepeatedQuestionReportItem]:
    """Hydrates Stage 5's already-ranked most_repeated_questions with the
    paper-level provenance and dedup status/confidence that Stage 4's own
    CanonicalQuestion record carries but RankedCanonicalQuestion doesn't --
    via the SAME build_canonical_frequency helper frequency_analysis/
    aggregate.py itself uses (reuse, not a second implementation)."""
    canonical_by_id = {cq.canonical_question_id: cq for cq in canonical_questions}
    items: list[RepeatedQuestionReportItem] = []
    for ranked in ranked_canonical:
        cq = canonical_by_id.get(ranked.canonical_question_id)
        if cq is None:
            # Should not happen when data_integrity.reconciled is True --
            # surfaced with empty provenance rather than silently dropped.
            items.append(
                RepeatedQuestionReportItem(
                    rank=ranked.rank,
                    canonical_question_id=ranked.canonical_question_id,
                    canonical_question_text=ranked.canonical_question_text,
                    occurrence_count=ranked.occurrence_count,
                    unique_paper_count=ranked.unique_paper_count,
                    years=ranked.years,
                    question_types=ranked.question_types,
                    topic_id=ranked.topic_id,
                    topic_name=ranked.topic_name,
                    dedup_status="unknown",
                    dedup_confidence=None,
                    paper_ids=[],
                    source_filenames=[],
                    source_question_ids=[],
                )
            )
            continue
        freq = build_canonical_frequency(cq, questions_by_id)
        items.append(
            RepeatedQuestionReportItem(
                rank=ranked.rank,
                canonical_question_id=freq.canonical_question_id,
                canonical_question_text=freq.canonical_question_text,
                occurrence_count=freq.occurrence_count,
                unique_paper_count=freq.unique_paper_count,
                years=freq.years,
                question_types=freq.question_types,
                topic_id=freq.topic_id,
                topic_name=freq.topic_name,
                dedup_status=freq.dedup_status,
                dedup_confidence=freq.dedup_confidence,
                paper_ids=freq.paper_ids,
                source_filenames=freq.source_filenames,
                source_question_ids=freq.source_question_ids,
            )
        )
    return items


def hydrate_unclassified(
    question_ids: list[str], questions_by_id: dict[str, Question]
) -> list[UnclassifiedQuestionReportItem]:
    """Full-text hydration of no_match/ambiguous question_id lists (Stage 5
    only carries the bare IDs) -- straight lookups, no new judgment."""
    items = []
    for qid in question_ids:
        q = questions_by_id.get(qid)
        if q is None:
            continue
        items.append(
            UnclassifiedQuestionReportItem(
                question_id=q.question_id,
                question_text=q.question_text,
                question_type=q.question_type,
                classification_status=q.classification_status,
                source_filename=q.source_filename,
                paper_id=q.paper_id,
                paper_year=q.paper_year,
                topic_confidence=q.topic_confidence,
            )
        )
    return items


def build_data_quality(summary: FrequencyAnalysisSummary) -> DataQualityNotes:
    return DataQualityNotes(
        no_match_count=len(summary.no_match_questions),
        ambiguous_count=len(summary.ambiguous_questions),
        unclassified_count=len(summary.unclassified_questions),
        year_conflict_paper_count=len(summary.year_conflict_paper_ids),
        data_integrity_reconciled=summary.data_integrity.reconciled,
        notes=list(_DATA_QUALITY_NOTES),
    )


def build_final_report(
    summary: FrequencyAnalysisSummary,
    canonical_questions: list[CanonicalQuestion],
    questions: list[Question],
) -> FinalAnalysisReport:
    """Single entry point: composes every builder above into one
    FinalAnalysisReport. Deterministic given identical inputs -- no
    randomness, no LLM/embedding calls, every ranking uses an explicit sort
    key (mirrors frequency_analysis.aggregate.build_frequency_analysis)."""
    questions_by_id = {q.question_id: q for q in questions}
    tested, untested = split_topic_analysis(summary.topic_summaries)
    return FinalAnalysisReport(
        textbook_file_hash=summary.textbook_file_hash,
        summary=build_executive_summary(summary),
        question_type_distribution=build_question_type_distribution(
            summary.question_type_distribution, summary.total_question_occurrences
        ),
        most_repeated_questions=build_repeated_questions(
            summary.most_repeated_questions, canonical_questions, questions_by_id
        ),
        most_tested_topics_by_occurrences=topics_by_occurrences(summary.most_tested_topics),
        most_tested_topics_by_unique_questions=rank_topics_by_unique_questions(summary.topic_summaries),
        topic_analysis=tested,
        untested_topics=untested,
        no_match_questions=hydrate_unclassified(summary.no_match_questions, questions_by_id),
        ambiguous_questions=hydrate_unclassified(summary.ambiguous_questions, questions_by_id),
        data_quality=build_data_quality(summary),
    )
