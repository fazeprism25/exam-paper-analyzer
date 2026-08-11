"""Pure, deterministic aggregation functions over already-loaded Stage 1-4
records (Topic/Question/CanonicalQuestion/Paper) -- no database access, no
LLM calls, no embedding calls. Kept pure specifically so this module is
testable with synthetic in-memory data (see tests/test_frequency_analysis_
aggregate.py) without needing a real Database or PDF fixtures.

`build_frequency_analysis` is the single entry point; everything else here
is a building block it composes. frequency_analysis/pipeline.py is the only
caller that touches a Database -- it loads the Stage 1-4 records and hands
them to this module unchanged.
"""
from __future__ import annotations

import re

from exampapersorter.schemas import (
    CanonicalQuestion,
    CanonicalQuestionFrequency,
    DataIntegrityReport,
    FrequencyAnalysisSummary,
    Paper,
    Question,
    QuestionTypeCounts,
    RankedCanonicalQuestion,
    RankedTopic,
    Topic,
    TopicFrequencySummary,
    TopicRepeatedQuestion,
    TypeRepetitionStats,
)

_QUESTION_TYPE_KEYS = ("mcq", "long_answer", "short_answer", "essay", "short_essay", "other", "unknown")
_CLASSIFICATION_STATUS_KEYS = ("classified", "no_match", "ambiguous", "unclassified")

_LEADING_YEAR_RE = re.compile(r"^(\d{4})")


def _year_sort_key(year: str) -> tuple[int, int, str]:
    """Deterministic ordering for year strings (project notes section 30):
    a value with a parseable leading 4-digit year (e.g. "2017-18" -> 2017)
    sorts numerically first; anything else sorts after, alphabetically. The
    original string is never rewritten -- this only controls order."""
    match = _LEADING_YEAR_RE.match(year)
    if match:
        return (0, int(match.group(1)), year)
    return (1, 0, year)


def compute_years_represented(questions: list[Question]) -> list[str]:
    years = {q.paper_year for q in questions if q.paper_year}
    return sorted(years, key=_year_sort_key)


def first_and_latest_year(years: list[str]) -> tuple[str | None, str | None]:
    """Best-effort earliest/latest year among already-extracted year
    strings, derived only via _year_sort_key -- never invented. Returns
    (None, None) if no year in `years` has a parseable leading 4-digit
    number (project notes section 8: do not fabricate missing years)."""
    parseable = [y for y in years if _LEADING_YEAR_RE.match(y)]
    if not parseable:
        return None, None
    ordered = sorted(parseable, key=_year_sort_key)
    return ordered[0], ordered[-1]


def compute_question_type_counts(questions: list[Question]) -> QuestionTypeCounts:
    counts = {key: 0 for key in _QUESTION_TYPE_KEYS}
    for q in questions:
        counts[q.question_type] = counts.get(q.question_type, 0) + 1
    return QuestionTypeCounts(**counts)


def compute_classification_status_counts(questions: list[Question]) -> dict[str, int]:
    counts = {key: 0 for key in _CLASSIFICATION_STATUS_KEYS}
    for q in questions:
        counts[q.classification_status] = counts.get(q.classification_status, 0) + 1
    return counts


def compute_year_conflict_paper_ids(papers: list[Paper]) -> list[str]:
    """paper_id values whose Stage 2 year metadata was flagged as
    conflicting between document and filename provenance -- exposed here
    rather than silently resolved (project notes section 10/28); Stage 5
    introduces no new conflict-resolution policy of its own."""
    return sorted(p.paper_id for p in papers if p.metadata.year.conflict)


def question_to_canonical_map(canonical_questions: list[CanonicalQuestion]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for cq in canonical_questions:
        for qid in cq.source_question_ids:
            mapping[qid] = cq.canonical_question_id
    return mapping


def verify_data_integrity(
    questions: list[Question], canonical_questions: list[CanonicalQuestion], question_to_canonical: dict[str, str]
) -> DataIntegrityReport:
    all_ids = [q.question_id for q in questions]
    linked_ids = list(question_to_canonical.keys())
    orphaned = sorted(set(all_ids) - set(linked_ids))

    membership_counts: dict[str, int] = {}
    for cq in canonical_questions:
        for qid in cq.source_question_ids:
            membership_counts[qid] = membership_counts.get(qid, 0) + 1
    duplicates = sorted(qid for qid, count in membership_counts.items() if count > 1)

    distinct_canonical_ids = len(set(question_to_canonical.values()))
    reconciled = len(all_ids) == len(linked_ids) and not orphaned and not duplicates

    return DataIntegrityReport(
        total_raw_occurrences=len(all_ids),
        total_canonical_links=len(linked_ids),
        distinct_canonical_question_ids_in_links=distinct_canonical_ids,
        orphaned_occurrences=orphaned,
        duplicate_links=duplicates,
        reconciled=reconciled,
    )


def build_canonical_frequency(
    cq: CanonicalQuestion, questions_by_id: dict[str, Question]
) -> CanonicalQuestionFrequency:
    members = [questions_by_id[qid] for qid in cq.source_question_ids if qid in questions_by_id]
    question_types = sorted({m.question_type for m in members})
    paper_ids = sorted({m.paper_id for m in members})
    source_filenames = sorted({m.source_filename for m in members})
    first_year, latest_year = first_and_latest_year(cq.years)
    return CanonicalQuestionFrequency(
        canonical_question_id=cq.canonical_question_id,
        canonical_question_text=cq.canonical_question_text,
        question_types=question_types,
        topic_id=cq.topic_id,
        topic_name=cq.topic_name,
        occurrence_count=cq.occurrence_count,
        unique_paper_count=cq.unique_paper_count,
        years=cq.years,
        first_year=first_year,
        latest_year=latest_year,
        paper_ids=paper_ids,
        source_filenames=source_filenames,
        source_question_ids=cq.source_question_ids,
        dedup_status=cq.dedup_status,
        dedup_confidence=cq.dedup_confidence,
    )


def rank_canonical_questions(frequencies: list[CanonicalQuestionFrequency]) -> list[RankedCanonicalQuestion]:
    """Deterministic ranking (project notes section 15): occurrence_count
    desc, then unique_paper_count desc, then canonical_question_id asc as a
    fully deterministic final tiebreaker. Never LLM judgment."""
    ordered = sorted(
        frequencies, key=lambda f: (-f.occurrence_count, -f.unique_paper_count, f.canonical_question_id)
    )
    return [
        RankedCanonicalQuestion(
            rank=i + 1,
            canonical_question_id=f.canonical_question_id,
            canonical_question_text=f.canonical_question_text,
            occurrence_count=f.occurrence_count,
            unique_paper_count=f.unique_paper_count,
            years=f.years,
            question_types=f.question_types,
            topic_id=f.topic_id,
            topic_name=f.topic_name,
        )
        for i, f in enumerate(ordered)
    ]


def build_topic_summary(
    topic: Topic,
    topic_questions: list[Question],
    question_to_canonical: dict[str, str],
    canonical_by_id: dict[str, CanonicalQuestion],
) -> TopicFrequencySummary:
    type_counts = compute_question_type_counts(topic_questions)
    paper_ids = sorted({q.paper_id for q in topic_questions})
    years = compute_years_represented(topic_questions)

    members_by_canonical: dict[str, list[Question]] = {}
    for q in topic_questions:
        cid = question_to_canonical.get(q.question_id)
        if cid:
            members_by_canonical.setdefault(cid, []).append(q)

    repeated: list[TopicRepeatedQuestion] = []
    for cid, members in members_by_canonical.items():
        if len(members) <= 1:
            continue
        cq = canonical_by_id.get(cid)
        text = cq.canonical_question_text if cq else members[0].question_text
        repeated.append(
            TopicRepeatedQuestion(
                canonical_question_id=cid,
                canonical_question_text=text,
                occurrences_in_topic=len(members),
                question_types=sorted({m.question_type for m in members}),
                years=compute_years_represented(members),
            )
        )
    repeated.sort(key=lambda r: (-r.occurrences_in_topic, r.canonical_question_id))

    return TopicFrequencySummary(
        topic_id=topic.id,
        topic_name=topic.name,
        level=topic.level,
        parent_id=topic.parent_id,
        total_occurrences=len(topic_questions),
        unique_canonical_questions=len(members_by_canonical),
        type_counts=type_counts,
        paper_count=len(paper_ids),
        years=years,
        repeated_canonical_questions=repeated,
    )


def rank_topics(summaries: list[TopicFrequencySummary]) -> list[RankedTopic]:
    """Deterministic ranking (project notes section 16): total_occurrences
    desc, then unique_canonical_questions desc, then topic_id asc."""
    ordered = sorted(
        summaries, key=lambda s: (-s.total_occurrences, -s.unique_canonical_questions, s.topic_id)
    )
    return [
        RankedTopic(
            rank=i + 1,
            topic_id=s.topic_id,
            topic_name=s.topic_name,
            total_occurrences=s.total_occurrences,
            unique_canonical_questions=s.unique_canonical_questions,
        )
        for i, s in enumerate(ordered)
    ]


def compute_repetition_by_type(
    questions: list[Question], question_to_canonical: dict[str, str]
) -> dict[str, TypeRepetitionStats]:
    by_type: dict[str, list[Question]] = {key: [] for key in _QUESTION_TYPE_KEYS}
    for q in questions:
        by_type.setdefault(q.question_type, []).append(q)

    result: dict[str, TypeRepetitionStats] = {}
    for qtype, qs in by_type.items():
        counts: dict[str, int] = {}
        for q in qs:
            cid = question_to_canonical.get(q.question_id)
            if cid:
                counts[cid] = counts.get(cid, 0) + 1
        result[qtype] = TypeRepetitionStats(
            total_occurrences=len(qs),
            unique_canonical_questions=len(counts),
            repeated_canonical_questions=sum(1 for c in counts.values() if c > 1),
        )
    return result


def build_frequency_analysis(
    topics: list[Topic],
    questions: list[Question],
    canonical_questions: list[CanonicalQuestion],
    papers: list[Paper],
    textbook_file_hash: str,
) -> FrequencyAnalysisSummary:
    """Single entry point: composes every aggregation above into one
    FrequencyAnalysisSummary. Fully deterministic given identical inputs
    (project notes section 30) -- no randomness, no LLM/embedding calls,
    every ranking uses an explicit sort key. Handles an empty corpus
    (topics/questions/canonical_questions/papers all empty) gracefully,
    returning an all-zero summary rather than raising."""
    questions_by_id = {q.question_id: q for q in questions}
    canonical_by_id = {cq.canonical_question_id: cq for cq in canonical_questions}
    question_to_canonical = question_to_canonical_map(canonical_questions)

    questions_by_topic: dict[str, list[Question]] = {}
    for q in questions:
        if q.topic_id:
            questions_by_topic.setdefault(q.topic_id, []).append(q)

    topic_summaries = sorted(
        (
            build_topic_summary(t, questions_by_topic.get(t.id, []), question_to_canonical, canonical_by_id)
            for t in topics
        ),
        key=lambda s: s.topic_id,
    )
    most_tested_topics = rank_topics([s for s in topic_summaries if s.total_occurrences > 0])

    canonical_freqs = [build_canonical_frequency(cq, questions_by_id) for cq in canonical_questions]
    ranked_canonical = rank_canonical_questions(canonical_freqs)
    most_repeated_questions = [r for r in ranked_canonical if r.occurrence_count > 1]

    no_match_ids = sorted(q.question_id for q in questions if q.classification_status == "no_match")
    ambiguous_ids = sorted(q.question_id for q in questions if q.classification_status == "ambiguous")
    unclassified_ids = sorted(q.question_id for q in questions if q.classification_status == "unclassified")

    return FrequencyAnalysisSummary(
        textbook_file_hash=textbook_file_hash,
        total_question_occurrences=len(questions),
        total_canonical_questions=len(canonical_questions),
        total_papers=len(papers),
        topic_count=len(topics),
        question_type_distribution=compute_question_type_counts(questions),
        classification_status_distribution=compute_classification_status_counts(questions),
        years_represented=compute_years_represented(questions),
        year_conflict_paper_ids=compute_year_conflict_paper_ids(papers),
        most_repeated_questions=most_repeated_questions,
        most_tested_topics=most_tested_topics,
        topic_summaries=topic_summaries,
        repetition_by_question_type=compute_repetition_by_type(questions, question_to_canonical),
        no_match_questions=no_match_ids,
        ambiguous_questions=ambiguous_ids,
        unclassified_questions=unclassified_ids,
        data_integrity=verify_data_integrity(questions, canonical_questions, question_to_canonical),
    )
