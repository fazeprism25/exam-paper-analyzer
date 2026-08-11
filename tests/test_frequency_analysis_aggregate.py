from exampapersorter.frequency_analysis.aggregate import (
    build_canonical_frequency,
    build_frequency_analysis,
    build_topic_summary,
    compute_classification_status_counts,
    compute_question_type_counts,
    compute_repetition_by_type,
    compute_year_conflict_paper_ids,
    compute_years_represented,
    first_and_latest_year,
    question_to_canonical_map,
    rank_canonical_questions,
    rank_topics,
    verify_data_integrity,
)
from exampapersorter.schemas import (
    CanonicalQuestion,
    MetadataFieldValue,
    Paper,
    PaperMetadata,
    Question,
    Topic,
)


def t(id, level=1, parent_id=None, name=None):
    return Topic(id=id, name=name or id, level=level, parent_id=parent_id, source_pages=[10])


def q(
    question_id,
    text="Explain X.",
    paper_id="p1",
    paper_year=None,
    qtype="short_answer",
    topic_id=None,
    topic_name=None,
    status="unclassified",
    source_filename="paper.pdf",
):
    return Question(
        question_id=question_id, paper_id=paper_id, section_id=None,
        source_filename=source_filename, source_file_hash="hash1", paper_year=paper_year,
        question_number="1", question_text=text, question_type=qtype,
        source_pages=[1], extraction_confidence=0.9,
        topic_id=topic_id, topic_name=topic_name, topic_confidence=0.9 if topic_id else None,
        classification_status=status,
    )


def cq(canonical_id, source_ids, text="Explain X.", qtype="short_answer", topic_id=None, topic_name=None,
       occurrence_count=None, unique_paper_count=None, years=None, dedup_status="singleton", dedup_confidence=None):
    return CanonicalQuestion(
        canonical_question_id=canonical_id, canonical_question_text=text, question_type=qtype,
        topic_id=topic_id, topic_name=topic_name, representative_question_id=source_ids[0],
        dedup_confidence=dedup_confidence, dedup_status=dedup_status,
        occurrence_count=occurrence_count if occurrence_count is not None else len(source_ids),
        unique_paper_count=unique_paper_count if unique_paper_count is not None else len(source_ids),
        years=years or [], source_question_ids=source_ids,
    )


def paper(paper_id, year_conflict=False):
    field = lambda **kw: MetadataFieldValue(**kw)
    return Paper(
        paper_id=paper_id, file_hash="h1", start_page=1, end_page=5,
        metadata=PaperMetadata(
            exam_name=field(), institution=field(), subject=field(),
            date=field(), year=field(conflict=year_conflict), paper_identifier=field(),
        ),
        boundary_confidence=0.9,
    )


# --- overall totals ---


def test_question_type_counts_sums_to_total_and_includes_all_keys():
    questions = [q("q1", qtype="mcq"), q("q2", qtype="mcq"), q("q3", qtype="long_answer")]
    counts = compute_question_type_counts(questions)
    assert counts.mcq == 2
    assert counts.long_answer == 1
    assert counts.short_answer == 0  # present, zero, not omitted
    assert counts.total() == 3


def test_classification_status_counts_includes_all_four_keys():
    questions = [q("q1", status="classified"), q("q2", status="no_match")]
    counts = compute_classification_status_counts(questions)
    assert counts == {"classified": 1, "no_match": 1, "ambiguous": 0, "unclassified": 0}


# --- years ---


def test_years_are_deduplicated_and_sorted_deterministically():
    questions = [q("q1", paper_year="2022"), q("q2", paper_year="2017-18"), q("q3", paper_year="2022")]
    assert compute_years_represented(questions) == ["2017-18", "2022"]


def test_missing_year_does_not_fabricate_a_value():
    questions = [q("q1", paper_year=None), q("q2", paper_year="2022")]
    assert compute_years_represented(questions) == ["2022"]
    assert compute_years_represented([q("q1", paper_year=None)]) == []


def test_first_and_latest_year_derived_without_inventing_values():
    first, latest = first_and_latest_year(["2022", "2017-18", "2024-25"])
    assert first == "2017-18"
    assert latest == "2024-25"


def test_first_and_latest_year_none_when_nothing_parseable():
    assert first_and_latest_year([]) == (None, None)
    assert first_and_latest_year(["unknown", "n/a"]) == (None, None)


def test_year_conflict_paper_ids_exposes_conflict_without_resolving_it():
    papers = [paper("p1", year_conflict=True), paper("p2", year_conflict=False)]
    assert compute_year_conflict_paper_ids(papers) == ["p1"]


# --- canonical question frequency / traceability ---


def test_build_canonical_frequency_traces_back_to_source_occurrences():
    questions_by_id = {
        "q1": q("q1", paper_id="p2", paper_year="2022", source_filename="a.pdf"),
        "q2": q("q2", paper_id="p5", paper_year="2024", source_filename="b.pdf"),
    }
    canonical = cq("c1", ["q1", "q2"], years=["2022", "2024"])
    freq = build_canonical_frequency(canonical, questions_by_id)
    assert freq.occurrence_count == 2
    assert freq.paper_ids == ["p2", "p5"]
    assert freq.source_filenames == ["a.pdf", "b.pdf"]
    assert freq.source_question_ids == ["q1", "q2"]  # traceable, not flattened to a count


def test_canonical_group_with_mixed_source_types_is_represented_honestly():
    questions_by_id = {
        "q1": q("q1", qtype="short_answer"),
        "q2": q("q2", qtype="long_answer"),
    }
    canonical = cq("c1", ["q1", "q2"])
    freq = build_canonical_frequency(canonical, questions_by_id)
    assert freq.question_types == ["long_answer", "short_answer"]  # both preserved, not collapsed to one


# --- ranking ---


def test_most_repeated_ranking_orders_by_occurrence_then_paper_count_then_id():
    questions_by_id = {f"q{i}": q(f"q{i}") for i in range(1, 6)}
    canonicals = [
        cq("c_low", ["q1"], occurrence_count=1, unique_paper_count=1),
        cq("c_b", ["q2", "q3"], occurrence_count=2, unique_paper_count=2),
        cq("c_a", ["q4", "q5"], occurrence_count=2, unique_paper_count=1),
    ]
    freqs = [build_canonical_frequency(c, questions_by_id) for c in canonicals]
    ranked = rank_canonical_questions(freqs)
    assert [r.canonical_question_id for r in ranked] == ["c_b", "c_a", "c_low"]
    assert [r.rank for r in ranked] == [1, 2, 3]


def test_ranking_tie_break_is_deterministic_canonical_id_order():
    questions_by_id = {f"q{i}": q(f"q{i}") for i in range(1, 5)}
    canonicals = [
        cq("c_zzz", ["q1", "q2"], occurrence_count=2, unique_paper_count=2),
        cq("c_aaa", ["q3", "q4"], occurrence_count=2, unique_paper_count=2),
    ]
    freqs = [build_canonical_frequency(c, questions_by_id) for c in canonicals]
    ranked = rank_canonical_questions(freqs)
    assert [r.canonical_question_id for r in ranked] == ["c_aaa", "c_zzz"]


def test_topic_ranking_tie_break_is_deterministic_topic_id_order():
    from exampapersorter.schemas import TopicFrequencySummary, QuestionTypeCounts

    summaries = [
        TopicFrequencySummary(topic_id="t_z", topic_name="Z", level=1, total_occurrences=5, unique_canonical_questions=3, type_counts=QuestionTypeCounts(), paper_count=1, years=[]),
        TopicFrequencySummary(topic_id="t_a", topic_name="A", level=1, total_occurrences=5, unique_canonical_questions=3, type_counts=QuestionTypeCounts(), paper_count=1, years=[]),
    ]
    ranked = rank_topics(summaries)
    assert [r.topic_id for r in ranked] == ["t_a", "t_z"]


# --- topic-level aggregation ---


def test_topic_summary_reconciles_occurrences_and_unique_questions():
    topic = t("chapter_a")
    questions = [
        q("q1", topic_id="chapter_a", topic_name="Chapter A", status="classified", qtype="mcq"),
        q("q2", topic_id="chapter_a", topic_name="Chapter A", status="classified", qtype="mcq"),
        q("q3", topic_id="chapter_a", topic_name="Chapter A", status="classified", qtype="long_answer"),
    ]
    canonicals = [cq("c1", ["q1", "q2"]), cq("c2", ["q3"])]
    q2c = question_to_canonical_map(canonicals)
    canonical_by_id = {c.canonical_question_id: c for c in canonicals}
    summary = build_topic_summary(topic, questions, q2c, canonical_by_id)
    assert summary.total_occurrences == 3
    assert summary.unique_canonical_questions == 2
    assert summary.type_counts.mcq == 2
    assert summary.type_counts.long_answer == 1
    assert len(summary.repeated_canonical_questions) == 1
    assert summary.repeated_canonical_questions[0].occurrences_in_topic == 2


def test_zero_question_topic_is_represented_not_omitted():
    topic = t("empty_chapter")
    summary = build_topic_summary(topic, [], {}, {})
    assert summary.total_occurrences == 0
    assert summary.unique_canonical_questions == 0
    assert summary.type_counts.total() == 0
    assert summary.repeated_canonical_questions == []


def test_no_match_and_ambiguous_questions_are_not_assigned_to_a_topic():
    topics = [t("chapter_a")]
    questions = [
        q("q1", topic_id="chapter_a", status="classified"),
        q("q2", topic_id=None, status="no_match"),
        q("q3", topic_id=None, status="ambiguous"),
    ]
    canonicals = [cq("c1", ["q1"]), cq("c2", ["q2"]), cq("c3", ["q3"])]
    papers = [paper("p1")]
    summary = build_frequency_analysis(topics, questions, canonicals, papers, "hash1")
    # still counted overall
    assert summary.total_question_occurrences == 3
    assert summary.no_match_questions == ["q2"]
    assert summary.ambiguous_questions == ["q3"]
    # but not folded into the chapter's totals
    chapter_summary = next(s for s in summary.topic_summaries if s.topic_id == "chapter_a")
    assert chapter_summary.total_occurrences == 1


# --- repetition by type ---


def test_repetition_by_type_counts_reconcile():
    questions = [
        q("q1", qtype="mcq"), q("q2", qtype="mcq"), q("q3", qtype="mcq"),  # q1/q2 same canonical
        q("q4", qtype="long_answer"),
    ]
    canonicals = [cq("c1", ["q1", "q2"]), cq("c2", ["q3"]), cq("c3", ["q4"])]
    q2c = question_to_canonical_map(canonicals)
    stats = compute_repetition_by_type(questions, q2c)
    assert stats["mcq"].total_occurrences == 3
    assert stats["mcq"].unique_canonical_questions == 2
    assert stats["mcq"].repeated_canonical_questions == 1
    assert stats["long_answer"].total_occurrences == 1
    assert stats["long_answer"].repeated_canonical_questions == 0
    assert stats["short_answer"].total_occurrences == 0  # present, zero, not omitted


# --- data integrity ---


def test_data_integrity_reconciles_for_well_formed_state():
    questions = [q("q1"), q("q2")]
    canonicals = [cq("c1", ["q1", "q2"])]
    q2c = question_to_canonical_map(canonicals)
    report = verify_data_integrity(questions, canonicals, q2c)
    assert report.total_raw_occurrences == 2
    assert report.total_canonical_links == 2
    assert report.distinct_canonical_question_ids_in_links == 1
    assert report.orphaned_occurrences == []
    assert report.duplicate_links == []
    assert report.reconciled is True


def test_data_integrity_flags_orphaned_occurrence():
    questions = [q("q1"), q("q2")]
    canonicals = [cq("c1", ["q1"])]  # q2 has no link
    q2c = question_to_canonical_map(canonicals)
    report = verify_data_integrity(questions, canonicals, q2c)
    assert report.orphaned_occurrences == ["q2"]
    assert report.reconciled is False


def test_data_integrity_flags_duplicate_link():
    questions = [q("q1")]
    canonicals = [cq("c1", ["q1"]), cq("c2", ["q1"])]  # q1 counted twice
    q2c = question_to_canonical_map(canonicals)
    report = verify_data_integrity(questions, canonicals, q2c)
    assert report.duplicate_links == ["q1"]
    assert report.reconciled is False


# --- full pipeline / reconciliation against the 215/191-style fixture shape ---


def test_build_frequency_analysis_reconciles_totals_end_to_end():
    topics = [t("chapter_a"), t("chapter_b")]
    questions = (
        [q(f"a{i}", topic_id="chapter_a", topic_name="A", status="classified", qtype="mcq", paper_year="2022") for i in range(1, 4)]
        + [q(f"b{i}", topic_id="chapter_b", topic_name="B", status="classified", qtype="long_answer", paper_year="2023") for i in range(1, 3)]
    )
    canonicals = [cq("c_a1", ["a1", "a2"], years=["2022"]), cq("c_a2", ["a3"], years=["2022"]),
                  cq("c_b1", ["b1", "b2"], years=["2023"])]
    papers = [paper("p1")]
    summary = build_frequency_analysis(topics, questions, canonicals, papers, "hash1")

    assert summary.total_question_occurrences == 5
    assert summary.total_canonical_questions == 3
    assert summary.data_integrity.reconciled is True
    # type counts sum to total occurrences
    assert summary.question_type_distribution.total() == 5
    # topic totals reconcile with classified occurrences
    assert sum(s.total_occurrences for s in summary.topic_summaries) == 5
    assert len(summary.topic_summaries) == 2  # both chapters present
    assert summary.most_repeated_questions[0].canonical_question_id == "c_a1"


# --- empty corpus ---


def test_build_frequency_analysis_handles_empty_corpus_gracefully():
    summary = build_frequency_analysis([], [], [], [], "hash1")
    assert summary.total_question_occurrences == 0
    assert summary.total_canonical_questions == 0
    assert summary.topic_summaries == []
    assert summary.most_repeated_questions == []
    assert summary.most_tested_topics == []
    assert summary.data_integrity.reconciled is True


# --- idempotency ---


def test_build_frequency_analysis_is_idempotent_over_unchanged_input():
    topics = [t("chapter_a")]
    questions = [q("q1", topic_id="chapter_a", status="classified"), q("q2", topic_id="chapter_a", status="classified")]
    canonicals = [cq("c1", ["q1", "q2"])]
    papers = [paper("p1")]
    first = build_frequency_analysis(topics, questions, canonicals, papers, "hash1")
    second = build_frequency_analysis(topics, questions, canonicals, papers, "hash1")
    assert first.model_dump() == second.model_dump()
