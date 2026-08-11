"""Synthetic-data tests for final_analysis/build.py's pure functions --
mirrors tests/test_frequency_analysis_aggregate.py's style. Covers project
notes section 24's checklist: repeated + singleton questions, multiple
question types (including a canonical group spanning more than one type,
BMR-style), multiple chapters with a preserved section/group/leaf hierarchy,
a zero-question chapter, no-match/ambiguous questions, missing + multiple
years, and determinism.
"""
from __future__ import annotations

from exampapersorter.final_analysis.build import (
    build_data_quality,
    build_executive_summary,
    build_final_report,
    build_question_type_distribution,
    build_repeated_questions,
    hydrate_unclassified,
    rank_topics_by_unique_questions,
    split_topic_analysis,
    topics_by_occurrences,
)
from exampapersorter.frequency_analysis.aggregate import build_frequency_analysis
from exampapersorter.schemas import CanonicalQuestion, MetadataFieldValue, Paper, PaperMetadata, Question, Topic


def t(id, level=1, parent_id=None, name=None):
    return Topic(id=id, name=name or id, level=level, parent_id=parent_id, source_pages=[10])


def q(question_id, paper_id="p1", topic_id=None, status="unclassified", qtype="short_answer",
      paper_year=None, source_filename="paper.pdf", text="Explain X."):
    return Question(
        question_id=question_id, paper_id=paper_id, section_id=None,
        source_filename=source_filename, source_file_hash="hash1", paper_year=paper_year,
        question_number="1", question_text=text, question_type=qtype,
        source_pages=[1], extraction_confidence=0.9,
        topic_id=topic_id, classification_status=status,
        topic_confidence=0.8 if status == "classified" else None,
    )


def paper(paper_id="p1"):
    field = lambda **kw: MetadataFieldValue(**kw)
    return Paper(
        paper_id=paper_id, file_hash="h1", start_page=1, end_page=5,
        metadata=PaperMetadata(
            exam_name=field(), institution=field(), subject=field(),
            date=field(), year=field(), paper_identifier=field(),
        ),
        boundary_confidence=0.9,
    )


def _synthetic_corpus():
    """A small corpus exercising every case from project notes section 24:
    Section One > Metabolism > {Metabolism of lipids, Introduction to
    metabolism}; Section One > Organ systems > Organ function tests;
    Section Two > Genetics (zero questions, untested).
    """
    topics = [
        t("sec1", level=1, name="Section One"),
        t("grp_metab", level=2, parent_id="sec1", name="Metabolism"),
        t("topic_lipids", level=3, parent_id="grp_metab", name="Metabolism of lipids"),
        t("topic_intro", level=3, parent_id="grp_metab", name="Introduction to metabolism"),
        t("grp_organ", level=2, parent_id="sec1", name="Organ systems"),
        t("topic_organ", level=3, parent_id="grp_organ", name="Organ function tests"),
        t("sec2", level=1, name="Section Two"),
        t("topic_genetics", level=2, parent_id="sec2", name="Genetics"),
    ]

    questions = [
        # Repeated canonical group (3 occurrences), single type, multiple years -> topic_lipids
        q("q1", paper_id="p1", topic_id="topic_lipids", status="classified", qtype="mcq", paper_year="2019-20"),
        q("q2", paper_id="p2", topic_id="topic_lipids", status="classified", qtype="mcq", paper_year="2022"),
        q("q3", paper_id="p3", topic_id="topic_lipids", status="classified", qtype="mcq", paper_year="2019-20"),
        # Singleton -> topic_lipids, missing year
        q("q4", paper_id="p2", topic_id="topic_lipids", status="classified", qtype="short_answer",
          paper_year=None, text="Define fatty acid."),
        # BMR-style repeated group spanning two question types -> topic_intro
        q("q5", paper_id="p1", topic_id="topic_intro", status="classified", qtype="short_answer",
          paper_year="2019-20", text="Define BMR / factors / diagnostic importance."),
        q("q6", paper_id="p3", topic_id="topic_intro", status="classified", qtype="long_answer",
          paper_year="2022", text="Define BMR / factors / diagnostic importance."),
        # topic_organ: singleton
        q("q7", paper_id="p1", topic_id="topic_organ", status="classified", qtype="mcq", paper_year="2024"),
        # no_match / ambiguous -- not attributed to any topic
        q("q8", paper_id="p2", status="no_match", qtype="mcq", text="Unclear question A."),
        q("q9", paper_id="p3", status="ambiguous", qtype="short_answer", text="Unclear question B."),
    ]

    canonicals = [
        CanonicalQuestion(
            canonical_question_id="c_lipid", canonical_question_text="Beta oxidation of palmitic acid.",
            question_type="mcq", topic_id="topic_lipids", topic_name="Metabolism of lipids",
            representative_question_id="q1", dedup_status="semantic_merge", dedup_confidence=0.75,
        ),
        CanonicalQuestion(
            canonical_question_id="c_lipid2", canonical_question_text="Define fatty acid.",
            question_type="short_answer", topic_id="topic_lipids", topic_name="Metabolism of lipids",
            representative_question_id="q4", dedup_status="singleton",
        ),
        CanonicalQuestion(
            canonical_question_id="c_bmr", canonical_question_text="Define BMR / factors / diagnostic importance.",
            question_type="short_answer", topic_id="topic_intro", topic_name="Introduction to metabolism",
            representative_question_id="q5", dedup_status="semantic_merge", dedup_confidence=0.7,
        ),
        CanonicalQuestion(
            canonical_question_id="c_organ", canonical_question_text="Explain X.",
            question_type="mcq", topic_id="topic_organ", topic_name="Organ function tests",
            representative_question_id="q7", dedup_status="singleton",
        ),
        CanonicalQuestion(
            canonical_question_id="c_nomatch", canonical_question_text="Unclear question A.",
            question_type="mcq", representative_question_id="q8", dedup_status="singleton",
        ),
        CanonicalQuestion(
            canonical_question_id="c_ambig", canonical_question_text="Unclear question B.",
            question_type="short_answer", representative_question_id="q9", dedup_status="singleton",
        ),
    ]
    links = {
        "q1": "c_lipid", "q2": "c_lipid", "q3": "c_lipid",
        "q4": "c_lipid2",
        "q5": "c_bmr", "q6": "c_bmr",
        "q7": "c_organ",
        "q8": "c_nomatch",
        "q9": "c_ambig",
    }
    # replace_canonical_state expects a dict; build_frequency_analysis takes
    # the canonical list directly with source_question_ids populated (as if
    # already hydrated by Database._hydrate_canonical).
    for cq in canonicals:
        cq.source_question_ids = sorted(k for k, v in links.items() if v == cq.canonical_question_id)
        cq.occurrence_count = len(cq.source_question_ids)
        member_qs = [qq for qq in questions if qq.question_id in cq.source_question_ids]
        cq.unique_paper_count = len({m.paper_id for m in member_qs})
        cq.years = sorted({m.paper_year for m in member_qs if m.paper_year})

    papers = [paper("p1"), paper("p2"), paper("p3")]
    return topics, questions, canonicals, papers


def _summary():
    topics, questions, canonicals, papers = _synthetic_corpus()
    return build_frequency_analysis(topics, questions, canonicals, papers, "hash1"), questions, canonicals


def test_executive_summary_counts_match_stage5():
    summary, questions, canonicals = _summary()
    es = build_executive_summary(summary)
    assert es.total_papers == summary.total_papers == 3
    assert es.total_question_occurrences == summary.total_question_occurrences == 9
    assert es.total_canonical_questions == summary.total_canonical_questions == 6
    assert es.total_topics == summary.topic_count == 8
    assert es.no_match_count == 1
    assert es.ambiguous_count == 1
    assert es.unclassified_count == 0
    assert es.data_integrity_reconciled is True
    # 3 leaf topics have classified questions (lipids, intro, organ)
    assert es.topics_tested == 3
    assert es.topics_untested == 5


def test_repeated_questions_include_all_groups_and_exclude_singletons():
    summary, questions, canonicals = _summary()
    questions_by_id = {q.question_id: q for q in questions}
    items = build_repeated_questions(summary.most_repeated_questions, canonicals, questions_by_id)

    ids = {i.canonical_question_id for i in items}
    assert ids == {"c_lipid", "c_bmr"}
    assert "c_lipid2" not in ids and "c_organ" not in ids and "c_nomatch" not in ids and "c_ambig" not in ids

    lipid = next(i for i in items if i.canonical_question_id == "c_lipid")
    assert lipid.occurrence_count == 3
    assert lipid.unique_paper_count == 3
    assert lipid.dedup_status == "semantic_merge"
    assert lipid.dedup_confidence == 0.75
    assert set(lipid.paper_ids) == {"p1", "p2", "p3"}

    bmr = next(i for i in items if i.canonical_question_id == "c_bmr")
    # Multi-type canonical group -- both member types preserved, not collapsed to one.
    assert set(bmr.question_types) == {"short_answer", "long_answer"}
    assert bmr.occurrence_count == 2


def test_repeated_questions_are_ranked_by_rank_field_ascending():
    summary, questions, canonicals = _summary()
    questions_by_id = {q.question_id: q for q in questions}
    items = build_repeated_questions(summary.most_repeated_questions, canonicals, questions_by_id)
    ranks = [i.rank for i in items]
    assert ranks == sorted(ranks)
    # c_lipid (3 occurrences) must outrank c_bmr (2 occurrences)
    by_id = {i.canonical_question_id: i.rank for i in items}
    assert by_id["c_lipid"] < by_id["c_bmr"]


def test_chapter_occurrence_ranking():
    summary, _, _ = _summary()
    ranking = topics_by_occurrences(summary.most_tested_topics)
    by_id = {r.topic_id: r for r in ranking}
    assert by_id["topic_lipids"].total_occurrences == 4  # q1,q2,q3,q4
    assert by_id["topic_intro"].total_occurrences == 2   # q5,q6
    assert by_id["topic_organ"].total_occurrences == 1   # q7
    ranks = [r.topic_id for r in sorted(ranking, key=lambda r: r.rank)]
    assert ranks[0] == "topic_lipids"


def test_chapter_unique_question_ranking_can_diverge_from_occurrence_ranking():
    summary, _, _ = _summary()
    unique_ranking = rank_topics_by_unique_questions(summary.topic_summaries)
    by_id = {r.topic_id: r for r in unique_ranking}
    # topic_lipids: 2 unique (c_lipid, c_lipid2); topic_intro: 1 unique (c_bmr); topic_organ: 1 unique (c_organ)
    assert by_id["topic_lipids"].unique_canonical_questions == 2
    assert by_id["topic_intro"].unique_canonical_questions == 1
    assert by_id["topic_organ"].unique_canonical_questions == 1
    # Untested topics never appear in either ranking.
    assert "topic_genetics" not in by_id


def test_question_types_counts_and_percentages_reconcile():
    summary, _, _ = _summary()
    report = build_question_type_distribution(summary.question_type_distribution, summary.total_question_occurrences)
    assert report.total_occurrences == 9
    total_from_shares = sum(s.occurrence_count for s in report.by_type)
    assert total_from_shares == 9
    mcq_share = next(s for s in report.by_type if s.question_type == "mcq")
    assert mcq_share.occurrence_count == 5  # q1,q2,q3,q7,q8 (q8 is no_match but still counted -- occurrence-level)
    assert mcq_share.percentage == round(5 / 9 * 100, 1)


def test_question_type_distribution_handles_zero_total():
    from exampapersorter.schemas import QuestionTypeCounts
    report = build_question_type_distribution(QuestionTypeCounts(), 0)
    assert all(s.percentage == 0.0 for s in report.by_type)


def test_zero_question_topics_all_appear_as_untested():
    """Only topics whose ENTIRE subtree has zero questions are 'untested'.
    sec1/grp_metab/grp_organ have zero DIRECT occurrences of their own but
    each has a tested descendant leaf, so they belong in the Analysis-by-
    Chapter tree (as structural headers), not the flat Untested Topics
    list -- see build.split_topic_analysis's docstring, confirmed against
    the real fixture where 14 of 16 Section/Group nodes are in exactly
    this position."""
    summary, _, _ = _summary()
    tested, untested = split_topic_analysis(summary.topic_summaries)
    tested_ids = {t.topic_id for t in tested}
    untested_ids = {u.topic_id for u in untested}
    assert tested_ids == {"topic_lipids", "topic_intro", "topic_organ", "sec1", "grp_metab", "grp_organ"}
    # sec2 -> topic_genetics is a fully empty subtree end to end.
    assert untested_ids == {"sec2", "topic_genetics"}
    assert len(tested) + len(untested) == summary.topic_count
    # Structural ancestors are honestly reported as having 0 of their own.
    by_id = {t.topic_id: t for t in tested}
    assert by_id["sec1"].total_occurrences == 0
    assert by_id["grp_metab"].total_occurrences == 0


def test_no_match_and_ambiguous_questions_stay_visible_with_full_text():
    summary, questions, _ = _summary()
    questions_by_id = {q.question_id: q for q in questions}
    no_match = hydrate_unclassified(summary.no_match_questions, questions_by_id)
    ambiguous = hydrate_unclassified(summary.ambiguous_questions, questions_by_id)
    assert len(no_match) == 1
    assert no_match[0].question_id == "q8"
    assert no_match[0].question_text == "Unclear question A."
    assert no_match[0].classification_status == "no_match"
    assert len(ambiguous) == 1
    assert ambiguous[0].question_id == "q9"
    assert ambiguous[0].classification_status == "ambiguous"


def test_missing_metadata_stays_missing_not_fabricated():
    summary, questions, canonicals = _summary()
    questions_by_id = {q.question_id: q for q in questions}
    # q4 (singleton, missing paper_year) is not a repeated group, so it
    # never appears in most_repeated_questions -- confirmed directly here.
    ids = {i.canonical_question_id for i in build_repeated_questions(
        summary.most_repeated_questions, canonicals, questions_by_id
    )}
    assert "c_lipid2" not in ids
    q4 = questions_by_id["q4"]
    assert q4.paper_year is None


def test_data_quality_notes_reconcile_with_summary_counts():
    summary, _, _ = _summary()
    dq = build_data_quality(summary)
    assert dq.no_match_count == len(summary.no_match_questions) == 1
    assert dq.ambiguous_count == len(summary.ambiguous_questions) == 1
    assert dq.data_integrity_reconciled == summary.data_integrity.reconciled is True
    assert len(dq.notes) > 0


def test_build_final_report_is_deterministic():
    summary, questions, canonicals = _summary()
    first = build_final_report(summary, canonicals, questions)
    second = build_final_report(summary, canonicals, questions)
    assert first.model_dump() == second.model_dump()


def test_build_final_report_json_round_trips():
    summary, questions, canonicals = _summary()
    report = build_final_report(summary, canonicals, questions)
    dumped = report.model_dump_json()
    from exampapersorter.schemas import FinalAnalysisReport
    reloaded = FinalAnalysisReport.model_validate_json(dumped)
    assert reloaded == report


def test_build_final_report_preserves_topic_hierarchy():
    summary, questions, canonicals = _summary()
    report = build_final_report(summary, canonicals, questions)
    by_id = {t.topic_id: t for t in report.topic_analysis}
    assert by_id["topic_lipids"].parent_id == "grp_metab"
    assert by_id["topic_intro"].parent_id == "grp_metab"
    assert by_id["topic_organ"].parent_id == "grp_organ"


def test_build_final_report_chapter_type_breakdown_matches_stage5():
    summary, questions, canonicals = _summary()
    report = build_final_report(summary, canonicals, questions)
    by_id = {t.topic_id: t for t in report.topic_analysis}
    stage5_by_id = {s.topic_id: s for s in summary.topic_summaries}
    # topic_analysis reuses Stage 5's TopicFrequencySummary verbatim -- its
    # type_counts must be identical, not recomputed.
    for topic_id in ("topic_lipids", "topic_intro", "topic_organ"):
        assert by_id[topic_id].type_counts == stage5_by_id[topic_id].type_counts
    lipids = by_id["topic_lipids"]
    assert lipids.type_counts.mcq == 3  # q1,q2,q3
    assert lipids.type_counts.short_answer == 1  # q4


def test_build_final_report_preserves_original_year_strings():
    summary, questions, canonicals = _summary()
    report = build_final_report(summary, canonicals, questions)
    lipid = next(r for r in report.most_repeated_questions if r.canonical_question_id == "c_lipid")
    # "2019-20" must survive verbatim, never normalized to "2019-2020"
    assert "2019-20" in lipid.years
    assert "2019-2020" not in lipid.years
