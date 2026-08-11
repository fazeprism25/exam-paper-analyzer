from exampapersorter.deduplication.grouping import build_canonical_questions, build_groups, select_representative
from exampapersorter.schemas import DedupPairDecision, Question


def q(question_id, text, qtype="short_answer", ambiguous=False, confidence=0.9):
    return Question(
        question_id=question_id, paper_id="p1", section_id=None,
        source_filename="paper.pdf", source_file_hash="hash1",
        question_number="1", question_text=text, question_type=qtype,
        source_pages=[1], extraction_confidence=confidence, ambiguous=ambiguous,
    )


def decision(a, b, verdict, confidence, source="llm_judge", reason=None):
    return DedupPairDecision(question_id_a=a, question_id_b=b, decision_source=source, verdict=verdict, confidence=confidence, reason=reason)


# --- build_groups (union-find) ---


def test_every_question_gets_a_group_even_with_no_decisions():
    groups = build_groups(["q1", "q2", "q3"], [])
    assert sorted(len(m) for m in groups.values()) == [1, 1, 1]


def test_same_question_edge_unions_two_questions():
    groups = build_groups(["q1", "q2", "q3"], [decision("q1", "q2", "same_question", 0.9)])
    sizes = sorted(len(m) for m in groups.values())
    assert sizes == [1, 2]


def test_transitive_merge_via_chained_edges():
    groups = build_groups(
        ["q1", "q2", "q3"],
        [decision("q1", "q2", "same_question", 0.9), decision("q2", "q3", "same_question", 0.9)],
    )
    assert len(groups) == 1
    (only_group,) = groups.values()
    assert sorted(only_group) == ["q1", "q2", "q3"]


def test_grouping_is_deterministic_regardless_of_decision_order():
    a = build_groups(["q1", "q2", "q3"], [decision("q1", "q2", "same_question", 0.9), decision("q2", "q3", "same_question", 0.9)])
    b = build_groups(["q1", "q2", "q3"], [decision("q2", "q3", "same_question", 0.9), decision("q1", "q2", "same_question", 0.9)])
    assert {frozenset(m) for m in a.values()} == {frozenset(m) for m in b.values()}


# --- select_representative ---


def test_representative_prefers_non_ambiguous_occurrence():
    members = [q("q1", "Explain glycolysis fully.", ambiguous=True, confidence=0.95), q("q2", "Explain glycolysis.", ambiguous=False, confidence=0.5)]
    assert select_representative(members).question_id == "q2"


def test_representative_prefers_higher_confidence_when_both_non_ambiguous():
    members = [q("q1", "Explain glycolysis.", confidence=0.5), q("q2", "Explain glycolysis.", confidence=0.95)]
    assert select_representative(members).question_id == "q2"


def test_representative_prefers_more_complete_longer_text_as_tiebreak():
    members = [q("q1", "Explain glycolysis.", confidence=0.9), q("q2", "Explain glycolysis in full detail.", confidence=0.9)]
    assert select_representative(members).question_id == "q2"


def test_representative_selection_is_fully_deterministic_on_ties():
    members = [q("qb", "Explain glycolysis.", confidence=0.9), q("qa", "Explain glycolysis.", confidence=0.9)]
    # everything else tied -> lexicographically smallest question_id wins
    assert select_representative(members).question_id == "qa"


# --- build_canonical_questions ---


def test_exact_duplicate_only_group_gets_exact_duplicate_status_and_full_confidence():
    questions_by_id = {"q1": q("q1", "Explain glycolysis."), "q2": q("q2", "Explain glycolysis.")}
    decisions = [decision("q1", "q2", "same_question", 1.0, source="exact_duplicate")]
    canonical, links = build_canonical_questions(questions_by_id, decisions)
    assert len(canonical) == 1
    assert canonical[0].dedup_status == "exact_duplicate"
    assert canonical[0].dedup_confidence == 1.0
    assert links == {"q1": canonical[0].canonical_question_id, "q2": canonical[0].canonical_question_id}


def test_llm_merged_group_gets_semantic_merge_status_and_min_confidence():
    questions_by_id = {"q1": q("q1", "Explain glycolysis."), "q2": q("q2", "Describe glycolysis."), "q3": q("q3", "Discuss glycolysis.")}
    decisions = [
        decision("q1", "q2", "same_question", 0.9),
        decision("q2", "q3", "same_question", 0.7),
    ]
    canonical, links = build_canonical_questions(questions_by_id, decisions)
    assert len(canonical) == 1
    assert canonical[0].dedup_status == "semantic_merge"
    assert canonical[0].dedup_confidence == 0.7  # min across the group's own merge edges
    assert set(links) == {"q1", "q2", "q3"}


def test_unrelated_singleton_has_no_confidence_and_singleton_status():
    questions_by_id = {"q1": q("q1", "Explain glycolysis.")}
    canonical, links = build_canonical_questions(questions_by_id, [])
    assert canonical[0].dedup_status == "singleton"
    assert canonical[0].dedup_confidence is None


def test_uncertain_neighbor_flags_an_otherwise_singleton_as_needs_review():
    questions_by_id = {"q1": q("q1", "Explain glycolysis."), "q2": q("q2", "Describe glycolysis in detail today.")}
    decisions = [decision("q1", "q2", "uncertain", 0.4, reason="ocr too corrupted to judge")]
    canonical, links = build_canonical_questions(questions_by_id, decisions)
    by_id = {c.representative_question_id: c for c in canonical}
    assert by_id["q1"].dedup_status == "needs_review"
    assert by_id["q1"].dedup_confidence == 0.4
    assert by_id["q2"].dedup_status == "needs_review"
    # never merged despite the flag
    assert links["q1"] != links["q2"]


def test_different_question_verdict_never_merges_and_never_flags_review():
    questions_by_id = {
        "q1": q("q1", "Name the rate-limiting enzyme of glycolysis."),
        "q2": q("q2", "Explain the regulation of glycolysis."),
    }
    decisions = [decision("q1", "q2", "different_question", 0.9)]
    canonical, links = build_canonical_questions(questions_by_id, decisions)
    assert links["q1"] != links["q2"]
    assert {c.dedup_status for c in canonical} == {"singleton"}


def test_same_set_of_members_produces_a_stable_canonical_id_across_calls():
    questions_by_id = {"q1": q("q1", "Explain glycolysis."), "q2": q("q2", "Explain glycolysis.")}
    decisions = [decision("q1", "q2", "same_question", 1.0, source="exact_duplicate")]
    first, _ = build_canonical_questions(questions_by_id, decisions)
    second, _ = build_canonical_questions(questions_by_id, decisions)
    assert first[0].canonical_question_id == second[0].canonical_question_id


def test_canonical_text_is_a_verbatim_source_occurrence_never_rewritten():
    questions_by_id = {"q1": q("q1", "Explain glycolysis."), "q2": q("q2", "Explain glycolysis.")}
    decisions = [decision("q1", "q2", "same_question", 1.0, source="exact_duplicate")]
    canonical, _ = build_canonical_questions(questions_by_id, decisions)
    assert canonical[0].canonical_question_text in ("Explain glycolysis.",)
    assert canonical[0].representative_question_id in ("q1", "q2")
