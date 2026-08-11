from dataclasses import replace

from exampapersorter.config import DEFAULT_CONFIG
from exampapersorter.deduplication.candidates import (
    exact_duplicate_decisions,
    generate_candidate_pairs,
    group_exact_duplicates,
    ordered_pair,
)
from exampapersorter.schemas import Question, QuestionEmbedding


def q(question_id, text, qtype="short_answer", options=None):
    return Question(
        question_id=question_id, paper_id="p1", section_id=None,
        source_filename="paper.pdf", source_file_hash="hash1",
        question_number="1", question_text=text, question_type=qtype,
        options=options or [], source_pages=[1], extraction_confidence=0.9,
    )


def emb(question_id, vector):
    return QuestionEmbedding(question_id=question_id, model="m", version="v1", content_hash="h", vector=vector)


# --- ordered_pair ---


def test_ordered_pair_is_lexicographically_stable_regardless_of_input_order():
    assert ordered_pair("q2", "q1") == ordered_pair("q1", "q2") == ("q1", "q2")


# --- group_exact_duplicates / exact_duplicate_decisions ---


def test_exact_duplicates_are_grouped_together():
    questions = [q("q1", "Explain glycolysis."), q("q2", "Explain   glycolysis"), q("q3", "What is glycogen?")]
    groups = group_exact_duplicates(questions)
    sizes = sorted(len(members) for members in groups.values())
    assert sizes == [1, 2]


def test_exact_duplicate_decisions_cover_every_pair_within_a_group():
    questions = [q("q1", "Explain glycolysis."), q("q2", "Explain glycolysis."), q("q3", "Explain glycolysis.")]
    groups = group_exact_duplicates(questions)
    decisions = exact_duplicate_decisions(groups)
    pairs = {(d.question_id_a, d.question_id_b) for d in decisions}
    assert pairs == {("q1", "q2"), ("q1", "q3"), ("q2", "q3")}
    assert all(d.verdict == "same_question" and d.confidence == 1.0 and d.decision_source == "exact_duplicate" for d in decisions)


def test_singleton_groups_produce_no_decisions():
    questions = [q("q1", "Explain glycolysis."), q("q2", "What is glycogen?")]
    groups = group_exact_duplicates(questions)
    assert exact_duplicate_decisions(groups) == []


def test_different_question_types_are_not_grouped_as_exact_duplicates():
    questions = [q("q1", "Explain glycolysis.", qtype="short_answer"), q("q2", "Explain glycolysis.", qtype="long_answer")]
    groups = group_exact_duplicates(questions)
    assert sorted(len(m) for m in groups.values()) == [1, 1]


# --- generate_candidate_pairs ---


def test_high_similarity_same_type_pair_becomes_a_candidate():
    questions = [q("q1", "Explain glycolysis.", qtype="short_answer"), q("q2", "Describe glycolysis.", qtype="short_answer")]
    embeddings_by_id = {"q1": emb("q1", [1.0, 0.0]), "q2": emb("q2", [0.99, 0.14])}
    pairs = generate_candidate_pairs(questions, embeddings_by_id, DEFAULT_CONFIG)
    assert len(pairs) == 1
    a, b, sim = pairs[0]
    assert (a, b) == ("q1", "q2")
    assert sim >= DEFAULT_CONFIG.embedding_similarity_threshold


def test_low_similarity_same_type_pair_is_not_a_candidate():
    questions = [q("q1", "Explain glycolysis.", qtype="short_answer"), q("q2", "What is the normal plasma pH?", qtype="short_answer")]
    embeddings_by_id = {"q1": emb("q1", [1.0, 0.0]), "q2": emb("q2", [0.0, 1.0])}
    pairs = generate_candidate_pairs(questions, embeddings_by_id, DEFAULT_CONFIG)
    assert pairs == []


def test_cross_type_pair_requires_the_stricter_threshold():
    # Similarity clears the ordinary same-type threshold but not the
    # stricter cross-type one -- must NOT become a candidate.
    config = replace(DEFAULT_CONFIG, embedding_similarity_threshold=0.85, embedding_cross_type_similarity_threshold=0.95)
    questions = [q("q1", "Which enzyme regulates glycolysis?", qtype="mcq"), q("q2", "Explain the regulation of glycolysis.", qtype="long_answer")]
    embeddings_by_id = {"q1": emb("q1", [1.0, 0.0]), "q2": emb("q2", [0.9, 0.436])}  # cos ~ 0.9, between the two thresholds
    pairs = generate_candidate_pairs(questions, embeddings_by_id, config)
    assert pairs == []


def test_cross_type_pair_can_still_become_a_candidate_above_the_stricter_threshold():
    config = replace(DEFAULT_CONFIG, embedding_similarity_threshold=0.85, embedding_cross_type_similarity_threshold=0.95)
    questions = [q("q1", "Explain glycolysis.", qtype="short_answer"), q("q2", "Explain glycolysis.", qtype="long_answer")]
    embeddings_by_id = {"q1": emb("q1", [1.0, 0.0]), "q2": emb("q2", [0.999, 0.045])}  # cos ~ 0.999
    pairs = generate_candidate_pairs(questions, embeddings_by_id, config)
    assert len(pairs) == 1


def test_missing_embedding_is_skipped_not_crashed():
    questions = [q("q1", "Explain glycolysis."), q("q2", "Describe glycolysis.")]
    embeddings_by_id = {"q1": emb("q1", [1.0, 0.0])}  # q2 missing
    pairs = generate_candidate_pairs(questions, embeddings_by_id, DEFAULT_CONFIG)
    assert pairs == []


def test_candidate_pair_ordering_is_canonical_regardless_of_input_order():
    questions = [q("q2", "Describe glycolysis."), q("q1", "Explain glycolysis.")]
    embeddings_by_id = {"q1": emb("q1", [1.0, 0.0]), "q2": emb("q2", [0.99, 0.14])}
    pairs = generate_candidate_pairs(questions, embeddings_by_id, DEFAULT_CONFIG)
    assert pairs[0][:2] == ("q1", "q2")
