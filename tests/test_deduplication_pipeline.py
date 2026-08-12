"""End-to-end Stage 4 orchestration tests. The real fastembed model and the
real LLM backend are never invoked here:
  - embeddings.get_or_compute_embeddings is monkeypatched to hand back
    hand-crafted vectors (so candidate generation's real cosine-similarity
    + threshold logic in candidates.py still runs, just against controlled
    numbers instead of a real model's output).
  - semantic_judge.judge_candidate_pairs is monkeypatched per test (mirrors
    test_topic_classification_pipeline.py's monkeypatching of
    classify_questions) so each test controls exactly what the "LLM" says,
    without needing network access or a real model.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from exampapersorter.config import DEFAULT_CONFIG
from exampapersorter.database import Database
from exampapersorter.deduplication import pipeline as pipeline_module
from exampapersorter.llm_client import LLMCallFailed
from exampapersorter.schemas import PairEquivalenceVerdict, Question, QuestionEmbedding, SemanticEquivalenceResult

CONFIG = replace(DEFAULT_CONFIG, dedup_pair_batch_size=10)


def q(question_id, text, qtype="short_answer", paper_id="p1", paper_year="2022", topic_id=None, topic_name=None):
    return Question(
        question_id=question_id, paper_id=paper_id, section_id=None,
        source_filename="paper.pdf", source_file_hash="hash1", paper_year=paper_year,
        question_number="1", question_text=text, question_type=qtype,
        source_pages=[1], extraction_confidence=0.9,
        topic_id=topic_id, topic_name=topic_name, classification_status="classified" if topic_id else "unclassified",
    )


def install_fake_embeddings(monkeypatch, vectors_by_id: dict[str, list[float]]):
    """Any question_id not in `vectors_by_id` gets an arbitrary-but-distinct
    orthogonal-ish vector so it never accidentally becomes a candidate."""
    def fake(questions, config, db):
        result = {}
        for i, question in enumerate(questions):
            vector = vectors_by_id.get(question.question_id)
            if vector is None:
                vector = [0.0] * 8 + [float(100 + i)]  # far from everything configured above
            result[question.question_id] = QuestionEmbedding(
                question_id=question.question_id, model=config.embedding_model,
                version=config.embedding_model_version, content_hash="h", vector=vector,
            )
        return result

    monkeypatch.setattr(pipeline_module.embeddings, "get_or_compute_embeddings", fake)


def install_fake_judge(monkeypatch, verdict_fn):
    """verdict_fn(pair_id, question_a, question_b) -> PairEquivalenceVerdict"""
    calls = {"n": 0, "pairs": []}

    def fake(pairs, config):
        calls["n"] += 1
        calls["pairs"].extend(p[0] for p in pairs)
        return SemanticEquivalenceResult(verdicts=[verdict_fn(pid, qa, qb) for pid, qa, qb in pairs])

    monkeypatch.setattr(pipeline_module, "judge_candidate_pairs", fake)
    return calls


# --- exact duplicates never touch the LLM ---


def test_exact_duplicate_merge_requires_no_llm_call(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    db.save_questions([q("q1", "Explain glycolysis."), q("q2", "Explain glycolysis.")])
    install_fake_embeddings(monkeypatch, {})
    judge_calls = install_fake_judge(monkeypatch, lambda pid, a, b: PairEquivalenceVerdict(pair_id=pid, verdict="same_question", confidence=0.9))

    result = pipeline_module.run_deduplication(CONFIG, db)

    assert judge_calls["n"] == 0
    assert result.canonical_group_count == 1
    assert result.status_counts == {"exact_duplicate": 1}
    detail = db.get_canonical_question_detail(db.get_canonical_question_id_for_question("q1"))
    assert detail.occurrence_count == 2
    db.close()


# --- LLM-confirmed paraphrase ---


def test_llm_confirmed_paraphrase_merges_via_candidate_pair(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    db.save_questions([q("q1", "Explain glycolysis."), q("q2", "Describe the process of glycolysis.")])
    install_fake_embeddings(monkeypatch, {"q1": [1.0, 0.0], "q2": [0.99, 0.14]})  # cos ~0.99
    judge_calls = install_fake_judge(monkeypatch, lambda pid, a, b: PairEquivalenceVerdict(pair_id=pid, verdict="same_question", confidence=0.9, reason="paraphrase"))

    result = pipeline_module.run_deduplication(CONFIG, db)

    assert judge_calls["n"] == 1
    assert result.canonical_group_count == 1
    canonical = result.canonical_questions[0]
    assert canonical.dedup_status == "semantic_merge"
    assert canonical.dedup_confidence == 0.9
    db.close()


# --- same topic, different question: must NOT merge ---


def test_same_topic_different_question_stays_separate(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    db.save_questions([
        q("q1", "Name the rate-limiting enzyme of glycolysis.", topic_id="t1", topic_name="Glycolysis"),
        q("q2", "Explain the regulation of glycolysis.", topic_id="t1", topic_name="Glycolysis"),
    ])
    install_fake_embeddings(monkeypatch, {"q1": [1.0, 0.0], "q2": [0.9, 0.436]})  # candidate, but...
    install_fake_judge(monkeypatch, lambda pid, a, b: PairEquivalenceVerdict(
        pair_id=pid, verdict="different_question", confidence=0.9, reason="same topic, different task"
    ))

    result = pipeline_module.run_deduplication(CONFIG, db)

    assert result.canonical_group_count == 2
    assert db.get_canonical_question_id_for_question("q1") != db.get_canonical_question_id_for_question("q2")
    assert all(cq.dedup_status == "singleton" for cq in result.canonical_questions)
    db.close()


# --- cross-type: high similarity alone must not even become a candidate ---


def test_cross_type_pair_below_stricter_threshold_never_reaches_the_llm(tmp_path, monkeypatch):
    config = replace(CONFIG, embedding_similarity_threshold=0.85, embedding_cross_type_similarity_threshold=0.95)
    db = Database(tmp_path / "t.db")
    db.save_questions([
        q("q1", "Which enzyme regulates glycolysis?", qtype="mcq"),
        q("q2", "Explain the regulation of glycolysis.", qtype="long_answer"),
    ])
    # cos ~0.9 -- clears the ordinary threshold but not the cross-type one
    install_fake_embeddings(monkeypatch, {"q1": [1.0, 0.0], "q2": [0.9, 0.436]})
    judge_calls = install_fake_judge(monkeypatch, lambda pid, a, b: PairEquivalenceVerdict(pair_id=pid, verdict="same_question", confidence=0.99))

    result = pipeline_module.run_deduplication(config, db)

    assert judge_calls["n"] == 0
    assert result.candidate_pairs_generated == 0
    assert result.canonical_group_count == 2
    db.close()


# --- OCR corruption: uncertain verdict, no hallucinated correction, no merge ---


def test_uncertain_verdict_on_corrupted_text_does_not_merge_or_rewrite_text(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    db.save_questions([
        q("q1", "Mousy odour of the urne is seen in ---"),
        q("q2", "Mousy odor of teh urine ---- disease"),
    ])
    install_fake_embeddings(monkeypatch, {"q1": [1.0, 0.0], "q2": [0.99, 0.14]})
    install_fake_judge(monkeypatch, lambda pid, a, b: PairEquivalenceVerdict(
        pair_id=pid, verdict="uncertain", confidence=0.3, reason="text too corrupted to judge reliably"
    ))

    result = pipeline_module.run_deduplication(CONFIG, db)

    assert result.canonical_group_count == 2
    statuses = {cq.dedup_status for cq in result.canonical_questions}
    assert statuses == {"needs_review"}
    # canonical text for each is exactly its own verbatim (corrupted) source text -- never corrected/rewritten
    texts = {cq.canonical_question_text for cq in result.canonical_questions}
    assert texts == {"Mousy odour of the urne is seen in ---", "Mousy odor of teh urine ---- disease"}
    db.close()


# --- explicit "ambiguous pair remains separate" scenario ---


def test_ambiguous_pair_remains_separate_and_is_listed_for_review(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    db.save_questions([q("q1", "What is glycogen?"), q("q2", "Explain glycogen metabolism.")])
    install_fake_embeddings(monkeypatch, {"q1": [1.0, 0.0], "q2": [0.99, 0.14]})
    install_fake_judge(monkeypatch, lambda pid, a, b: PairEquivalenceVerdict(pair_id=pid, verdict="uncertain", confidence=0.5))

    pipeline_module.run_deduplication(CONFIG, db)

    assert db.get_canonical_question_id_for_question("q1") != db.get_canonical_question_id_for_question("q2")
    review = db.list_dedup_pair_decisions(verdict="uncertain")
    assert len(review) == 1
    assert {review[0].question_id_a, review[0].question_id_b} == {"q1", "q2"}
    db.close()


# --- resumability ---


def test_resume_reuses_cached_llm_decisions_and_never_recalls_the_llm(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    db.save_questions([q("q1", "Explain glycolysis."), q("q2", "Describe the process of glycolysis.")])
    install_fake_embeddings(monkeypatch, {"q1": [1.0, 0.0], "q2": [0.99, 0.14]})
    judge_calls = install_fake_judge(monkeypatch, lambda pid, a, b: PairEquivalenceVerdict(pair_id=pid, verdict="same_question", confidence=0.9))

    first = pipeline_module.run_deduplication(CONFIG, db)
    assert judge_calls["n"] == 1
    assert first.llm_pairs_newly_judged == 1

    second = pipeline_module.run_deduplication(CONFIG, db)
    assert judge_calls["n"] == 1  # no new LLM call on rerun
    assert second.llm_pairs_reused_from_cache == 1
    assert second.llm_pairs_newly_judged == 0
    assert second.canonical_group_count == first.canonical_group_count
    db.close()


# --- LLM call failure: never crashes, never fabricates ---


def test_llm_call_failure_marks_pairs_uncertain_and_run_continues(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    db.save_questions([q("q1", "Explain glycolysis."), q("q2", "Describe the process of glycolysis.")])
    install_fake_embeddings(monkeypatch, {"q1": [1.0, 0.0], "q2": [0.99, 0.14]})

    def failing_judge(pairs, config):
        raise LLMCallFailed("retries exhausted")

    monkeypatch.setattr(pipeline_module, "judge_candidate_pairs", failing_judge)

    result = pipeline_module.run_deduplication(CONFIG, db)

    assert result.llm_batches_failed == 1
    assert result.canonical_group_count == 2  # never merged on a failed judgment
    decisions = db.list_dedup_pair_decisions(verdict="uncertain")
    assert len(decisions) == 1
    assert "LLM call failed" in decisions[0].reason
    db.close()


def test_quota_exhausted_propagates_instead_of_being_marked_uncertain(tmp_path, monkeypatch):
    """Unlike an ordinary LLM failure (see the test above, which marks the
    batch "uncertain" and keeps going), a quota-exhausted failure (see
    LLMCallFailed.quota_exhausted) must escape run_deduplication entirely so
    the caller can stop the whole run instead of burning through every
    remaining candidate batch against an exhausted request budget."""
    db = Database(tmp_path / "t.db")
    db.save_questions([q("q1", "Explain glycolysis."), q("q2", "Describe the process of glycolysis.")])
    install_fake_embeddings(monkeypatch, {"q1": [1.0, 0.0], "q2": [0.99, 0.14]})

    def quota_exhausted_judge(pairs, config):
        raise LLMCallFailed("quota exhausted", quota_exhausted=True)

    monkeypatch.setattr(pipeline_module, "judge_candidate_pairs", quota_exhausted_judge)

    with pytest.raises(LLMCallFailed) as exc_info:
        pipeline_module.run_deduplication(CONFIG, db)

    assert exc_info.value.quota_exhausted is True
    # No pair decision was persisted for this batch -- resume retries it cleanly.
    assert db.list_dedup_pair_decisions(verdict="uncertain") == []
    db.close()


# --- invalid semantic judgment (unknown pair_id / missing verdict) never corrupts state ---


def test_invalid_llm_response_never_silently_merges(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    db.save_questions([
        q("q1", "Explain glycolysis."), q("q2", "Describe the process of glycolysis."),
        q("q3", "What is glycogen?"), q("q4", "Explain glycogen metabolism."),
    ])
    install_fake_embeddings(monkeypatch, {
        "q1": [1.0, 0.0, 0.0, 0.0], "q2": [0.99, 0.14, 0.0, 0.0],
        "q3": [0.0, 0.0, 1.0, 0.0], "q4": [0.0, 0.0, 0.99, 0.14],
    })

    def bogus_judge(pairs, config):
        # returns a verdict for a pair_id that was never asked about, and
        # OMITS a verdict for one that was -- both must be handled safely.
        real_pair_ids = [pid for pid, _, _ in pairs]
        verdicts = [PairEquivalenceVerdict(pair_id="not_a_real_pair_id", verdict="same_question", confidence=0.99)]
        if real_pair_ids:
            # only answer the FIRST of the (possibly two) candidate pairs, never the rest
            verdicts.append(PairEquivalenceVerdict(pair_id=real_pair_ids[0], verdict="same_question", confidence=0.9))
        return SemanticEquivalenceResult(verdicts=verdicts)

    monkeypatch.setattr(pipeline_module, "judge_candidate_pairs", bogus_judge)

    result = pipeline_module.run_deduplication(CONFIG, db)

    # the invented pair_id must never appear as a persisted decision
    all_decisions = db.list_dedup_pair_decisions()
    assert all(
        {d.question_id_a, d.question_id_b} <= {"q1", "q2", "q3", "q4"} for d in all_decisions
    )
    # whichever pair got no verdict back defaults to uncertain, never a silent merge
    assert result.canonical_group_count >= 3
    db.close()


# --- empty corpus ---


def test_empty_corpus_produces_an_empty_result(tmp_path):
    db = Database(tmp_path / "t.db")
    result = pipeline_module.run_deduplication(CONFIG, db)
    assert result.total_questions == 0
    assert result.canonical_group_count == 0
    db.close()
