"""Deterministic exact-duplicate grouping + embedding-based candidate pair
generation -- Stage 4's first two steps, entirely LLM-free.

A pair of questions only ever reaches the LLM judge (semantic_judge.py) if:
  1. it is NOT already an exact/near-exact duplicate (those are merged here,
     deterministically, with zero LLM calls -- see group_exact_duplicates),
  2. it scores at/above Config.embedding_similarity_threshold cosine
     similarity (same question_type) or the stricter
     Config.embedding_cross_type_similarity_threshold (different
     question_type) -- see generate_candidate_pairs.

This keeps the LLM call count bounded by candidate pairs, not O(n^2) over
the whole corpus (see project notes: "avoid O(n^2) LLM calls when users
eventually provide thousands of questions"). The embedding cosine-similarity
pass itself IS still O(n^2) comparisons -- that's fine at this stage's
scale (local vector math over a few hundred/thousand questions), it is the
LLM calls specifically that must stay bounded.
"""
from __future__ import annotations

import math
from itertools import combinations

from exampapersorter.config import Config
from exampapersorter.deduplication.normalize import exact_match_key
from exampapersorter.schemas import DedupPairDecision, Question, QuestionEmbedding


def ordered_pair(a: str, b: str) -> tuple[str, str]:
    """Canonical (lexicographically ordered) pair ordering -- every table/
    cache keyed on a pair of question_ids uses this so the same unordered
    pair always produces the same key regardless of which order it was
    discovered in."""
    return (a, b) if a < b else (b, a)


def group_exact_duplicates(questions: list[Question]) -> dict[str, list[Question]]:
    """Groups questions whose normalized (question_type, stem, options) key
    is byte-identical (see normalize.exact_match_key). Returns
    {key: [questions...]}; a group of size 1 just means that question had
    no exact duplicate. This is the ONLY merge path in Stage 4 that
    involves neither an embedding comparison nor an LLM call."""
    groups: dict[str, list[Question]] = {}
    for q in questions:
        groups.setdefault(exact_match_key(q), []).append(q)
    return groups


def exact_duplicate_decisions(groups: dict[str, list[Question]]) -> list[DedupPairDecision]:
    """One DedupPairDecision per pair WITHIN each exact-duplicate group of
    size >= 2, for the audit trail (see database.dedup_pair_decisions).
    Every pair inside a group gets its own record -- not just enough edges
    to connect the group -- so the audit trail directly answers "why are
    these two the same" for any two members, not only the ones that happen
    to form a spanning tree."""
    decisions: list[DedupPairDecision] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        ids = sorted({m.question_id for m in members})
        for a, b in combinations(ids, 2):
            decisions.append(
                DedupPairDecision(
                    question_id_a=a, question_id_b=b, decision_source="exact_duplicate",
                    verdict="same_question", confidence=1.0,
                    reason="Normalized question text (and options, and question_type) are identical",
                )
            )
    return decisions


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def generate_candidate_pairs(
    questions: list[Question],
    embeddings_by_id: dict[str, QuestionEmbedding],
    config: Config,
) -> list[tuple[str, str, float]]:
    """Pairwise cosine similarity over `questions` (the caller passes one
    representative per exact-duplicate group, never every raw occurrence --
    see pipeline.py), returning (question_id_a, question_id_b, similarity)
    -- ordered via ordered_pair -- for every pair clearing the relevant
    threshold.

    High similarity is a candidate-generation signal ONLY, never a
    confirmed duplicate -- every pair returned here still goes to the LLM
    judge (semantic_judge.py) for the actual same-question decision."""
    candidates: list[tuple[str, str, float]] = []
    for qa, qb in combinations(questions, 2):
        emb_a = embeddings_by_id.get(qa.question_id)
        emb_b = embeddings_by_id.get(qb.question_id)
        if emb_a is None or emb_b is None:
            continue
        similarity = _cosine_similarity(emb_a.vector, emb_b.vector)
        threshold = (
            config.embedding_similarity_threshold
            if qa.question_type == qb.question_type
            else config.embedding_cross_type_similarity_threshold
        )
        if similarity >= threshold:
            a_id, b_id = ordered_pair(qa.question_id, qb.question_id)
            candidates.append((a_id, b_id, similarity))
    return candidates
