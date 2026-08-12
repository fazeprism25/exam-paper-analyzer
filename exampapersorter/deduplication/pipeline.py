"""Orchestrates Stage 4 end to end: semantic deduplication over every
already-extracted, already-classified question occurrence (Stage 2/3
output, already persisted) -- corpus-wide, not per-paper, since a duplicate
question can appear in any exam paper regardless of which file it came
from.

Deliberately does NOT re-run Stages 1-3 -- all are read from the database
as already-completed prerequisites (mirrors topic_classification/
pipeline.py's own relationship to Stages 1-2).

Resumable at two granularities:
  - embeddings (Database.get_question_embedding / see embeddings.py) --
    a crash after computing 500 of 900 embeddings never recomputes those
    500 on rerun.
  - pair decisions (Database.get_dedup_pair_decision) -- a crash mid-batch
    of LLM judgments never re-asks about (re-bills) a pair that already has
    a cached verdict.

The final canonical-grouping assembly itself is cheap, pure, deterministic
computation over already-cached decisions, so it is always fully
recomputed from the current decision set and written via
Database.replace_canonical_state -- never incrementally patched -- which is
what keeps a resumed/rerun result always internally consistent (see
grouping.py's module docstring).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from exampapersorter.config import Config
from exampapersorter.database import Database
from exampapersorter.deduplication import candidates, embeddings, grouping
from exampapersorter.deduplication.reconciliation import reconcile_pair_verdicts
from exampapersorter.deduplication.semantic_judge import judge_candidate_pairs
from exampapersorter.llm_client import LLMCallFailed, get_last_call_metadata
from exampapersorter.schemas import CanonicalQuestion, DedupPairDecision

logger = logging.getLogger(__name__)


def _call_metadata() -> dict | None:
    meta = get_last_call_metadata()
    return meta.to_dict() if meta else None


@dataclass
class DeduplicationResult:
    total_questions: int = 0
    exact_duplicate_groups: int = 0
    candidate_pairs_generated: int = 0
    llm_pairs_reused_from_cache: int = 0
    llm_pairs_newly_judged: int = 0
    llm_batches_failed: int = 0
    canonical_group_count: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    canonical_questions: list[CanonicalQuestion] = field(default_factory=list)


def _chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def run_deduplication(config: Config, db: Database) -> DeduplicationResult:
    questions = db.get_all_questions()
    if not questions:
        return DeduplicationResult()

    questions_by_id = {q.question_id: q for q in questions}
    result = DeduplicationResult(total_questions=len(questions))

    # --- Step 1: deterministic exact-duplicate grouping (no embeddings, no LLM) ---
    exact_groups = candidates.group_exact_duplicates(questions)
    exact_decisions = candidates.exact_duplicate_decisions(exact_groups)
    for d in exact_decisions:
        db.save_dedup_pair_decision(d)
    result.exact_duplicate_groups = sum(1 for members in exact_groups.values() if len(members) > 1)
    logger.info(
        "Exact-duplicate grouping: %d question(s) collapsed into %d exact-duplicate group(s) "
        "(%d group(s) with more than one occurrence)",
        len(questions), len(exact_groups), result.exact_duplicate_groups,
    )

    # One representative per exact-duplicate group carries the group forward
    # into embedding/candidate generation -- its exact-dup siblings are
    # already fully resolved and don't need their own candidate comparisons.
    representatives = [
        grouping.select_representative([questions_by_id[q.question_id] for q in members])
        for members in exact_groups.values()
    ]

    # --- Step 2: persistent, resumable local embeddings ---
    embeddings_by_id = embeddings.get_or_compute_embeddings(representatives, config, db)

    # --- Step 3: embedding-based candidate pair generation ---
    candidate_pairs = candidates.generate_candidate_pairs(representatives, embeddings_by_id, config)
    result.candidate_pairs_generated = len(candidate_pairs)
    logger.info("Generated %d candidate pair(s) for semantic-equivalence judging", len(candidate_pairs))

    # --- Step 4: resumable LLM semantic-equivalence judging ---
    all_pair_decisions: list[DedupPairDecision] = list(exact_decisions)
    to_judge: list[tuple[str, str]] = []
    for a_id, b_id, _similarity in candidate_pairs:
        cached = db.get_dedup_pair_decision(a_id, b_id, config.model_identifier)
        if cached is not None:
            all_pair_decisions.append(cached)
            result.llm_pairs_reused_from_cache += 1
        else:
            to_judge.append((a_id, b_id))

    for batch in _chunk(to_judge, config.dedup_pair_batch_size):
        pair_ids = [f"{a}::{b}" for a, b in batch]
        llm_pairs = [(pid, questions_by_id[a], questions_by_id[b]) for pid, (a, b) in zip(pair_ids, batch)]

        try:
            llm_result = judge_candidate_pairs(llm_pairs, config)
        except LLMCallFailed as exc:
            if exc.quota_exhausted:
                raise
            logger.error(
                "Semantic-equivalence judging failed for a batch of %d pair(s): %s -- marking uncertain",
                len(batch), exc,
            )
            result.llm_batches_failed += 1
            for a_id, b_id in batch:
                decision = DedupPairDecision(
                    question_id_a=a_id, question_id_b=b_id, decision_source="llm_judge",
                    verdict="uncertain", confidence=0.0, reason=f"LLM call failed: {exc}",
                )
                db.save_dedup_pair_decision(decision, model_used=config.model_identifier)
                all_pair_decisions.append(decision)
                result.llm_pairs_newly_judged += 1
            continue

        reconciled = reconcile_pair_verdicts(pair_ids, llm_result, config)
        call_metadata = _call_metadata()
        for (a_id, b_id), verdict in zip(batch, reconciled):
            decision = DedupPairDecision(
                question_id_a=a_id, question_id_b=b_id, decision_source="llm_judge",
                verdict=verdict.verdict, confidence=verdict.confidence, reason=verdict.reason,
            )
            db.save_dedup_pair_decision(decision, model_used=config.model_identifier, call_metadata=call_metadata)
            all_pair_decisions.append(decision)
            result.llm_pairs_newly_judged += 1

    # --- Step 5: deterministic canonical grouping + persistence ---
    canonical_questions, links = grouping.build_canonical_questions(questions_by_id, all_pair_decisions)
    db.replace_canonical_state(canonical_questions, links)

    result.canonical_group_count = len(canonical_questions)
    result.canonical_questions = canonical_questions
    for cq in canonical_questions:
        result.status_counts[cq.dedup_status] = result.status_counts.get(cq.dedup_status, 0) + 1

    logger.info(
        "Stage 4 complete: %d question(s) -> %d canonical group(s) (%s)",
        len(questions), result.canonical_group_count, result.status_counts,
    )
    return result
