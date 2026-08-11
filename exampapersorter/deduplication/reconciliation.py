"""Deterministic post-hoc reconciliation between the LLM's semantic-
equivalence response and the (already fixed, already known) batch of
candidate pairs it was asked to judge -- mirrors
topic_classification/reconciliation.py's "never fabricate, never silently
trust" philosophy.

Two things the JSON schema alone cannot guarantee:
  - every pair_id actually asked about ends up with a verdict, even if the
    model's response silently omitted one;
  - a pair_id the response cites that was NOT part of the asked batch is
    never looked up (there is nothing in our own state to reconcile it
    against) -- it is simply dropped, the same treatment an invented
    topic_id gets in Stage 3.

This is also where the deterministic confidence floor
(Config.dedup_min_merge_confidence) is enforced: a verdict of
"same_question" below that floor is downgraded to "uncertain" before it
ever reaches grouping.py, so a single overconfident-sounding-but-actually-
low-confidence LLM judgment can never merge two distinct questions by
itself (see project acceptance criteria: "False/invalid semantic judgments
cannot silently corrupt the database").
"""
from __future__ import annotations

import logging

from exampapersorter.config import Config
from exampapersorter.schemas import PairEquivalenceVerdict, SemanticEquivalenceResult

logger = logging.getLogger(__name__)


def reconcile_pair_verdicts(
    pair_ids: list[str], result: SemanticEquivalenceResult, config: Config
) -> list[PairEquivalenceVerdict]:
    """Returns exactly one PairEquivalenceVerdict per entry in `pair_ids`,
    in that order -- regardless of how many verdicts (if any) the LLM
    response actually contained for each."""
    known_pair_ids = set(pair_ids)
    by_pair_id = {v.pair_id: v for v in result.verdicts}

    unknown_ids = set(by_pair_id) - known_pair_ids
    if unknown_ids:
        logger.warning(
            "Semantic equivalence response cited %d pair_id(s) not among the pairs asked about; discarded: %s",
            len(unknown_ids), sorted(unknown_ids),
        )

    reconciled: list[PairEquivalenceVerdict] = []
    for pair_id in pair_ids:
        v = by_pair_id.get(pair_id)
        if v is None:
            logger.warning("No semantic-equivalence verdict returned for pair %s; marking uncertain", pair_id)
            reconciled.append(
                PairEquivalenceVerdict(
                    pair_id=pair_id, verdict="uncertain", confidence=0.0,
                    reason="LLM response did not include a verdict for this pair",
                )
            )
            continue

        if v.verdict == "same_question" and v.confidence < config.dedup_min_merge_confidence:
            logger.info(
                "Pair %s: verdict=same_question but confidence %.2f is below the merge floor (%.2f); "
                "downgrading to uncertain", pair_id, v.confidence, config.dedup_min_merge_confidence,
            )
            downgraded_reason = f"Below merge-confidence floor (LLM said same_question at {v.confidence:.2f})"
            if v.reason:
                downgraded_reason += f"; {v.reason}"
            reconciled.append(
                PairEquivalenceVerdict(
                    pair_id=pair_id, verdict="uncertain", confidence=v.confidence,
                    reason=downgraded_reason, evidence=v.evidence,
                )
            )
            continue

        reconciled.append(v)

    return reconciled
