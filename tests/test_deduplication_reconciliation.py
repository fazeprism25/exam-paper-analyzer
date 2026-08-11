from dataclasses import replace

from exampapersorter.config import DEFAULT_CONFIG
from exampapersorter.deduplication.reconciliation import reconcile_pair_verdicts
from exampapersorter.schemas import PairEquivalenceVerdict, SemanticEquivalenceResult


def test_reconcile_passes_through_confident_same_question_verdict():
    result = SemanticEquivalenceResult(
        verdicts=[PairEquivalenceVerdict(pair_id="q1::q2", verdict="same_question", confidence=0.9)]
    )
    reconciled = reconcile_pair_verdicts(["q1::q2"], result, DEFAULT_CONFIG)
    assert reconciled[0].verdict == "same_question"
    assert reconciled[0].confidence == 0.9


def test_reconcile_passes_through_different_question_and_uncertain_unchanged():
    result = SemanticEquivalenceResult(
        verdicts=[
            PairEquivalenceVerdict(pair_id="q1::q2", verdict="different_question", confidence=0.9),
            PairEquivalenceVerdict(pair_id="q3::q4", verdict="uncertain", confidence=0.4),
        ]
    )
    reconciled = reconcile_pair_verdicts(["q1::q2", "q3::q4"], result, DEFAULT_CONFIG)
    by_id = {v.pair_id: v for v in reconciled}
    assert by_id["q1::q2"].verdict == "different_question"
    assert by_id["q3::q4"].verdict == "uncertain"


def test_missing_verdict_defaults_to_uncertain_not_a_silent_merge():
    result = SemanticEquivalenceResult(verdicts=[])
    reconciled = reconcile_pair_verdicts(["q1::q2"], result, DEFAULT_CONFIG)
    assert len(reconciled) == 1
    assert reconciled[0].verdict == "uncertain"
    assert reconciled[0].confidence == 0.0


def test_verdict_for_pair_not_asked_about_is_discarded():
    result = SemanticEquivalenceResult(
        verdicts=[
            PairEquivalenceVerdict(pair_id="q1::q2", verdict="same_question", confidence=0.9),
            PairEquivalenceVerdict(pair_id="q_not_asked::q_other", verdict="same_question", confidence=0.9),
        ]
    )
    reconciled = reconcile_pair_verdicts(["q1::q2"], result, DEFAULT_CONFIG)
    assert len(reconciled) == 1
    assert reconciled[0].pair_id == "q1::q2"


def test_low_confidence_same_question_is_downgraded_to_uncertain():
    """A same_question verdict below the configured merge floor must never
    reach grouping.py as a mergeable edge -- this is the deterministic
    defense-in-depth gate (Config.dedup_min_merge_confidence)."""
    config = replace(DEFAULT_CONFIG, dedup_min_merge_confidence=0.6)
    result = SemanticEquivalenceResult(
        verdicts=[PairEquivalenceVerdict(pair_id="q1::q2", verdict="same_question", confidence=0.4, reason="looked similar")]
    )
    reconciled = reconcile_pair_verdicts(["q1::q2"], result, config)
    assert reconciled[0].verdict == "uncertain"
    assert reconciled[0].confidence == 0.4
    assert "looked similar" in reconciled[0].reason
    assert "0.4" in reconciled[0].reason


def test_high_confidence_same_question_clears_the_merge_floor():
    config = replace(DEFAULT_CONFIG, dedup_min_merge_confidence=0.6)
    result = SemanticEquivalenceResult(
        verdicts=[PairEquivalenceVerdict(pair_id="q1::q2", verdict="same_question", confidence=0.9)]
    )
    reconciled = reconcile_pair_verdicts(["q1::q2"], result, config)
    assert reconciled[0].verdict == "same_question"


def test_reconcile_preserves_order_of_requested_pair_ids():
    result = SemanticEquivalenceResult(
        verdicts=[PairEquivalenceVerdict(pair_id="q3::q4", verdict="same_question", confidence=0.9)]
    )
    reconciled = reconcile_pair_verdicts(["q1::q2", "q3::q4"], result, DEFAULT_CONFIG)
    assert [v.pair_id for v in reconciled] == ["q1::q2", "q3::q4"]
