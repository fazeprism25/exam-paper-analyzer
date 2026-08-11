"""Deterministic canonical-question grouping -- Stage 4's final assembly
step. Given the full set of this run's pair decisions (exact-duplicate ones
from candidates.py, LLM ones reconciled and confidence-gated by
reconciliation.py), builds connected components via union-find over
verdict="same_question" edges and materializes one CanonicalQuestion per
component -- including size-1 components, so every question occurrence
ends up with exactly one canonical group, never left unassigned.

Known, documented limitation (see project notes: "do not over-engineer a
perfect incremental algorithm now"): grouping is the transitive closure of
pairwise "same_question" edges. If A~B and B~C are both confirmed but A~C
was never itself a candidate pair (or came back different_question), A and
C still end up in the same canonical group through B. This is a standard,
accepted limitation of pairwise-judgment clustering, not a bug in this
module -- Stage 4's summary output flags how many groups have size > 2 so a
human reviewer can spot-check the ones most likely affected.
"""
from __future__ import annotations

import hashlib
from itertools import combinations

from exampapersorter.deduplication.candidates import ordered_pair
from exampapersorter.schemas import CanonicalQuestion, DedupPairDecision, Question


class _UnionFind:
    def __init__(self, items: list[str]):
        self._parent = {x: x for x in items}

    def __contains__(self, x: str) -> bool:
        return x in self._parent

    def find(self, x: str) -> str:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic tie-break (lexicographically smaller root wins)
            # so the resulting grouping never depends on decision-list
            # iteration order -- reruns over unchanged input are stable.
            root, child = (ra, rb) if ra < rb else (rb, ra)
            self._parent[child] = root


def build_groups(question_ids: list[str], merge_decisions: list[DedupPairDecision]) -> dict[str, list[str]]:
    """Every question_id in `question_ids` ends up in exactly one group --
    singletons included. `merge_decisions` must already be confidence-gated
    (see reconciliation.py): only decisions the caller actually wants
    unioned should be passed in (verdict="same_question" only; never
    "different_question"/"uncertain")."""
    uf = _UnionFind(question_ids)
    for d in merge_decisions:
        if d.question_id_a in uf and d.question_id_b in uf:
            uf.union(d.question_id_a, d.question_id_b)

    groups: dict[str, list[str]] = {}
    for qid in question_ids:
        root = uf.find(qid)
        groups.setdefault(root, []).append(qid)
    return groups


def select_representative(members: list[Question]) -> Question:
    """Deterministic preference order for which source occurrence's text
    becomes canonical_question_text: prefer a non-ambiguous occurrence, then
    the highest extraction_confidence, then the longest (most complete)
    text, then the lexicographically smallest question_id as a final, fully
    deterministic tiebreaker. Never an LLM rewrite or paraphrase -- always
    one real, verbatim source occurrence, per project notes ("preserve one
    source occurrence as representative")."""
    return min(
        members,
        key=lambda q: (q.ambiguous, -q.extraction_confidence, -len(q.question_text), q.question_id),
    )


def _canonical_id(question_ids: list[str]) -> str:
    """Deterministic and stable across runs: the same set of member
    question_ids always produces the same canonical_question_id, so
    rerunning Stage 4 over an unchanged corpus doesn't reassign new ids to
    unchanged groups."""
    digest = hashlib.sha256("|".join(sorted(question_ids)).encode("utf-8")).hexdigest()[:16]
    return f"canonical_{digest}"


def build_canonical_questions(
    questions_by_id: dict[str, Question], all_pair_decisions: list[DedupPairDecision]
) -> tuple[list[CanonicalQuestion], dict[str, str]]:
    """`all_pair_decisions`: every pair decision Stage 4 made this run --
    exact-duplicate decisions (always verdict=same_question) plus every
    reconciled LLM verdict, INCLUDING different_question/uncertain ones (not
    just the ones that caused a merge) -- the non-merging ones are still
    needed here to flag an otherwise-ordinary singleton as needs_review.

    Returns (canonical_questions, links) where `links` maps every
    question_id -> its canonical_question_id, ready for
    Database.replace_canonical_state -- computed in the same pass as the
    groups themselves so the two can never drift apart."""
    question_ids = sorted(questions_by_id)
    merge_decisions = [d for d in all_pair_decisions if d.verdict == "same_question"]
    groups = build_groups(question_ids, merge_decisions)

    decisions_by_pair: dict[tuple[str, str], DedupPairDecision] = {
        ordered_pair(d.question_id_a, d.question_id_b): d for d in all_pair_decisions
    }

    # question_id -> uncertain-verdict decisions touching it (used only to
    # flag an otherwise-plain singleton group as needs_review).
    uncertain_by_question: dict[str, list[DedupPairDecision]] = {}
    for d in all_pair_decisions:
        if d.verdict == "uncertain":
            uncertain_by_question.setdefault(d.question_id_a, []).append(d)
            uncertain_by_question.setdefault(d.question_id_b, []).append(d)

    results: list[CanonicalQuestion] = []
    links: dict[str, str] = {}
    for member_ids in groups.values():
        member_ids = sorted(member_ids)
        members = [questions_by_id[qid] for qid in member_ids]
        representative = select_representative(members)

        if len(member_ids) == 1:
            touching = uncertain_by_question.get(member_ids[0], [])
            if touching:
                status = "needs_review"
                confidence = max(d.confidence for d in touching)
            else:
                status = "singleton"
                confidence = None
        else:
            internal_pairs = [ordered_pair(a, b) for a, b in combinations(member_ids, 2)]
            merged = [
                decisions_by_pair[p] for p in internal_pairs
                if p in decisions_by_pair and decisions_by_pair[p].verdict == "same_question"
            ]
            if merged and all(d.decision_source == "exact_duplicate" for d in merged):
                status = "exact_duplicate"
                confidence = 1.0
            else:
                status = "semantic_merge"
                confidence = min(d.confidence for d in merged)

        canonical_id = _canonical_id(member_ids)
        for qid in member_ids:
            links[qid] = canonical_id
        results.append(
            CanonicalQuestion(
                canonical_question_id=canonical_id,
                canonical_question_text=representative.question_text,
                question_type=representative.question_type,
                topic_id=representative.topic_id,
                topic_name=representative.topic_name,
                representative_question_id=representative.question_id,
                dedup_confidence=confidence,
                dedup_status=status,
            )
        )
    return results, links
