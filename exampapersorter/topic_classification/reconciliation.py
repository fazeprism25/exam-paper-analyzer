"""Deterministic post-hoc reconciliation between the LLM's classification
response and the (already fixed, already known) set of questions and
topics it was asked about.

Two things the JSON schema alone cannot guarantee, mirroring the "never
fabricate, never silently trust" philosophy used elsewhere (see
question_extraction/reconciliation.py and validation.py):

  - Every topic_id the model cites for a "classified" verdict must
    actually be one of the ids it was shown. A schema can't express "this
    string must be a member of THIS run's dynamic id set" -- only real
    membership-checking after the fact can catch an invented, altered, or
    stale id. A citation that fails this check is downgraded to
    status="unclassified" rather than trusted or silently dropped.

  - Every question actually asked about must end up with a verdict, even
    if the model's response omitted it entirely (a possible failure mode
    independent of schema validity -- the response can be valid JSON and
    still just be short a few entries). Missing questions are filled in as
    status="unclassified" rather than left unrepresented.

A classification for a question_id that was NOT asked about (invented, or
left over from a different paper) is simply never looked up and so never
makes it into the reconciled result -- there is nothing to downgrade,
since we only ever iterate over our own known question list.
"""
from __future__ import annotations

import logging

from exampapersorter.schemas import Question, QuestionTopicClassification, Topic, TopicClassificationResult

logger = logging.getLogger(__name__)


def reconcile_classifications(
    questions: list[Question], topics: list[Topic], result: TopicClassificationResult
) -> list[QuestionTopicClassification]:
    known_topic_ids = {t.id for t in topics}
    known_question_ids = {q.question_id for q in questions}
    by_question_id = {c.question_id: c for c in result.classifications}

    unknown_ids = set(by_question_id) - known_question_ids
    if unknown_ids:
        logger.warning(
            "Topic classification response cited %d question_id(s) not among the questions asked "
            "about; discarded: %s", len(unknown_ids), sorted(unknown_ids),
        )

    reconciled: list[QuestionTopicClassification] = []
    for q in questions:
        c = by_question_id.get(q.question_id)
        if c is None:
            logger.warning("No classification verdict returned for question %s; marking unclassified", q.question_id)
            reconciled.append(
                QuestionTopicClassification(
                    question_id=q.question_id, status="unclassified", topic_id=None, confidence=0.0,
                    reason="LLM response did not include a verdict for this question",
                )
            )
            continue

        if c.status == "classified" and c.topic_id not in known_topic_ids:
            logger.warning(
                "Question %s: LLM cited topic_id %r which is not in the known topic list; discarding",
                q.question_id, c.topic_id,
            )
            reconciled.append(
                QuestionTopicClassification(
                    question_id=q.question_id, status="unclassified", topic_id=None, confidence=0.0,
                    reason=f"LLM cited topic_id {c.topic_id!r}, which is not in the known topic list",
                )
            )
            continue

        reconciled.append(c)

    return reconciled
