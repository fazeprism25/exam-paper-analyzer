"""Classify already-extracted exam questions (Stage 2 output) against a
textbook's topic list (Stage 1 output) -- Stage 3.

One LLM call per exam paper, batching all of that paper's questions
against the FULL topic list (every level, not just leaves), so the model
can pick whichever granularity actually fits. Mirrors Stage 2's per-paper
call granularity (see question_extraction/paper_extraction.py) for the
same reasons: keeps each call's token budget bounded to one paper's worth
of content, and keeps caching/resumability at the same granularity Stage 2
already uses (see topic_classification/pipeline.py).

Deterministic reconciliation (topic_classification/reconciliation.py)
handles what the schema alone can't guarantee: that every returned
topic_id is actually one this call was given (never trust an invented
id), and that every question asked about got a verdict (never silently
drop one the response omitted).
"""
from __future__ import annotations

import logging

from exampapersorter.config import Config
from exampapersorter.llm_client import call_structured
from exampapersorter.schemas import Question, Topic, TopicClassificationResult

logger = logging.getLogger(__name__)

# Sized the same way as evidence_format.MAX_EVIDENCE_CHARS -- a safety net
# against an unbounded topic list blowing up the prompt, not a claim that
# any real textbook approaches this. Truncation is logged, never silent
# (see evidence_format.render_evidence's own comment on why that matters).
MAX_TOPIC_LIST_CHARS = 20000

SYSTEM_PROMPT = """You are classifying exam questions against a textbook's topic list, so each question can \
later be grouped by the syllabus topic it tests.

You are given: (a) a hierarchical topic list from the textbook this exam is presumably based on, each \
entry showing its topic_id, indentation depth, and name, and (b) a batch of exam questions, each with a \
question_id.

For EACH question given, report exactly one verdict:
- status="classified": the question clearly tests one identifiable topic. Set topic_id to that topic's \
EXACT id string from the list given -- never invent an id, never alter one, never guess an id that looks \
plausible but wasn't shown to you. Prefer the MOST SPECIFIC (deepest) topic that genuinely fits over a \
broad parent topic, but only if you are actually confident in that specific pick.
- status="no_match": none of the given topics are a plausible fit for this question (e.g. it's general \
knowledge, or about a subject area the topic list doesn't cover). Leave topic_id null.
- status="ambiguous": more than one topic is plausible and you cannot confidently choose between them. \
Leave topic_id null and briefly name the competing topics in `reason`.

Always include a `confidence` (0-1) reflecting how sure you are of the verdict itself (including for \
no_match/ambiguous -- confidence there means how sure you are that no single topic fits / that it's \
genuinely ambiguous, not confidence in a topic pick). Set `reason` briefly for no_match and ambiguous \
verdicts; it may be null for a confident classified verdict.

Return a verdict for EVERY question_id given, echoing it back exactly. Never invent a question_id, and \
never fabricate a topic_id that wasn't in the list you were shown."""


def _render_topics(topics: list[Topic]) -> str:
    by_parent: dict[str | None, list[Topic]] = {}
    for t in topics:
        by_parent.setdefault(t.parent_id, []).append(t)

    lines: list[str] = []

    def walk(parent_id: str | None, depth: int) -> None:
        for t in by_parent.get(parent_id, []):
            lines.append(f"{'  ' * depth}- [{t.id}] {t.name}")
            walk(t.id, depth + 1)

    walk(None, 0)
    rendered = "\n".join(lines)

    if len(rendered) > MAX_TOPIC_LIST_CHARS:
        logger.warning(
            "Topic list rendering truncated from %d to %d chars -- topics past this point "
            "were NOT shown to the classifier.", len(rendered), MAX_TOPIC_LIST_CHARS,
        )
        rendered = rendered[:MAX_TOPIC_LIST_CHARS] + "\n...[topic list truncated]..."
    return rendered


def _render_questions(questions: list[Question]) -> str:
    lines: list[str] = []
    for q in questions:
        header = f"[{q.question_id}] ({q.question_type})"
        if q.question_number:
            header += f" #{q.question_number}"
        lines.append(header)
        lines.append(q.question_text)
        for i, opt in enumerate(q.options):
            lines.append(f"  {chr(ord('A') + i)}. {opt}")
        lines.append("")
    return "\n".join(lines)


def classify_questions(questions: list[Question], topics: list[Topic], config: Config) -> TopicClassificationResult:
    user_prompt = (
        f"Topic list ({len(topics)} topics):\n\n{_render_topics(topics)}\n\n"
        f"Questions to classify ({len(questions)}):\n\n{_render_questions(questions)}\n\n"
        "Classify every question listed above."
    )
    return call_structured(
        config, SYSTEM_PROMPT, user_prompt, TopicClassificationResult, call_label="topic_classification"
    )
