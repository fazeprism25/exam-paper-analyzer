"""LLM semantic-equivalence judge for Stage 4 candidate pairs.

One call judges a BATCH of candidate pairs (see Config.dedup_pair_batch_size),
mirroring Stage 3's per-paper batching (topic_classification/classify.py) --
callers chunk a larger candidate set across multiple calls rather than
sending one oversized request. Goes through the existing call_structured()
LLM abstraction only; this module never talks to OpenRouter/Ollama
directly, and never introduces a separate model pool (see project notes:
"the SAME 10-model pool must be used for every LLM task").

The model's ONLY question is whether two questions ask for substantially
the same answer/task -- explicitly NOT whether they're about the same
topic (see schemas.PairEquivalenceVerdict's module comment). It is
instructed to be conservative: prefer "uncertain" over a confident-sounding
guess, since a false merge is worse than a missed duplicate.
"""
from __future__ import annotations

from exampapersorter.config import Config
from exampapersorter.llm_client import call_structured
from exampapersorter.schemas import Question, SemanticEquivalenceResult

SYSTEM_PROMPT = """You are judging whether pairs of exam questions are asking for SUBSTANTIALLY THE SAME \
ANSWER OR TASK -- i.e. would a student's correct answer to one also correctly answer the other, and are \
they testing the same specific thing at the same depth?

You are NOT judging whether the two questions are about the same topic or subject area. Two questions can \
share a topic and still be different questions -- for example "Name the rate-limiting enzyme of glycolysis" \
and "Explain the regulation of glycolysis" are both about glycolysis regulation but ask for entirely \
different answers (one wants a single enzyme name, the other wants a full explanation of the regulatory \
mechanism): that pair is different_question, not same_question. Never merge two questions just because they \
mention the same keywords, fact, or subject.

Each question is given with its question_type (mcq, short_answer, long_answer, essay, short_essay, other, \
unknown) and, for MCQs, its answer options. A meaningfully different question_type is evidence AGAINST the \
pair being the same question (a single-fact MCQ and a long-form essay prompt are usually testing different \
things even when they concern the same fact) -- but do not treat a type mismatch as an automatic veto if the \
underlying task is genuinely, unambiguously identical.

You may also be shown each question's topic classification (topic_name). This is CONTEXT ONLY, not a \
verdict input: do not merge because the topics match, and do not refuse to merge because the topics differ \
-- topic classification is itself probabilistic and can be wrong.

Some question text may be OCR-corrupted or otherwise degraded. If you cannot reliably tell whether two \
corrupted/incomplete texts are the same question, say so -- do NOT guess a "corrected" reading of either \
question's text, and do NOT invent content that isn't in the evidence you were given.

For EACH pair given, report exactly one verdict:
- verdict="same_question": you are confident these ask for substantially the same answer/task.
- verdict="different_question": you are confident these do NOT ask for the same answer/task. This includes \
"same topic, different question" cases -- those are different_question, not uncertain.
- verdict="uncertain": you cannot confidently decide either way (including: the text is too corrupted to \
judge reliably). Prefer this over a low-confidence guess in either direction.

Always include a `confidence` (0-1) reflecting how sure you are of the verdict itself. Set `reason` briefly, \
especially for different_question and uncertain verdicts. Set `evidence` to the specific wording in the two \
questions that grounds your verdict.

Return a verdict for EVERY pair_id given, echoing it back exactly. Never invent a pair_id that wasn't shown \
to you."""


def _render_question(q: Question) -> str:
    lines = [f"type={q.question_type}"]
    if q.topic_name:
        lines.append(f"topic={q.topic_name}")
    lines.append(f"text: {q.question_text}")
    for i, opt in enumerate(q.options):
        lines.append(f"  {chr(ord('A') + i)}. {opt}")
    return "\n".join(lines)


def _render_pairs(pairs: list[tuple[str, Question, Question]]) -> str:
    blocks = []
    for pair_id, qa, qb in pairs:
        blocks.append(
            f"[{pair_id}]\nQuestion A ({qa.question_id}):\n{_render_question(qa)}\n\n"
            f"Question B ({qb.question_id}):\n{_render_question(qb)}"
        )
    return "\n\n---\n\n".join(blocks)


def judge_candidate_pairs(pairs: list[tuple[str, Question, Question]], config: Config) -> SemanticEquivalenceResult:
    """`pairs`: list of (pair_id, question_a, question_b), pair_id built by
    the caller as f"{a.question_id}::{b.question_id}" with a < b (see
    pipeline.py / candidates.ordered_pair). One call per batch -- callers
    are responsible for chunking a larger candidate set to
    Config.dedup_pair_batch_size before calling this."""
    user_prompt = (
        f"Candidate pairs to judge ({len(pairs)}):\n\n{_render_pairs(pairs)}\n\n"
        "Judge every pair listed above."
    )
    return call_structured(
        config, SYSTEM_PROMPT, user_prompt, SemanticEquivalenceResult, call_label="semantic_equivalence"
    )
