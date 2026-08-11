"""Deterministic text normalization for exact-duplicate detection -- Stage 4.

Deliberately conservative (see project notes: "do NOT perform aggressive
text rewriting that could alter meaning"). This is NOT a general text
cleanup utility; it exists solely to make two occurrences that are the same
question, differing only in incidental formatting, compare equal WITHOUT
spending an LLM call on them. Anything beyond whitespace/Unicode/harmless
punctuation normalization belongs to the embedding-based candidate path
(candidates.py) or the LLM judge (semantic_judge.py), never here.
"""
from __future__ import annotations

import re
import unicodedata

from exampapersorter.schemas import Question

_WHITESPACE_RE = re.compile(r"\s+")

# Visually/semantically equivalent punctuation that OCR or different
# typesetting commonly renders inconsistently -- normalized to a single
# canonical form so it never causes a spurious mismatch. Never maps
# punctuation that changes meaning (e.g. "?" and "!" are left alone).
_PUNCT_MAP = {
    "‘": "'", "’": "'",  # curly single quotes
    "“": '"', "”": '"',  # curly double quotes
    "–": "-", "—": "-",  # en/em dash
    "…": "...",  # ellipsis
}


def normalize_for_exact_match(text: str) -> str:
    """Unicode-normalize, harmonize cosmetic punctuation, collapse
    whitespace, strip trailing sentence-ending periods (but not other
    trailing punctuation -- a trailing "?" can matter), and lowercase.
    Two occurrences that normalize to the same string differ only in
    formatting, never in the actual question content, by construction."""
    normalized = unicodedata.normalize("NFKC", text)
    for src, dst in _PUNCT_MAP.items():
        normalized = normalized.replace(src, dst)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    normalized = normalized.rstrip(".")
    return normalized.lower()


def exact_match_key(question: Question) -> str:
    """Grouping key for deterministic exact-duplicate detection. Includes
    question_type (a byte-identical stem extracted once as an MCQ and once
    as, say, a short_answer is a Stage 2 data anomaly, not a confirmed
    duplicate -- see candidates.py) and normalized options (an MCQ with the
    same stem but different answer choices is NOT the same question)."""
    stem = normalize_for_exact_match(question.question_text)
    options = tuple(normalize_for_exact_match(o) for o in question.options)
    return f"{question.question_type}||{stem}||{'|'.join(options)}"
