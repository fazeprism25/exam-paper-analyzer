"""Render deterministic evidence blocks into a prompt-friendly string.

Shared by every stage that talks to the LLM: the LLM never sees Docling's
raw objects, only this rendering. Stage-specific block *selection* (which
blocks to include at all) lives with each stage instead -- what counts as
"high signal" for a textbook table of contents is not the same as what
counts as high signal for an exam paper's structure, so that judgment call
does not belong in this shared module.
"""
from __future__ import annotations

import logging

from exampapersorter.schemas import EvidenceBlock

logger = logging.getLogger(__name__)

# Sized against config.llm_num_ctx (8192 tokens by default). Ollama does NOT
# auto-scale context to what a model supports -- it must be requested
# explicitly (see llm_client.call_structured), and a larger context window
# costs more RAM for the KV cache. ~3.5 chars/token for English text, minus
# headroom for the system prompt and the model's response, land this budget
# comfortably under 8192 tokens.
#
# A truncation bug here previously caused a real, silent failure: on a page
# range where dense front-matter prose ran ~19000 chars before the actual
# table-of-contents content, the TOC block got cut off after ~600 of its
# ~3600 characters, and the LLM correctly reported "neither" because it
# never saw the content. Two lessons from that: truncation must never be
# silent (see the warning log below), and evidence should be filtered to
# high-signal blocks before it's rendered at all -- see the stage-specific
# selection functions that call this.
MAX_EVIDENCE_CHARS = 20000

# A table of contents or index is built from titles, headings, list items,
# and tables -- never from paragraph prose. Dropping dense "text" blocks
# before rendering is a deterministic, defensible filter (not a judgment
# call) for THAT use case -- see topic_extraction/evidence_format.py for
# where this is actually applied. Kept here because it's a generic-enough
# operation (drop one block type) that more than one stage may want it,
# not because "text is always noise" is a universal truth.
_LOW_SIGNAL_BLOCK_TYPES = {"text"}


def filter_high_signal_blocks(blocks: list[EvidenceBlock]) -> list[EvidenceBlock]:
    return [b for b in blocks if b.block_type not in _LOW_SIGNAL_BLOCK_TYPES]


def render_evidence(blocks: list[EvidenceBlock]) -> str:
    lines = []
    for block in blocks:
        pages = ",".join(str(p) for p in block.page_numbers)
        # Docling frequently parses a list item's marker (e.g. "A.") out of
        # `text` entirely rather than leaving it concatenated -- confirmed
        # directly against a real fixture, where an MCQ option's `text` was
        # bare "Obstructive Jaundice" with "A." living only in `marker`.
        # Re-attach it for display so the LLM sees what a human reader
        # would ("A. Obstructive Jaundice"), even for list items our own
        # deterministic mcq_grouping didn't bundle -- this is the fallback
        # layer, not the primary fix (see mcq_grouping.py).
        text = f"{block.marker} {block.text}" if block.marker else block.text
        lines.append(f"[page {pages}] ({block.block_type}) {text}")
    rendered = "\n".join(lines)

    if len(rendered) > MAX_EVIDENCE_CHARS:
        logger.warning(
            "Evidence rendering truncated from %d to %d chars -- content past this point "
            "was NOT shown to the LLM. If detection/extraction misses content, this is why.",
            len(rendered), MAX_EVIDENCE_CHARS,
        )
        rendered = rendered[:MAX_EVIDENCE_CHARS] + "\n...[evidence truncated]..."

    return rendered
