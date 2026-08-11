"""Render deterministic evidence blocks into a prompt-friendly string.

Kept separate from docling_processing.py: that module is about extracting
evidence, this one is about presenting it to the LLM. The LLM never sees
Docling's raw objects, only this rendering.

render_evidence/filter_high_signal_blocks/MAX_EVIDENCE_CHARS now live in the
top-level exampapersorter.evidence_format module (Stage 2 needs them too,
and they were never actually textbook-specific) -- re-exported here so
every existing import of this module keeps working unchanged.
select_topic_extraction_evidence below IS genuinely Stage-1-specific
(the document_index/table/header three-tier fallback only makes sense for
"find the table of contents"), so it stays here.
"""
from __future__ import annotations

from exampapersorter.evidence_format import MAX_EVIDENCE_CHARS, filter_high_signal_blocks, render_evidence
from exampapersorter.schemas import EvidenceBlock

__all__ = [
    "MAX_EVIDENCE_CHARS",
    "filter_high_signal_blocks",
    "render_evidence",
    "select_topic_extraction_evidence",
]

# A table of contents or index is built from titles, headings, list items,
# and tables -- never from paragraph prose. Dropping dense "text" blocks
# before rendering is a deterministic, defensible filter (not a judgment
# call), and it matters in practice: on our real test textbook, a 25-page
# window had 293 evidence blocks, 173 of them dense paragraph text. That
# volume of irrelevant content reliably distracted the LLM into citing
# unrelated chapter body text instead of the two real document_index
# blocks also present in the same window.


def select_topic_extraction_evidence(blocks: list[EvidenceBlock]) -> list[EvidenceBlock]:
    """Narrow a confirmed TOC page range down to the blocks that should
    actually drive topic extraction.

    Why this exists: filter_high_signal_blocks alone (used for TOC
    detection) still keeps every section_header/list_item in the window --
    fine for "is a TOC present", wrong for "what does the TOC say". On our
    real test textbook, a confirmed 25-page window contained not just the
    2 document_index tables that ARE the table of contents, but also ~55
    section headers belonging to Chapter 1's own body content (e.g.
    "Nucleus", "Mitochondria", "Golgi apparatus"). Handed all of that, the
    LLM tried to extract every section header it saw as if it were a
    legitimate topic, producing a sprawling response that blew through the
    token budget.

    Three-tier fallback, each tier used only if the previous one is empty:
      1. document_index blocks -- Docling's own layout model already
         specifically suspects these ARE a contents/index listing. The
         strongest, most specific signal, so if any exist they're used
         exclusively (ordinary data tables elsewhere in the range are
         noise, not corroborating evidence).
      2. generic table blocks -- for textbooks whose TOC is tabular but
         Docling didn't tag it document_index specifically.
      3. the broader header/list-based filter -- for a TOC/index that
         isn't tabular at all (plain-text contents page, back-of-book
         index).
    """
    document_index_blocks = [b for b in blocks if b.block_type == "document_index"]
    if document_index_blocks:
        return document_index_blocks

    table_blocks = [b for b in blocks if b.block_type == "table"]
    if table_blocks:
        return table_blocks

    return filter_high_signal_blocks(blocks)
