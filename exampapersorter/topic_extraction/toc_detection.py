"""LLM call: does this page range contain a table of contents / index?

This is the only place we ask the LLM a yes/no-ish structural question
about a page range. It must ground its verdict in the actual evidence
text (the `evidence` field of TOCDetectionResult echoes back page numbers
and snippets) -- so a "table_of_contents" verdict with an empty evidence
list is treated as invalid by construction, not just by convention.
"""
from __future__ import annotations

from exampapersorter.config import Config
from exampapersorter.llm_client import call_structured
from exampapersorter.schemas import PageRangeEvidence, TOCDetectionResult
from exampapersorter.topic_extraction.evidence_format import filter_high_signal_blocks, render_evidence

SYSTEM_PROMPT = """You are analyzing pages from a textbook to determine whether they contain \
a table of contents or an index.

A table of contents lists chapters/sections/topics with page numbers, usually near the front \
of the book. An index is an alphabetical list of terms with page numbers, usually near the back. \
Evidence blocks tagged (document_index) come from a layout model that already suspects the block \
is a contents/index table -- this is a strong signal. You must examine every block tagged \
(document_index) carefully before concluding "neither"; do not overlook it in favor of other content.

Dense paragraph text has been deliberately removed from this evidence -- you are only shown \
titles, headings, list items, captions, and tables, because a table of contents/index is never \
built from paragraph prose. Do not conclude "neither" merely because you don't see narrative text.

Neither a title page, preface, list of figures/tables, nor a list of abbreviations counts as a \
table of contents or index.

You must ground your verdict in the supplied evidence: the `evidence` field must contain the \
specific page numbers and text snippets that justify your classification. If you classify as \
"neither", evidence may be a brief note of what the pages actually contain instead.

Respond only with the structured fields requested. Do not invent page numbers or text that is \
not present in the evidence given to you."""


def detect_toc(evidence: PageRangeEvidence, config: Config) -> TOCDetectionResult:
    filtered_blocks = filter_high_signal_blocks(evidence.blocks)
    user_prompt = (
        f"Evidence from PDF pages {evidence.start_page}-{evidence.end_page}:\n\n"
        f"{render_evidence(filtered_blocks)}\n\n"
        "Classify this page range as table_of_contents, index, or neither."
    )
    return call_structured(config, SYSTEM_PROMPT, user_prompt, TOCDetectionResult)
