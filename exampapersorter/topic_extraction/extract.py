"""LLM call: turn confirmed TOC/index evidence into a structured topic list.

The evidence handed to this call is everything Docling extracted from the
page range the search step confirmed -- which may include more than one
contents-like listing (a master table of contents plus a shorter
per-section one, as in our real test textbook). The prompt tells the
model to prefer the most complete listing and use any partial ones only
to sanity-check names, rather than trying to merge them by rule here in
code -- reconciling "which of these two listings is the truth" is exactly
the kind of judgment call that belongs to the LLM, not deterministic code.
"""
from __future__ import annotations

from exampapersorter.config import Config
from exampapersorter.llm_client import call_structured
from exampapersorter.schemas import PageRangeEvidence, TopicExtractionResult
from exampapersorter.topic_extraction.evidence_format import render_evidence, select_topic_extraction_evidence

SYSTEM_PROMPT = """You are extracting the topic/chapter structure of a textbook from its own \
table of contents or index, exactly as evidence.

Rules:
- Preserve the textbook's own wording for topic/chapter names. Do not replace them with synonyms \
or general-knowledge terminology.
- If the evidence shows a hierarchy (e.g. sections containing chapters, or chapters containing \
subtopics), represent it with `level` and `parent_id`. `level` starts at 1 for the top level. \
If there is no real hierarchy, use a flat list where every topic has level=1 and parent_id=null.
- Each topic's `id` must be a short unique machine-friendly slug you invent (e.g. "chapter_1", \
"chapter_1_1"). `parent_id` must reference another topic's `id` in the same output, or be null.
- `source_pages` must be the PDF page number(s) where the evidence block that established this \
topic came from (the [page N] tag on the evidence you were given), never invented.
- `declared_page_number` is DIFFERENT from `source_pages`: it is the book's own printed page \
number for where this topic starts, if the contents/index listing shows one (e.g. the number \
printed at the end of a contents row, like the "244" in "Lipid Metabolism ... 244"). This is the \
textbook's internal page numbering, not a PDF page index. ALWAYS set this whenever the evidence \
shows a page number for this specific topic, REGARDLESS of whether you consider the name resolved \
-- it may be needed later even for a topic you're confident about. Never invent it; leave it null \
only if no page number is associated with this topic in the evidence at all.
- If the evidence contains more than one contents-like listing (e.g. a master table of contents \
and a shorter section-specific one), use the most complete listing that covers the whole book as \
your primary source for which topics exist and their order/hierarchy.
- For EVERY topic, set `resolution_status`:
  - "resolved" if you found a clear, legible name for it. Set `name_evidence_source` to "master_toc" \
if that name was directly legible in the primary listing, or "section_toc" if you had to read or \
confirm it from a separate, shorter listing also present in the evidence (e.g. correcting garbled \
letter-spaced OCR text like "B i o m o l ecu l es" using a clean title for the same topic found \
elsewhere in the evidence). Set `name_source_pages` to the page(s) of whichever block you actually \
read the accepted name from. If a shorter listing gives you the clean name, put that clean name \
directly in `name` -- do NOT create a separate child topic just to hold the corrected name.
  - "name_unresolved" if you can tell the topic exists (e.g. you can see its number/position) but \
no evidence available to you gives a confident, legible name for it. In this case, set `name` to \
whatever identifier is available (e.g. "12" or "Chapter 12"), and leave `name_evidence_source` null \
and `name_source_pages` empty. Do NOT guess a plausible-sounding title.
  - "ambiguous" only if you see multiple conflicting possible names for the same topic and cannot \
tell which is correct.
- Do not include front matter (preface, acknowledgements, title page), appendices, index, or \
answer keys as topics unless the textbook's own contents listing presents them as chapters/topics \
in the main sequence.
- To keep the response shorter: if resolution_status="resolved" via "master_toc", you may omit \
resolution_status, name_evidence_source, and name_source_pages (they default to exactly that \
state) -- but ALWAYS include declared_page_number when the evidence shows one, per the rule above; \
it does not have a meaningful default and omitting it loses real information.

Respond only with the structured topic list."""


def extract_topics(evidence: PageRangeEvidence, config: Config) -> TopicExtractionResult:
    filtered_blocks = select_topic_extraction_evidence(evidence.blocks)
    user_prompt = (
        f"Evidence from PDF pages {evidence.start_page}-{evidence.end_page}, confirmed to contain "
        f"the textbook's table of contents / index (dense paragraph prose has been filtered out; "
        f"you are shown titles, headings, list items, captions, and tables):\n\n"
        f"{render_evidence(filtered_blocks)}\n\n"
        "Extract the full topic/chapter list."
    )
    return call_structured(config, SYSTEM_PROMPT, user_prompt, TopicExtractionResult)
