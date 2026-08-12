#!/usr/bin/env python
"""Retrieval-assisted Stage 3 classification experiment.

Answers one question: if the classifier is shown actual textbook evidence
for a question's current topic pick vs. a competing candidate topic, does it
correct genuinely wrong verdicts without breaking already-correct ones?
Scoped to the existing 60-question ground-truth evaluation set (see
Biochemistry_Questions_By_Chapter.md / output/baseline_scored.json), not the
full 215-question corpus and not wired into the production `analyze` CLI.

Three phases, run via --phase:
  retrieve   -- local only, no LLM/API key needed. For each distinct topic
                referenced by the eval pairs below, run Docling (cached) over
                that topic's own textbook chapter page range and keep only
                structural evidence (chapter title + headings/subheadings) --
                never full paragraph text, per the "compact evidence bundle"
                requirement. Writes output/retrieval_evidence.json.
  adjudicate -- requires an OpenRouter key (Config.openrouter_api_key /
                OPENROUTER_API_KEY). One call per eval pair: question +
                current topic + rival topic + both topics' textbook evidence
                -> classified(must be one of the two topic_ids)/no_match/
                ambiguous. Never allowed to invent a topic_id outside the two
                given. Writes output/retrieval_adjudication_results.json.
  score      -- local only. Combines baseline_scored.json + adjudication
                results into the final comparison report.

Page-range retrieval design: Stage 1's topics.json carries each topic's own
`declared_page_number` (the book's own printed page number its chapter
starts on, read from the TOC). page_numbering.build_page_number_mapping
(header/footer digit matching) returns None on this textbook -- confirmed
empirically here too, on ordinary body pages, not just the Stage 1 TOC
window -- so the book-page -> PDF-page offset can't be derived that way.
Instead it was calibrated empirically (see conversation record): fetching
PDF pages 238-248 shows content still squarely inside the "Biological
oxidation" chapter (declared_page_number=221) ending in its own
self-assessment questions around book page ~241, i.e. PDF page ~248 -- so
BOOK_TO_PDF_OFFSET ~= 8-11. A generous asymmetric pad around
(declared_page + offset) absorbs the remaining uncertainty; this is a
one-time, cached-per-range calibration, not something a production
integration would need to redo per question.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from exampapersorter.config import DEFAULT_CONFIG, Config
from exampapersorter.database import Database
from exampapersorter.docling_processing import extract_page_range_evidence
from exampapersorter.evidence_format import filter_high_signal_blocks
from exampapersorter.llm_client import LLMCallFailed, call_structured
from exampapersorter.pdf_utils import compute_file_hash
from exampapersorter.schemas import EvidenceBlock, Topic
from pydantic import BaseModel, Field, model_validator
from typing import Literal

TEXTBOOK_PATH = REPO / "Satyanarayan Biochemistry.pdf"
TOPICS_PATH = REPO / "output" / "topics.json"
BASELINE_SCORED_PATH = REPO / "output" / "baseline_scored.json"
EVIDENCE_OUT_PATH = REPO / "output" / "retrieval_evidence.json"
ADJUDICATION_OUT_PATH = REPO / "output" / "retrieval_adjudication_results.json"
FINAL_REPORT_PATH = REPO / "output" / "retrieval_experiment_report.json"

# --- Calibration (see module docstring) ---
BOOK_TO_PDF_OFFSET = 9
WINDOW_PAD_START = 6
WINDOW_PAD_END = 4
MAX_CHAPTER_PAGES = 60  # safety cap; largest real chapter here is ~50 pages

# --- Evidence compactness ---
MAX_HEADINGS_PER_TOPIC = 80
MAX_EVIDENCE_CHARS_PER_TOPIC = 3500


# The 9 known genuine errors + 6 controls, expressed as (gt_index, rival_topic_id).
# gt_index indexes into output/gt_match_results.json / baseline_scored.json
# (0-based, matches Biochemistry_Questions_By_Chapter.md's question order).
# rival_topic_id is the ONE competing candidate shown alongside the question's
# current (baseline) topic_id -- see module docstring for why this experiment
# uses a fixed 2-candidate adjudication rather than building a separate
# candidate-generation mechanism first.
ERROR_PAIRS: dict[int, str] = {
    19: "section_three_chapter_3_3",  # ketone bodies -> should be Metabolism of lipids
    20: "section_three_chapter_3_3",  # ketone bodies -> should be Metabolism of lipids
    22: "section_three_chapter_3_3",  # ketone bodies -> should be Metabolism of lipids
    26: "section_three_chapter_3_3",  # beta-oxidation -> should be Metabolism of lipids
    14: "section_two_chapter_2_3",    # bilirubin -> should be Hemoglobin and porphyrins
    37: "section_three_chapter_3_7",  # fluorine -> should be Mineral metabolism
    42: "section_four_chapter_4_2",   # glomerular function -> should be Organ function tests
    27: "section_three_chapter_3_4",  # urea synthesis -> should be Metabolism of amino acids
    36: "section_three_chapter_3_4",  # urea causes -> should be Metabolism of amino acids
}

# Controls: already-correct baseline verdicts, paired with the SAME rival
# topic that a confusable sibling question in ERROR_PAIRS was wrongly sent
# to -- the most sensitive regression test (same decision boundary, opposite
# direction), rather than arbitrary controls.
CONTROL_PAIRS: dict[int, str] = {
    17: "section_three_chapter_3_2",  # beta-oxidation (correct: lipids) vs carbs -- same boundary as errors 19/20/22/26
    25: "section_three_chapter_3_2",  # ketone bodies (correct: lipids) vs carbs
    43: "section_four_chapter_4_4",   # enzymes in liver disease (correct: organ fn tests) vs tissue proteins/body fluids -- same boundary as error 42
    6: "section_four_chapter_4_2",    # enzyme inhibitions (correct: Enzymes) vs organ fn tests -- same boundary as isoenzymes defensible case
    39: "section_one_chapter_1_7",    # calcium (correct: mineral metabolism) vs vitamins -- same boundary as error 37
    29: "section_four_chapter_4_2",   # urea cycle (correct: amino acids) vs organ fn tests -- same boundary as errors 27/36
}

ALL_PAIRS: dict[int, str] = {**ERROR_PAIRS, **CONTROL_PAIRS}


def load_topics() -> list[Topic]:
    data = json.loads(TOPICS_PATH.read_text(encoding="utf-8"))
    return [Topic.model_validate(t) for t in data["topics"]]


def topic_book_page_ranges(topics: list[Topic]) -> dict[str, tuple[int, int]]:
    """book-page (start, end) for every topic that declares a page number,
    end = next such topic's start - 1 in GLOBAL declared-page order (chapters
    are sequential through the whole book, not just within one parent)."""
    dated = sorted([t for t in topics if t.declared_page_number is not None], key=lambda t: t.declared_page_number)
    ranges: dict[str, tuple[int, int]] = {}
    for i, t in enumerate(dated):
        start = t.declared_page_number
        end = dated[i + 1].declared_page_number - 1 if i + 1 < len(dated) else start + 30
        ranges[t.id] = (start, max(start, end))
    return ranges


def pdf_range_for_topic(topic_id: str, book_ranges: dict[str, tuple[int, int]], total_pages: int) -> tuple[int, int]:
    start, end = book_ranges[topic_id]
    pdf_start = max(1, start + BOOK_TO_PDF_OFFSET - WINDOW_PAD_START)
    pdf_end = min(total_pages, end + BOOK_TO_PDF_OFFSET + WINDOW_PAD_END)
    pdf_end = min(pdf_end, pdf_start + MAX_CHAPTER_PAGES)
    return pdf_start, max(pdf_start, pdf_end)


def _clean_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def render_topic_evidence(topic: Topic, blocks: list[EvidenceBlock]) -> str:
    """Compact evidence bundle: chapter title + deduped headings/subheadings
    only -- never dense paragraph text (filter_high_signal_blocks drops
    "text" blocks, same filter Stage 1 TOC extraction uses), capped so a
    50-page chapter's ~100 headings don't blow past a reasonable prompt
    budget."""
    seen: set[str] = set()
    lines: list[str] = [f"Chapter: {topic.name}"]
    for b in blocks:
        if b.block_type not in ("title", "section_header", "document_index"):
            continue
        cleaned = _clean_heading(b.text)
        if not cleaned or len(cleaned) > 120:  # long "headings" are usually mis-tagged body text
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {cleaned}")
        if len(lines) - 1 >= MAX_HEADINGS_PER_TOPIC:
            break
    rendered = "\n".join(lines)
    if len(rendered) > MAX_EVIDENCE_CHARS_PER_TOPIC:
        rendered = rendered[:MAX_EVIDENCE_CHARS_PER_TOPIC] + "\n...[evidence truncated]..."
    return rendered


def phase_retrieve(topics_by_id: dict[str, Topic]) -> None:
    from exampapersorter.pdf_utils import get_page_count

    baseline = json.loads(BASELINE_SCORED_PATH.read_text(encoding="utf-8"))
    baseline_by_index = {r["gt_index"]: r for r in baseline}
    current_topic_ids = {baseline_by_index[idx]["topic_id"] for idx in ALL_PAIRS}
    rival_topic_ids = set(ALL_PAIRS.values())
    needed_topic_ids = sorted(current_topic_ids | rival_topic_ids)
    print(f"Needed topics ({len(needed_topic_ids)}): {needed_topic_ids}", flush=True)

    total_pages = get_page_count(TEXTBOOK_PATH)
    file_hash = compute_file_hash(TEXTBOOK_PATH)
    db = Database(DEFAULT_CONFIG.database_path)
    db.register_textbook(file_hash, str(TEXTBOOK_PATH), total_pages)

    book_ranges = topic_book_page_ranges(list(topics_by_id.values()))

    evidence_bundle: dict[str, dict] = {}
    if EVIDENCE_OUT_PATH.exists():
        evidence_bundle = json.loads(EVIDENCE_OUT_PATH.read_text(encoding="utf-8"))

    for topic_id in needed_topic_ids:
        if topic_id in evidence_bundle:
            print(f"[skip, cached] {topic_id}", flush=True)
            continue
        topic = topics_by_id[topic_id]
        pdf_start, pdf_end = pdf_range_for_topic(topic_id, book_ranges, total_pages)
        print(f"[retrieve] {topic_id} ({topic.name}): book pages {book_ranges[topic_id]} -> PDF pages {pdf_start}-{pdf_end}", flush=True)

        cached = db.get_cached_evidence(file_hash, pdf_start, pdf_end)
        if cached is not None:
            evidence = cached
            print("  (docling result from cache)", flush=True)
        else:
            evidence = extract_page_range_evidence(TEXTBOOK_PATH, pdf_start, pdf_end)
            db.save_evidence(file_hash, evidence)

        print(f"  status={evidence.conversion_status} blocks={len(evidence.blocks)}", flush=True)
        filtered = filter_high_signal_blocks(evidence.blocks)
        rendered = render_topic_evidence(topic, filtered)
        evidence_bundle[topic_id] = {
            "topic_name": topic.name,
            "pdf_start": pdf_start,
            "pdf_end": pdf_end,
            "book_page_range": list(book_ranges[topic_id]),
            "conversion_status": evidence.conversion_status,
            "rendered_evidence": rendered,
            "heading_count": rendered.count("\n- "),
        }
        EVIDENCE_OUT_PATH.write_text(json.dumps(evidence_bundle, indent=2), encoding="utf-8")
        print(f"  -> {evidence_bundle[topic_id]['heading_count']} headings kept, saved.", flush=True)

    db.close()
    print(f"\nDone. Evidence for {len(evidence_bundle)} topics written to {EVIDENCE_OUT_PATH}")


# --- Adjudication (Phase 2) ---

AdjudicationVerdict = Literal["classified", "no_match", "ambiguous"]


class TopicAdjudication(BaseModel):
    status: AdjudicationVerdict
    topic_id: str | None = Field(
        default=None,
        description="Must be EXACTLY one of the two candidate topic_id values given -- never invented, "
        "never a third topic. Null unless status=classified.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(description="Which textbook evidence grounds this verdict, briefly")

    @model_validator(mode="after")
    def _topic_id_matches_status(self) -> "TopicAdjudication":
        if self.status == "classified" and not self.topic_id:
            raise ValueError("status is classified but topic_id was not provided")
        if self.status != "classified" and self.topic_id:
            raise ValueError("topic_id must be null unless status=classified")
        return self


ADJUDICATION_SYSTEM_PROMPT = """You are re-adjudicating ONE exam question's topic classification for a \
biochemistry textbook, because an earlier classification pass flagged it as uncertain between two \
specific candidate topics.

You are given: the question, TWO candidate topics (each with its own id and name), and TEXTBOOK EVIDENCE \
for each candidate -- the chapter title plus the actual headings/subheadings Docling extracted from that \
chapter's own pages in the real textbook PDF. This evidence is authoritative: it is literally what that \
chapter of the textbook contains, not a paraphrase or a general-knowledge summary.

Decide which candidate the question actually belongs to, using ONLY the textbook evidence given (not your \
own general biochemistry knowledge of how a textbook \"should\" be organized) to judge the fit. Two ways \
evidence can indicate a fit: the specific concept the question asks about appears literally as a heading/ \
subheading in that chapter, OR the chapter's overall subject matter obviously covers what's being asked \
even without an exact heading match.

Report exactly one verdict:
- status="classified": one candidate is clearly the better fit given the evidence. Set topic_id to that \
candidate's EXACT id -- it MUST be one of the two candidate ids given, never a third topic, never invented.
- status="no_match": neither candidate's evidence plausibly covers this question.
- status="ambiguous": the evidence genuinely supports both candidates comparably and you cannot confidently \
choose. Leave topic_id null.

Always include `confidence` (0-1) and a brief `reason` citing what in the evidence drove the verdict."""


def build_adjudication_prompt(question_text: str, cand_a: dict, cand_b: dict) -> str:
    return (
        f"Question:\n{question_text}\n\n"
        f"Candidate A -- topic_id: {cand_a['topic_id']}\nName: {cand_a['topic_name']}\n"
        f"Textbook evidence:\n{cand_a['evidence']}\n\n"
        f"Candidate B -- topic_id: {cand_b['topic_id']}\nName: {cand_b['topic_name']}\n"
        f"Textbook evidence:\n{cand_b['evidence']}\n\n"
        "Which candidate does this question actually belong to, per the textbook evidence above?"
    )


def phase_adjudicate(topics_by_id: dict[str, Topic], api_key: str | None) -> None:
    if not EVIDENCE_OUT_PATH.exists():
        raise SystemExit("Run --phase retrieve first (no retrieval_evidence.json found).")
    evidence_bundle = json.loads(EVIDENCE_OUT_PATH.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_SCORED_PATH.read_text(encoding="utf-8"))
    baseline_by_index = {r["gt_index"]: r for r in baseline}

    config = DEFAULT_CONFIG
    if api_key:
        config = Config(openrouter_api_key=api_key)

    results = []
    if ADJUDICATION_OUT_PATH.exists():
        results = json.loads(ADJUDICATION_OUT_PATH.read_text(encoding="utf-8"))
    done_indices = {r["gt_index"] for r in results}

    for gt_index, rival_topic_id in ALL_PAIRS.items():
        if gt_index in done_indices:
            print(f"[skip, cached] gt_index {gt_index}", flush=True)
            continue
        row = baseline_by_index[gt_index]
        current_topic_id = row["topic_id"]
        question_text = row["matched_question_text"]

        if current_topic_id not in evidence_bundle or rival_topic_id not in evidence_bundle:
            raise SystemExit(f"Missing evidence for gt_index {gt_index}: need {current_topic_id} and {rival_topic_id}")

        cand_a = {
            "topic_id": current_topic_id,
            "topic_name": evidence_bundle[current_topic_id]["topic_name"],
            "evidence": evidence_bundle[current_topic_id]["rendered_evidence"],
        }
        cand_b = {
            "topic_id": rival_topic_id,
            "topic_name": evidence_bundle[rival_topic_id]["topic_name"],
            "evidence": evidence_bundle[rival_topic_id]["rendered_evidence"],
        }
        prompt = build_adjudication_prompt(question_text, cand_a, cand_b)

        print(f"[adjudicate] gt_index={gt_index} kind={'error' if gt_index in ERROR_PAIRS else 'control'} "
              f"current={current_topic_id} rival={rival_topic_id}", flush=True)
        try:
            verdict = call_structured(config, ADJUDICATION_SYSTEM_PROMPT, prompt, TopicAdjudication,
                                       call_label="retrieval_adjudication")
        except LLMCallFailed as exc:
            print(f"  FAILED: {exc}", flush=True)
            results.append({
                "gt_index": gt_index, "kind": "error" if gt_index in ERROR_PAIRS else "control",
                "current_topic_id": current_topic_id, "rival_topic_id": rival_topic_id,
                "call_failed": True, "error": str(exc),
            })
            ADJUDICATION_OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
            continue

        print(f"  -> status={verdict.status} topic_id={verdict.topic_id} confidence={verdict.confidence}", flush=True)
        results.append({
            "gt_index": gt_index,
            "kind": "error" if gt_index in ERROR_PAIRS else "control",
            "current_topic_id": current_topic_id,
            "rival_topic_id": rival_topic_id,
            "call_failed": False,
            "adjudicated_status": verdict.status,
            "adjudicated_topic_id": verdict.topic_id,
            "adjudicated_confidence": verdict.confidence,
            "adjudicated_reason": verdict.reason,
        })
        ADJUDICATION_OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nDone. {len(results)} adjudications written to {ADJUDICATION_OUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["retrieve", "adjudicate"], required=True)
    parser.add_argument("--openrouter-api-key", default=None)
    args = parser.parse_args()

    topics = load_topics()
    topics_by_id = {t.id: t for t in topics}

    if args.phase == "retrieve":
        phase_retrieve(topics_by_id)
    elif args.phase == "adjudicate":
        import os
        api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            from exampapersorter.api_key_store import load_saved_key
            api_key = load_saved_key()
        if not api_key:
            raise SystemExit("No OpenRouter API key found (--openrouter-api-key / OPENROUTER_API_KEY / saved key).")
        phase_adjudicate(topics_by_id, api_key)


if __name__ == "__main__":
    main()
