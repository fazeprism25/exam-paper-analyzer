from dataclasses import replace
from pathlib import Path

from exampapersorter.config import DEFAULT_CONFIG
from exampapersorter.question_extraction import paper_boundaries as paper_boundaries_module
from exampapersorter.question_extraction.paper_boundaries import detect_paper_boundaries, resolve_paper_spans
from exampapersorter.schemas import (
    EvidenceBlock,
    EvidenceRef,
    PageRangeEvidence,
    PaperBoundaryCandidate,
    PaperBoundaryDetectionResult,
)


def cand(start, end, confidence=0.9):
    return PaperBoundaryCandidate(start_page=start, end_page=end, confidence=confidence, evidence=[])


# --- resolve_paper_spans: one paper per PDF ---


def test_single_span_covering_whole_document_passes_through_unchanged():
    spans, issues = resolve_paper_spans([cand(1, 10)], total_pages=10)
    assert [(s.start_page, s.end_page) for s in spans] == [(1, 10)]
    assert issues == []


# --- resolve_paper_spans: multiple papers in one PDF ---


def test_multiple_non_overlapping_spans_preserved_in_order():
    spans, issues = resolve_paper_spans([cand(6, 10), cand(1, 5)], total_pages=10)
    assert [(s.start_page, s.end_page) for s in spans] == [(1, 5), (6, 10)]
    assert issues == []


# --- overlap clipping ---


def test_overlapping_spans_are_clipped_and_flagged():
    spans, issues = resolve_paper_spans([cand(1, 6), cand(5, 10)], total_pages=10)
    assert [(s.start_page, s.end_page) for s in spans] == [(1, 6), (7, 10)]
    assert any(i.code == "overlap_clipped" for i in issues)


def test_span_entirely_swallowed_by_a_preceding_span_is_dropped_and_flagged():
    spans, issues = resolve_paper_spans([cand(1, 10), cand(3, 5)], total_pages=10)
    assert [(s.start_page, s.end_page) for s in spans] == [(1, 10)]
    assert any(i.code == "overlap_clipped" for i in issues)


# --- gap handling (numbering resets / missed boundaries) ---


def test_interior_gap_is_attached_to_preceding_paper_and_flagged():
    spans, issues = resolve_paper_spans([cand(1, 3), cand(6, 10)], total_pages=10)
    assert [(s.start_page, s.end_page) for s in spans] == [(1, 5), (6, 10)]
    gap_issues = [i for i in issues if i.code == "gap_filled"]
    assert len(gap_issues) == 1
    assert gap_issues[0].pages == [4, 5]


def test_trailing_gap_is_attached_to_last_paper_and_flagged():
    spans, issues = resolve_paper_spans([cand(1, 5)], total_pages=10)
    assert [(s.start_page, s.end_page) for s in spans] == [(1, 10)]
    assert any(i.code == "gap_filled" and i.pages == [6, 7, 8, 9, 10] for i in issues)


def test_leading_gap_with_no_preceding_paper_is_left_unassigned_not_dropped():
    spans, issues = resolve_paper_spans([cand(4, 10)], total_pages=10)
    assert [(s.start_page, s.end_page) for s in spans] == [(4, 10)]
    assert any(i.code == "gap_unassigned" and i.pages == [1, 2, 3] for i in issues)


def test_no_candidates_at_all_leaves_everything_unassigned():
    spans, issues = resolve_paper_spans([], total_pages=5)
    assert spans == []
    assert any(i.code == "gap_unassigned" and i.pages == [1, 2, 3, 4, 5] for i in issues)


# --- detect_paper_boundaries LLM call wiring ---


def block(block_type, text, pages=(1,)):
    return EvidenceBlock(page_numbers=list(pages), block_type=block_type, text=text)


def test_detect_paper_boundaries_grounds_prompt_and_returns_model_result(monkeypatch):
    evidence = PageRangeEvidence(
        source_path="fake.pdf", start_page=1, end_page=5, conversion_status="success",
        blocks=[block("title", "Some Institution", pages=(1,))],
    )
    captured = {}

    def fake_call_structured(config, system_prompt, user_prompt, response_model, call_label=None):
        captured["user_prompt"] = user_prompt
        captured["response_model"] = response_model
        return PaperBoundaryDetectionResult(
            papers=[PaperBoundaryCandidate(start_page=1, end_page=5, confidence=0.8, evidence=[EvidenceRef(page=1, text="Some Institution")])]
        )

    monkeypatch.setattr(paper_boundaries_module, "call_structured", fake_call_structured)
    result = detect_paper_boundaries(evidence, DEFAULT_CONFIG)

    assert len(result.papers) == 1
    assert "Some Institution" in captured["user_prompt"]
    assert captured["response_model"] is PaperBoundaryDetectionResult
