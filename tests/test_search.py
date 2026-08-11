from dataclasses import replace
from pathlib import Path

from exampapersorter.config import DEFAULT_CONFIG
from exampapersorter.database import Database
from exampapersorter.schemas import EvidenceBlock, PageRangeEvidence, TOCDetectionResult, TOCEvidenceRef
from exampapersorter.topic_extraction import search as search_module


def make_evidence(start, end):
    return PageRangeEvidence(
        source_path="fake.pdf",
        start_page=start,
        end_page=end,
        conversion_status="success",
        blocks=[EvidenceBlock(page_numbers=[start], block_type="text", text=f"content of {start}-{end}")],
    )


def test_stops_at_first_toc_match(tmp_path, monkeypatch):
    config = replace(
        DEFAULT_CONFIG, initial_textbook_search_range=10, textbook_search_step=10, maximum_textbook_search_pages=100
    )
    db = Database(tmp_path / "test.db")
    calls = []

    def fake_extract(pdf_path, start, end):
        calls.append((start, end))
        return make_evidence(start, end)

    def fake_detect(evidence, cfg):
        if evidence.start_page == 11:
            return TOCDetectionResult(
                document_structure="table_of_contents", confidence=0.9,
                evidence=[TOCEvidenceRef(page=11, text="Contents")],
            )
        return TOCDetectionResult(
            document_structure="neither", confidence=0.1,
            evidence=[TOCEvidenceRef(page=evidence.start_page, text="not toc")],
        )

    monkeypatch.setattr(search_module, "extract_page_range_evidence", fake_extract)
    monkeypatch.setattr(search_module, "detect_toc", fake_detect)

    result = search_module.find_table_of_contents(Path("fake.pdf"), "hash123", total_pages=100, db=db, config=config)

    assert result.found
    assert result.evidence.start_page == 11
    assert calls == [(1, 10), (11, 20)]  # search stopped, didn't continue past the match
    db.close()


def test_respects_max_search_pages_bound(tmp_path, monkeypatch):
    config = replace(
        DEFAULT_CONFIG, initial_textbook_search_range=10, textbook_search_step=10, maximum_textbook_search_pages=30
    )
    db = Database(tmp_path / "test.db")
    calls = []

    def fake_extract(pdf_path, start, end):
        calls.append((start, end))
        return make_evidence(start, end)

    def fake_detect(evidence, cfg):
        return TOCDetectionResult(
            document_structure="neither", confidence=0.1,
            evidence=[TOCEvidenceRef(page=evidence.start_page, text="not toc")],
        )

    monkeypatch.setattr(search_module, "extract_page_range_evidence", fake_extract)
    monkeypatch.setattr(search_module, "detect_toc", fake_detect)

    result = search_module.find_table_of_contents(Path("fake.pdf"), "hash123", total_pages=1000, db=db, config=config)

    assert not result.found
    # even though total_pages=1000, the search never goes past maximum_textbook_search_pages=30
    assert calls == [(1, 10), (11, 20), (21, 30)]
    db.close()


def test_resume_uses_cached_evidence_and_verdict_without_recalling(tmp_path, monkeypatch):
    config = replace(
        DEFAULT_CONFIG, initial_textbook_search_range=10, textbook_search_step=10, maximum_textbook_search_pages=20
    )
    db = Database(tmp_path / "test.db")
    file_hash = "hash123"

    # Simulate a prior run that already processed pages 1-10 and found "neither"
    db.save_evidence(file_hash, make_evidence(1, 10))
    db.log_toc_verdict(
        file_hash, 1, 10, config.model_identifier,
        TOCDetectionResult(document_structure="neither", confidence=0.1, evidence=[TOCEvidenceRef(page=1, text="not toc")]),
    )

    extract_calls = []
    detect_calls = []

    def fake_extract(pdf_path, start, end):
        extract_calls.append((start, end))
        return make_evidence(start, end)

    def fake_detect(evidence, cfg):
        detect_calls.append((evidence.start_page, evidence.end_page))
        return TOCDetectionResult(
            document_structure="table_of_contents", confidence=0.95,
            evidence=[TOCEvidenceRef(page=evidence.start_page, text="Contents")],
        )

    monkeypatch.setattr(search_module, "extract_page_range_evidence", fake_extract)
    monkeypatch.setattr(search_module, "detect_toc", fake_detect)

    result = search_module.find_table_of_contents(Path("fake.pdf"), file_hash, total_pages=20, db=db, config=config)

    assert result.found
    assert result.evidence.start_page == 11
    # pages 1-10 came from the DB cache -- Docling and the LLM were never re-invoked for that range
    assert (1, 10) not in extract_calls
    assert (1, 10) not in detect_calls
    db.close()
