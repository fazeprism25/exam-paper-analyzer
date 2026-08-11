from pathlib import Path

from exampapersorter.config import DEFAULT_CONFIG
from exampapersorter.database import Database
from exampapersorter.schemas import (
    ChapterNameRecoveryResult,
    ChapterNameResolution,
    EvidenceBlock,
    PageRangeEvidence,
    Topic,
)
from exampapersorter.topic_extraction import name_recovery as name_recovery_module
from exampapersorter.topic_extraction.name_recovery import recover_names


def block(pdf_page, block_type, text):
    return EvidenceBlock(page_numbers=[pdf_page], block_type=block_type, text=text)


def toc_window_with_mapping(offset=10):
    """A fake confirmed-TOC-window evidence with enough header/footer
    samples to build a page_number_mapping (pdf_page = book_page + offset)."""
    blocks = [
        block(1 + offset, "page_footer", "1"),
        block(2 + offset, "page_footer", "2"),
        block(5 + offset, "page_header", "5"),
        block(9 + offset, "page_footer", "9"),
    ]
    return PageRangeEvidence(
        source_path="fake.pdf", start_page=1, end_page=25, conversion_status="success", blocks=blocks
    )


def unresolved_topic(topic_id="ch12", name="12", book_page=244):
    return Topic(
        id=topic_id, name=name, level=1, parent_id=None, source_pages=[10],
        declared_page_number=book_page, resolution_status="name_unresolved",
    )


def test_no_unresolved_topics_is_a_no_op(tmp_path):
    db = Database(tmp_path / "test.db")
    topics = [Topic(id="ch1", name="Carbohydrates", level=1, parent_id=None, source_pages=[9])]
    result = recover_names(topics, toc_window_with_mapping(), Path("fake.pdf"), "hash", db, DEFAULT_CONFIG)
    assert result == topics
    db.close()


def test_mapping_failure_marks_all_unidentified(tmp_path, monkeypatch):
    db = Database(tmp_path / "test.db")
    topics = [unresolved_topic()]
    # No page_header/page_footer blocks at all -- mapping cannot be built.
    empty_toc_window = PageRangeEvidence(
        source_path="fake.pdf", start_page=1, end_page=25, conversion_status="success", blocks=[]
    )

    called = []
    monkeypatch.setattr(name_recovery_module, "call_structured", lambda *a, **k: called.append(1))

    result = recover_names(topics, empty_toc_window, Path("fake.pdf"), "hash", db, DEFAULT_CONFIG)
    assert result[0].resolution_status == "unidentified"
    assert called == []  # never even reached the LLM
    db.close()


def test_self_check_failure_marks_unidentified_without_calling_llm(tmp_path, monkeypatch):
    db = Database(tmp_path / "test.db")
    topics = [unresolved_topic(book_page=244)]

    # Candidate window fetched, but it never shows the expected book page 244 anywhere.
    def fake_extract(pdf_path, start, end):
        return PageRangeEvidence(
            source_path="fake.pdf", start_page=start, end_page=end, conversion_status="success",
            blocks=[block(start, "page_footer", "999")],
        )

    called = []
    monkeypatch.setattr(name_recovery_module, "extract_page_range_evidence", fake_extract)
    monkeypatch.setattr(name_recovery_module, "call_structured", lambda *a, **k: called.append(1))

    result = recover_names(topics, toc_window_with_mapping(), Path("fake.pdf"), "hash", db, DEFAULT_CONFIG)
    assert result[0].resolution_status == "unidentified"
    assert called == []
    db.close()


def test_confirmed_page_and_resolved_llm_response_updates_topic(tmp_path, monkeypatch):
    db = Database(tmp_path / "test.db")
    topics = [unresolved_topic(topic_id="ch12", book_page=244)]
    expected_pdf_page = 244 + 10  # offset=10 from toc_window_with_mapping()

    def fake_extract(pdf_path, start, end):
        return PageRangeEvidence(
            source_path="fake.pdf", start_page=start, end_page=end, conversion_status="success",
            blocks=[
                block(expected_pdf_page, "page_footer", "244"),
                block(expected_pdf_page, "section_header", "Lipid Metabolism"),
            ],
        )

    def fake_call_structured(config, system_prompt, user_prompt, response_model):
        return ChapterNameRecoveryResult(
            resolutions=[ChapterNameResolution(topic_id="ch12", status="resolved", name="Lipid Metabolism")]
        )

    monkeypatch.setattr(name_recovery_module, "extract_page_range_evidence", fake_extract)
    monkeypatch.setattr(name_recovery_module, "call_structured", fake_call_structured)

    result = recover_names(topics, toc_window_with_mapping(), Path("fake.pdf"), "hash", db, DEFAULT_CONFIG)
    updated = result[0]
    assert updated.resolution_status == "resolved"
    assert updated.name == "Lipid Metabolism"
    assert updated.name_evidence_source == "chapter_opening_page"
    assert updated.name_source_pages == [expected_pdf_page]
    db.close()


def test_not_found_llm_response_keeps_name_unresolved(tmp_path, monkeypatch):
    db = Database(tmp_path / "test.db")
    topics = [unresolved_topic(topic_id="ch12", book_page=244)]
    expected_pdf_page = 254

    def fake_extract(pdf_path, start, end):
        return PageRangeEvidence(
            source_path="fake.pdf", start_page=start, end_page=end, conversion_status="success",
            blocks=[block(expected_pdf_page, "page_footer", "244")],
        )

    def fake_call_structured(config, system_prompt, user_prompt, response_model):
        return ChapterNameRecoveryResult(
            resolutions=[ChapterNameResolution(topic_id="ch12", status="not_found", name=None)]
        )

    monkeypatch.setattr(name_recovery_module, "extract_page_range_evidence", fake_extract)
    monkeypatch.setattr(name_recovery_module, "call_structured", fake_call_structured)

    result = recover_names(topics, toc_window_with_mapping(), Path("fake.pdf"), "hash", db, DEFAULT_CONFIG)
    assert result[0].resolution_status == "name_unresolved"
    db.close()


def test_ambiguous_llm_response_sets_ambiguous_status(tmp_path, monkeypatch):
    db = Database(tmp_path / "test.db")
    topics = [unresolved_topic(topic_id="ch12", book_page=244)]
    expected_pdf_page = 254

    def fake_extract(pdf_path, start, end):
        return PageRangeEvidence(
            source_path="fake.pdf", start_page=start, end_page=end, conversion_status="success",
            blocks=[block(expected_pdf_page, "page_footer", "244")],
        )

    def fake_call_structured(config, system_prompt, user_prompt, response_model):
        return ChapterNameRecoveryResult(
            resolutions=[ChapterNameResolution(topic_id="ch12", status="ambiguous", reason="two titles present")]
        )

    monkeypatch.setattr(name_recovery_module, "extract_page_range_evidence", fake_extract)
    monkeypatch.setattr(name_recovery_module, "call_structured", fake_call_structured)

    result = recover_names(topics, toc_window_with_mapping(), Path("fake.pdf"), "hash", db, DEFAULT_CONFIG)
    assert result[0].resolution_status == "ambiguous"
    db.close()


def test_never_invents_a_name_not_present_in_evidence(tmp_path, monkeypatch):
    """If the LLM call fails outright, topics must not silently keep a
    fabricated-looking resolved status -- they fall back to unidentified."""
    db = Database(tmp_path / "test.db")
    topics = [unresolved_topic(topic_id="ch12", book_page=244)]
    expected_pdf_page = 254

    def fake_extract(pdf_path, start, end):
        return PageRangeEvidence(
            source_path="fake.pdf", start_page=start, end_page=end, conversion_status="success",
            blocks=[block(expected_pdf_page, "page_footer", "244")],
        )

    from exampapersorter.llm_client import LLMCallFailed

    def failing_call_structured(*args, **kwargs):
        raise LLMCallFailed("model unavailable")

    monkeypatch.setattr(name_recovery_module, "extract_page_range_evidence", fake_extract)
    monkeypatch.setattr(name_recovery_module, "call_structured", failing_call_structured)

    result = recover_names(topics, toc_window_with_mapping(), Path("fake.pdf"), "hash", db, DEFAULT_CONFIG)
    assert result[0].resolution_status == "unidentified"
    assert result[0].name == "12"  # unchanged, nothing invented
    db.close()
