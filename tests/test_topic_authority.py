import json
from dataclasses import replace

import pymupdf
import pytest

from exampapersorter.config import DEFAULT_CONFIG
from exampapersorter.database import Database
from exampapersorter.pdf_utils import compute_file_hash
from exampapersorter.schemas import EvidenceBlock, PageRangeEvidence, TOCDetectionResult, Topic, TopicExtractionResult
from exampapersorter.topic_authority import TopicAuthorityError, resolve_index_topic_authority, resolve_textbook_topic_authority
import exampapersorter.topic_authority as topic_authority_module
from exampapersorter.topic_extraction.search import SearchResult


def _config(tmp_path):
    return replace(DEFAULT_CONFIG, database_path=tmp_path / "t.db", output_directory=tmp_path / "output")


def _tiny_pdf(tmp_path, name="book.pdf", pages=3):
    path = tmp_path / name
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}", fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


def _fake_search_result():
    evidence = PageRangeEvidence(
        source_path="book.pdf", start_page=1, end_page=2, conversion_status="success",
        blocks=[EvidenceBlock(page_numbers=[1], block_type="title", text="Table of Contents")],
    )
    verdict = TOCDetectionResult(document_structure="table_of_contents", confidence=0.9, evidence=[])
    return SearchResult(found=True, evidence=evidence, verdict=verdict, attempts=[])


# --- Textbook mode ---


def test_textbook_mode_reuses_cached_topics_without_calling_find_toc(tmp_path, monkeypatch):
    config = _config(tmp_path)
    pdf_path = _tiny_pdf(tmp_path)
    db = Database(config.database_path)
    file_hash = compute_file_hash(pdf_path)

    cached_topics = TopicExtractionResult(topics=[Topic(id="c1", name="Chapter 1", level=1, source_pages=[1])])
    db.save_topic_extraction_run(file_hash, "prior-model", 1, 2, status="success", topics=cached_topics, validation=None)

    def _boom(*args, **kwargs):
        raise AssertionError("find_table_of_contents should not be called when topics are already cached")

    monkeypatch.setattr(topic_authority_module, "find_table_of_contents", _boom)

    result = resolve_textbook_topic_authority(pdf_path, config, db)
    db.close()

    assert result.from_cache is True
    assert result.source_type == "textbook"
    assert [t.id for t in result.topics] == ["c1"]


def test_textbook_mode_raises_when_no_toc_found(tmp_path, monkeypatch):
    config = _config(tmp_path)
    pdf_path = _tiny_pdf(tmp_path)
    db = Database(config.database_path)

    monkeypatch.setattr(
        topic_authority_module, "find_table_of_contents",
        lambda *a, **k: SearchResult(found=False, attempts=[]),
    )

    with pytest.raises(TopicAuthorityError):
        resolve_textbook_topic_authority(pdf_path, config, db)
    db.close()


def test_textbook_mode_raises_when_file_missing(tmp_path):
    config = _config(tmp_path)
    db = Database(config.database_path)
    with pytest.raises(TopicAuthorityError):
        resolve_textbook_topic_authority(tmp_path / "does_not_exist.pdf", config, db)
    db.close()


def test_textbook_mode_persists_fresh_topics_on_success(tmp_path, monkeypatch):
    config = _config(tmp_path)
    pdf_path = _tiny_pdf(tmp_path)
    db = Database(config.database_path)

    monkeypatch.setattr(topic_authority_module, "find_table_of_contents", lambda *a, **k: _fake_search_result())
    extracted = TopicExtractionResult(topics=[Topic(id="c1", name="Chapter 1", level=1, source_pages=[1])])
    monkeypatch.setattr(topic_authority_module, "extract_topics", lambda *a, **k: extracted)

    result = resolve_textbook_topic_authority(pdf_path, config, db)

    assert result.from_cache is False
    assert [t.id for t in result.topics] == ["c1"]

    file_hash = compute_file_hash(pdf_path)
    persisted = db.get_latest_topic_extraction(file_hash)
    db.close()
    assert persisted is not None
    assert [t.id for t in persisted.topics] == ["c1"]


# --- Index mode: deterministic text/markdown outline ---


def test_index_outline_mode_parses_and_persists(tmp_path):
    config = _config(tmp_path)
    db = Database(config.database_path)
    index_path = tmp_path / "syllabus.md"
    index_path.write_text("# Section A\n## Chapter 1\n## Chapter 2\n", encoding="utf-8")

    result = resolve_index_topic_authority(index_path, config, db)
    db.close()

    assert result.from_cache is False
    assert result.source_type == "index_outline"
    assert {t.name for t in result.topics} == {"Section A", "Chapter 1", "Chapter 2"}


def test_index_outline_mode_reuses_cache_on_second_call(tmp_path):
    config = _config(tmp_path)
    db = Database(config.database_path)
    index_path = tmp_path / "syllabus.txt"
    index_path.write_text("Section A\n  Chapter 1\n", encoding="utf-8")

    first = resolve_index_topic_authority(index_path, config, db)
    second = resolve_index_topic_authority(index_path, config, db)
    db.close()

    assert first.from_cache is False
    assert second.from_cache is True
    assert {t.name for t in second.topics} == {t.name for t in first.topics}


def test_index_outline_mode_raises_on_unparseable_content(tmp_path):
    config = _config(tmp_path)
    db = Database(config.database_path)
    index_path = tmp_path / "empty.txt"
    index_path.write_text("   \n\n", encoding="utf-8")

    with pytest.raises(TopicAuthorityError):
        resolve_index_topic_authority(index_path, config, db)
    db.close()


# --- Index mode: JSON ---


def test_index_json_mode_parses_nested_tree(tmp_path):
    config = _config(tmp_path)
    db = Database(config.database_path)
    index_path = tmp_path / "topics.json"
    index_path.write_text(
        json.dumps([{"name": "Section A", "children": [{"name": "Chapter 1"}]}]), encoding="utf-8"
    )

    result = resolve_index_topic_authority(index_path, config, db)
    db.close()

    assert result.source_type == "index_json"
    names = {t.name for t in result.topics}
    assert names == {"Section A", "Chapter 1"}


def test_index_json_mode_raises_on_malformed_json(tmp_path):
    config = _config(tmp_path)
    db = Database(config.database_path)
    index_path = tmp_path / "bad.json"
    index_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(TopicAuthorityError):
        resolve_index_topic_authority(index_path, config, db)
    db.close()


# --- Index mode: PDF (LLM call mocked) ---


def test_index_pdf_mode_calls_extract_topics_once_then_caches(tmp_path, monkeypatch):
    config = _config(tmp_path)
    db = Database(config.database_path)
    index_pdf = _tiny_pdf(tmp_path, name="index.pdf", pages=2)

    fake_evidence = PageRangeEvidence(
        source_path=str(index_pdf), start_page=1, end_page=2, conversion_status="success",
        blocks=[EvidenceBlock(page_numbers=[1], block_type="title", text="Contents")],
    )
    monkeypatch.setattr(topic_authority_module, "extract_page_range_evidence", lambda *a, **k: fake_evidence)

    call_count = {"n": 0}
    extracted = TopicExtractionResult(topics=[Topic(id="c1", name="Chapter 1", level=1, source_pages=[1])])

    def _fake_extract_topics(*args, **kwargs):
        call_count["n"] += 1
        return extracted

    monkeypatch.setattr(topic_authority_module, "extract_topics", _fake_extract_topics)

    first = resolve_index_topic_authority(index_pdf, config, db)
    second = resolve_index_topic_authority(index_pdf, config, db)
    db.close()

    assert first.source_type == "index_pdf"
    assert first.from_cache is False
    assert second.from_cache is True
    assert call_count["n"] == 1  # second call must reuse the cached topic list, not re-invoke the LLM


def test_index_pdf_mode_rejects_oversized_pdf(tmp_path, monkeypatch):
    config = _config(tmp_path)
    db = Database(config.database_path)
    index_pdf = _tiny_pdf(tmp_path, name="huge.pdf", pages=1)

    monkeypatch.setattr(topic_authority_module, "get_page_count", lambda *a, **k: config.maximum_textbook_search_pages + 1)

    with pytest.raises(TopicAuthorityError):
        resolve_index_topic_authority(index_pdf, config, db)
    db.close()


# --- Shared validation ---


def test_unsupported_index_extension_raises(tmp_path):
    config = _config(tmp_path)
    db = Database(config.database_path)
    index_path = tmp_path / "topics.docx"
    index_path.write_text("whatever", encoding="utf-8")

    with pytest.raises(TopicAuthorityError):
        resolve_index_topic_authority(index_path, config, db)
    db.close()


def test_missing_index_file_raises(tmp_path):
    config = _config(tmp_path)
    db = Database(config.database_path)
    with pytest.raises(TopicAuthorityError):
        resolve_index_topic_authority(tmp_path / "nope.txt", config, db)
    db.close()
