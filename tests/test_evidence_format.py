from exampapersorter.schemas import EvidenceBlock
from exampapersorter.topic_extraction.evidence_format import (
    filter_high_signal_blocks,
    select_topic_extraction_evidence,
)


def block(block_type, text, pages=(1,)):
    return EvidenceBlock(page_numbers=list(pages), block_type=block_type, text=text)


def test_filter_high_signal_blocks_drops_dense_text_only():
    blocks = [
        block("text", "a long paragraph of prose"),
        block("section_header", "Chapter 1"),
        block("table", "a | table | row"),
        block("document_index", "contents listing"),
        block("list_item", "a bullet point"),
        block("caption", "Fig 1.1"),
    ]
    result = filter_high_signal_blocks(blocks)
    assert [b.block_type for b in result] == [
        "section_header", "table", "document_index", "list_item", "caption",
    ]


def test_extraction_evidence_prefers_document_index_over_everything_else():
    blocks = [
        block("section_header", "Nucleus", pages=(14,)),  # body-content noise
        block("table", "unrelated data table", pages=(15,)),
        block("document_index", "the real contents table", pages=(10,)),
        block("document_index", "a second contents table", pages=(11,)),
    ]
    result = select_topic_extraction_evidence(blocks)
    assert len(result) == 2
    assert all(b.block_type == "document_index" for b in result)


def test_extraction_evidence_falls_back_to_table_when_no_document_index():
    blocks = [
        block("section_header", "Chapter 1", pages=(5,)),
        block("table", "chapter | page\n1 | 5\n2 | 20", pages=(6,)),
    ]
    result = select_topic_extraction_evidence(blocks)
    assert len(result) == 1
    assert result[0].block_type == "table"


def test_extraction_evidence_falls_back_to_headers_when_toc_is_plain_text():
    blocks = [
        block("text", "some unrelated prose", pages=(1,)),
        block("section_header", "Table of Contents", pages=(2,)),
        block("list_item", "Chapter 1: Introduction ... 1", pages=(2,)),
        block("list_item", "Chapter 2: Metabolism ... 20", pages=(2,)),
    ]
    result = select_topic_extraction_evidence(blocks)
    assert [b.block_type for b in result] == ["section_header", "list_item", "list_item"]
