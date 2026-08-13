"""Docling wrapper: PDF page range -> page-provenance evidence blocks.

Design decisions (validated empirically against the real textbook before
writing this, see conversation record):

- We slice with DocumentConverter.convert(path, page_range=(start, end))
  rather than pre-slicing the PDF into a temp file with PyMuPDF. Docling's
  own page_range keeps the ORIGINAL page numbers in provenance, while a
  manually sliced temp file gets renumbered from 1 -- that would require
  us to track and apply an offset everywhere, a whole class of bugs we get
  to skip. We confirmed on the real 809-page textbook that page_range also
  doesn't cost more as it's pointed deeper into the book, so it isn't
  secretly doing a full-document pass first.

- Evidence is built from a single doc.iterate_items() walk, not from
  doc.texts alone. On the real textbook, the entire 43-chapter table of
  contents was represented as a TableItem (Docling's layout model reads
  dot-leader TOC lines as tabular), and doc.texts on that page contained
  only stray page-number fragments. Docling also has a distinct
  DocItemLabel.DOCUMENT_INDEX for tables it already suspects are a TOC/
  index -- we surface that as its own block_type as a hint for the LLM,
  but the LLM still has to confirm it from the actual text (never trust
  the structural guess alone).
"""
from __future__ import annotations

import logging
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.table.table import TableItem
from docling_core.types.doc.labels import DocItemLabel

from exampapersorter.pdf_utils import page_range_has_text_layer
from exampapersorter.schemas import BlockType, EvidenceBlock, PageRangeEvidence

logger = logging.getLogger(__name__)

_LABEL_TO_BLOCK_TYPE: dict[DocItemLabel, BlockType] = {
    DocItemLabel.TITLE: "title",
    DocItemLabel.SECTION_HEADER: "section_header",
    DocItemLabel.TEXT: "text",
    DocItemLabel.PARAGRAPH: "text",
    DocItemLabel.REFERENCE: "text",
    DocItemLabel.FOOTNOTE: "text",
    DocItemLabel.LIST_ITEM: "list_item",
    DocItemLabel.CAPTION: "caption",
    DocItemLabel.PAGE_HEADER: "page_header",
    DocItemLabel.PAGE_FOOTER: "page_footer",
    DocItemLabel.TABLE: "table",
    DocItemLabel.DOCUMENT_INDEX: "document_index",
}

# DocumentConverter loads its OCR/layout models on first use; building a
# fresh one per call would pay that cost every time, so callers share one.
#
# Two converters, not one: Docling's do_ocr is set at DocumentConverter
# construction time, not per convert() call (there's no per-call pipeline
# option override in the installed version). We keep a do_ocr=True instance
# (Docling's own default) and a do_ocr=False instance, and pick between them
# per page-range via page_range_has_text_layer(). Confirmed empirically
# (see docling_exp/ in conversation record) that Docling's default do_ocr=True
# already skips OCR per-page when a page has low bitmap coverage -- but for a
# page that IS mostly a scanned bitmap, that decision is coverage-based, not
# text-based, so Docling still re-runs full OCR-engine inference on a page
# that was already OCR'd (e.g. by OCRmyPDF) before falling back to the
# pre-existing text (OCR cells that overlap existing cells are discarded --
# see base_ocr_model.py's _filter_ocr_cells). The re-run doesn't corrupt
# output, it just burns time for nothing across every already-texted page --
# measured ~2.6x slower per page than do_ocr=False for an already-OCR'd
# fixture, with byte-identical extracted text either way. do_ocr=False is
# only ever selected when EVERY page in the range already clears the text
# threshold, so an image-only page never gets silently dropped.
_converters: dict[bool, DocumentConverter] = {}


def _get_converter(do_ocr: bool) -> DocumentConverter:
    if do_ocr not in _converters:
        if do_ocr:
            _converters[do_ocr] = DocumentConverter()
        else:
            pipeline_options = PdfPipelineOptions(do_ocr=False)
            _converters[do_ocr] = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
    return _converters[do_ocr]


def _flatten_table_to_text(table_item: TableItem, doc: DoclingDocument) -> str:
    """Join a table's cells into readable pipe-separated rows.

    Docling's dataframe export raises a deprecation warning if called
    without `doc`; passing it also resolves cell text correctly for
    tables that reference document-level content.
    """
    try:
        df = table_item.export_to_dataframe(doc)
    except Exception as exc:
        logger.warning("Failed to export table to dataframe: %s", exc)
        return ""

    lines = []
    for row in df.itertuples(index=False):
        cells = [str(v).strip() for v in row if str(v).strip() and str(v).strip().lower() != "nan"]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _build_evidence_blocks(doc: DoclingDocument) -> list[EvidenceBlock]:
    blocks: list[EvidenceBlock] = []
    # iterate_items()'s level is the item's nesting depth in Docling's own
    # document tree -- kept as EvidenceBlock.list_level rather than
    # discarded, since it's a generic structural signal Stage 2's MCQ
    # stem/option grouping uses (see question_extraction/mcq_grouping.py):
    # sibling list items (e.g. an MCQ's answer options) tend to share a
    # depth, distinct from the stem or from an unrelated item at another
    # depth. It's a hint, not a guaranteed grouping.
    for item, level in doc.iterate_items():
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        page_numbers = sorted({p.page_no for p in prov})

        if isinstance(item, TableItem):
            text = _flatten_table_to_text(item, doc)
            if not text:
                continue
            block_type = _LABEL_TO_BLOCK_TYPE.get(item.label, "table")
            blocks.append(EvidenceBlock(page_numbers=page_numbers, block_type=block_type, text=text, list_level=level))
            continue

        text = getattr(item, "text", None)
        if not text or not text.strip():
            continue
        block_type = _LABEL_TO_BLOCK_TYPE.get(getattr(item, "label", None), "other")
        # ListItem carries its own parsed marker separately from `text` --
        # confirmed directly against the real fixture that Docling often
        # strips the marker out of `text` entirely (an MCQ option's text
        # can be bare "Obstructive Jaundice" with "A." living only in
        # `marker`), so this must be captured independently rather than
        # assumed to be part of `text`. getattr default handles item types
        # (TitleItem, TextItem, etc.) that don't have this attribute at all.
        marker = getattr(item, "marker", None) or None
        blocks.append(
            EvidenceBlock(
                page_numbers=page_numbers, block_type=block_type, text=text.strip(), list_level=level, marker=marker
            )
        )

    return blocks


def extract_page_range_evidence(pdf_path: Path, start_page: int, end_page: int) -> PageRangeEvidence:
    """Run Docling on [start_page, end_page] (1-indexed, inclusive) and
    return page-provenance evidence. Never raises -- failures are
    reported in the returned object's conversion_status/error_message so
    callers can log-and-continue per file/range rather than crash.
    """
    if not pdf_path.exists():
        return PageRangeEvidence(
            source_path=str(pdf_path),
            start_page=start_page,
            end_page=end_page,
            conversion_status="failure",
            blocks=[],
            error_message=f"File not found: {pdf_path}",
        )

    try:
        needs_ocr = not page_range_has_text_layer(pdf_path, start_page, end_page)
    except Exception as exc:
        # A file PyMuPDF can't even open (e.g. corrupt) shouldn't crash the
        # probe -- fall back to do_ocr=True and let Docling's own
        # raises_on_error=False / try-except below report the failure the
        # same way it always has.
        logger.warning("Text-layer probe failed for %s [%d-%d]: %s", pdf_path, start_page, end_page, exc)
        needs_ocr = True
    converter = _get_converter(do_ocr=needs_ocr)
    try:
        result = converter.convert(pdf_path, page_range=(start_page, end_page), raises_on_error=False)
    except Exception as exc:
        logger.error("Docling conversion raised for %s [%d-%d]: %s", pdf_path, start_page, end_page, exc)
        return PageRangeEvidence(
            source_path=str(pdf_path),
            start_page=start_page,
            end_page=end_page,
            conversion_status="failure",
            blocks=[],
            error_message=str(exc),
        )

    status = result.status.value
    error_message = "; ".join(e.error_message for e in result.errors) if result.errors else None

    if status in ("failure", "skipped", "pending", "started"):
        logger.warning("Docling status=%s for %s [%d-%d]: %s", status, pdf_path, start_page, end_page, error_message)
        return PageRangeEvidence(
            source_path=str(pdf_path),
            start_page=start_page,
            end_page=end_page,
            conversion_status=status,
            blocks=[],
            error_message=error_message,
        )

    if status == "partial_success":
        logger.warning("Docling partial_success for %s [%d-%d]: %s", pdf_path, start_page, end_page, error_message)

    blocks = _build_evidence_blocks(result.document)
    return PageRangeEvidence(
        source_path=str(pdf_path),
        start_page=start_page,
        end_page=end_page,
        conversion_status=status,
        blocks=blocks,
        error_message=error_message,
    )
