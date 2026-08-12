"""Pydantic models used across the pipeline.

Two categories live here:
  - Evidence models: deterministic, produced by our own code (Docling
    wrapper). Never touched by the LLM.
  - LLM output models: passed as `format=<schema>` to Ollama, and every
    LLM response is validated against one of these before being trusted.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# --- Evidence models (deterministic, from docling_processing.py) ---

BlockType = Literal[
    "title",
    "section_header",
    "text",
    "list_item",
    "table",
    "document_index",  # Docling's own layout model's guess at TOC/index tables -- a hint, not a verdict
    "caption",
    "page_header",
    "page_footer",
    "other",
    # Deterministic, code-produced (never from Docling directly): a run of
    # option-like list_items bundled with their preceding question stem --
    # see question_extraction/mcq_grouping.py. Only ever appears in evidence
    # rendered for the question-extraction LLM call, never in the cached
    # whole-document evidence.
    "mcq_group",
]


class EvidenceBlock(BaseModel):
    """One piece of extracted content with page provenance.

    For text-family Docling items this is one item. For a TableItem, this
    is the whole table flattened to pipe-separated rows -- see
    docling_processing.flatten_table_to_text for why tables can't be
    skipped: real tables of contents routinely get classified as tables
    by Docling's layout model, not as text.
    """

    page_numbers: list[int] = Field(description="Original PDF page number(s) this block came from")
    block_type: BlockType
    text: str
    list_level: int | None = Field(
        default=None,
        description="Docling's own nesting depth for this item (from DoclingDocument.iterate_items()'s level). "
        "Not populated for code-produced blocks like mcq_group. A generic structural signal -- e.g. two "
        "list_items at the same depth are more likely siblings (such as an MCQ's answer options) than two at "
        "different depths -- but not a guaranteed grouping on its own.",
    )
    marker: str | None = Field(
        default=None,
        description="Docling's own parsed list-item marker/enumeration (e.g. 'A.', '1.', '•'), kept SEPARATE "
        "from `text`. Confirmed directly against the real fixture that Docling frequently strips the marker out "
        "of `text` entirely rather than leaving it concatenated -- an option's `text` can be just 'Obstructive "
        "Jaundice' with 'A.' living only here. Only ever populated for list_item blocks.",
    )


class PageRangeEvidence(BaseModel):
    """Result of extracting one page range from a PDF via Docling."""

    source_path: str
    start_page: int
    end_page: int
    conversion_status: str
    blocks: list[EvidenceBlock]
    error_message: str | None = None


# --- LLM output models ---

DocumentStructure = Literal["table_of_contents", "index", "neither"]


class TOCEvidenceRef(BaseModel):
    page: int
    text: str


class TOCDetectionResult(BaseModel):
    """LLM's verdict on whether a page range contains a TOC/index."""

    document_structure: DocumentStructure
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[TOCEvidenceRef] = Field(
        description="Specific extracted text snippets (with page numbers) that support the verdict"
    )


# Distinguishes what kind of confidence we actually have in a topic's name,
# rather than silently treating "resolved" and "here's a number, name
# unknown" the same way:
#   resolved         - a legible name was found and is grounded in evidence.
#   name_unresolved  - the topic's position/number is known (we know it
#                       exists and roughly where), but no evidence source
#                       yielded a confident, legible name for it.
#   ambiguous        - more than one plausible, conflicting name was found
#                       and it isn't clear which is correct.
#   unidentified     - we don't even reliably know where to look (e.g. a
#                       book-page -> PDF-page mapping couldn't be built),
#                       so nothing further could be attempted.
ResolutionStatus = Literal["resolved", "name_unresolved", "ambiguous", "unidentified"]

# Which structural evidence actually supplied a topic's accepted name.
NameEvidenceSource = Literal["master_toc", "section_toc", "chapter_opening_page", "other"]


class Topic(BaseModel):
    id: str
    name: str
    level: int = Field(ge=1)
    parent_id: str | None = None
    source_pages: list[int] = Field(
        description="PDF page(s) where the evidence block that established this topic's existence/position "
        "(e.g. its row in the contents listing) was physically found -- Docling provenance, not the book's own "
        "printed page numbering"
    )
    declared_page_number: int | None = Field(
        default=None,
        description="The book's own printed page number this topic claims to start on, as written inside the "
        "contents/index listing itself (e.g. the '244' in a row reading 'Lipid Metabolism ... 244'). This is the "
        "book's internal numbering, NOT a PDF page index -- distinct from source_pages. Null if the listing didn't "
        "show a page number for this topic.",
    )
    resolution_status: ResolutionStatus = "resolved"
    name_evidence_source: NameEvidenceSource | None = Field(
        default=None, description="Which evidence source the accepted name came from; null unless resolution_status=resolved"
    )
    name_source_pages: list[int] = Field(
        default_factory=list,
        description="Page(s) of the specific block the accepted name was read from (may differ from source_pages)",
    )


class TopicExtractionResult(BaseModel):
    """LLM's structured extraction of the textbook's topic list."""

    # max_length becomes a maxItems constraint in the JSON schema handed to
    # Ollama, giving the grammar-constrained decoder a structural bound.
    # This is a technical safety cap against runaway/non-converging
    # generation (observed directly: one call passed 3500+ output tokens
    # with no sign of stopping for a ~43-chapter book), not a claim that no
    # real textbook has more topics than this. Set comfortably above
    # validation.py's default max_expected_topics (500, a human-facing
    # plausibility warning) so that warning path is actually reachable
    # instead of being pre-empted by this hard cap.
    topics: list[Topic] = Field(max_length=600)


# --- Validation models (deterministic, from validation.py) ---

IssueSeverity = Literal["error", "warning"]


class ValidationIssue(BaseModel):
    severity: IssueSeverity
    code: str
    message: str
    topic_id: str | None = None


class ValidationReport(BaseModel):
    passed: bool
    topic_count: int
    issues: list[ValidationIssue]


# --- Name recovery models (topic_extraction/name_recovery.py) ---

# Deliberately distinct spelling from Topic.resolution_status's
# "unidentified": this call only runs after a candidate page has already
# been deterministically located and confirmed (see
# page_numbering.confirm_expected_page), so "not_found" here means "the
# page was right, but no legible name was on it" -- which name_recovery.py
# maps back to the topic-level "name_unresolved", not "unidentified".
# Topic-level "unidentified" is reserved for when no candidate page could
# be located/confirmed at all, a case this call never even sees.
RecoveryStatus = Literal["resolved", "ambiguous", "not_found"]


class ChapterNameResolution(BaseModel):
    topic_id: str
    status: RecoveryStatus
    name: str | None = Field(default=None, description="The name found, verbatim/near-verbatim from evidence; null unless resolved")
    reason: str | None = Field(default=None, description="Brief note, especially for ambiguous/unidentified outcomes")


class ChapterNameRecoveryResult(BaseModel):
    resolutions: list[ChapterNameResolution] = Field(max_length=50)


# --- Stage 2: question-paper extraction models ---

# Generic "cite the page and text that grounds this claim" reference, used
# by every Stage 2 LLM call that has to ground a judgment in evidence
# (paper boundaries, metadata, sections). Deliberately not the Stage-1
# TOCEvidenceRef -- kept separate so Stage 1's schema is never touched by
# Stage 2 changes, even though the shape happens to be identical.
class EvidenceRef(BaseModel):
    page: int
    text: str


class PaperBoundaryCandidate(BaseModel):
    """LLM's claim that PDF pages [start_page, end_page] make up one
    distinct exam paper within a (possibly multi-paper) source PDF."""

    start_page: int
    end_page: int
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceRef] = Field(
        default_factory=list,
        description="Specific evidence (repeated headers, dates, numbering resets, etc.) that grounds this boundary",
    )


class PaperBoundaryDetectionResult(BaseModel):
    # If the whole document is a single paper, this is exactly one entry
    # spanning the whole page range -- never assumed by the caller, always
    # asserted by the model from evidence.
    papers: list[PaperBoundaryCandidate] = Field(min_length=1, max_length=100)


# Open-ended by design (Literal, not a hard-coded vocabulary of exam
# formats): "other"/"unknown" exist specifically so the model is never
# forced to misclassify something novel just because no better label fits.
SectionType = Literal["mcq", "long_answer", "short_answer", "essay", "short_essay", "other", "unknown"]


class DetectedSection(BaseModel):
    raw_label: str | None = Field(
        default=None,
        description="The section heading exactly as it appears in the evidence (e.g. 'II. Short Essay:'). "
        "Null only if a section boundary is inferred without any explicit heading text.",
    )
    section_type: SectionType
    start_page: int
    end_page: int
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class MetadataFieldExtraction(BaseModel):
    """One metadata field's reading, with its own grounding bundled in --
    not a shared pool of snippets the model can dump evidence into while
    leaving the typed value null (that's exactly the failure mode observed
    in the first real run: the model correctly cited "Date: 04-07-2025" in
    a shared evidence list yet left `date` itself null). Forcing value and
    evidence to live in the same object gives the model nowhere else to put
    the grounding, and the validator below makes "value without evidence"
    and "evidence without value" both a schema-validation failure that
    triggers a retry rather than silently passing through."""

    value: str | None = Field(default=None, description="As written in the evidence, not normalized/parsed")
    evidence: str | None = Field(
        default=None, description="The exact snippet this value was read from; null only if value is null"
    )
    source_pages: list[int] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _value_and_evidence_travel_together(self) -> "MetadataFieldExtraction":
        if self.value is not None and not self.evidence:
            raise ValueError("value is set but no grounding evidence was given")
        if self.value is None and self.evidence:
            raise ValueError("evidence was given but value is null")
        return self


class PaperStructureExtraction(BaseModel):
    """LLM's read of one already-isolated paper's own identifying metadata
    and internal section structure. Deliberately does NOT read the
    filename -- filename-derived evidence is merged in separately,
    deterministically, so the two provenances never get silently mixed
    (see question_extraction/metadata.py)."""

    exam_name: MetadataFieldExtraction = Field(default_factory=MetadataFieldExtraction)
    institution: MetadataFieldExtraction = Field(default_factory=MetadataFieldExtraction)
    subject: MetadataFieldExtraction = Field(default_factory=MetadataFieldExtraction)
    date: MetadataFieldExtraction = Field(default_factory=MetadataFieldExtraction)
    year: MetadataFieldExtraction = Field(default_factory=MetadataFieldExtraction)
    paper_identifier: MetadataFieldExtraction = Field(default_factory=MetadataFieldExtraction)
    other_identifiers: list[str] = Field(default_factory=list)
    sections: list[DetectedSection] = Field(default_factory=list, max_length=30)


QuestionType = Literal["mcq", "long_answer", "short_answer", "essay", "short_essay", "other", "unknown"]


class ExtractedQuestion(BaseModel):
    question_number: str | None = Field(
        default=None, description="Exactly as printed (e.g. '1', '2a', 'IV'); null if none is visible -- never invented"
    )
    question_text: str = Field(
        description="The question stem only -- do NOT include a leading number/label that belongs in "
        "question_number, and for an MCQ do NOT append its options here (those go in `options`)"
    )
    options: list[str] = Field(
        default_factory=list,
        description="For question_type=mcq only: each answer option's text, in the order printed, WITHOUT its "
        "own 'A./B./(a)/i.' marker. Empty for every non-MCQ question type. If the evidence bundles a stem with "
        "its options (an `mcq_group` block, or options immediately following a stem in the evidence), extract "
        "them as ONE question with this list populated -- never as separate questions, one per option.",
    )
    original_text: str | None = Field(
        default=None, description="Set only if correction_applied is true: the exact, uncorrected OCR text"
    )
    correction_applied: bool = False
    question_type: QuestionType
    source_pages: list[int]
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    section_index: int | None = Field(
        default=None, description="0-based index into the section list this question belongs to, if any"
    )
    ambiguous: bool = False
    ambiguity_note: str | None = None

    @model_validator(mode="after")
    def _correction_requires_original_text(self) -> "ExtractedQuestion":
        # A correction claim without the uncorrected text to back it up isn't
        # auditable -- treat it the same as any other malformed response
        # (validation failure -> retry) rather than silently accepting it.
        if self.correction_applied and not self.original_text:
            raise ValueError("correction_applied is true but original_text was not provided")
        return self


class QuestionExtractionResult(BaseModel):
    questions: list[ExtractedQuestion] = Field(default_factory=list, max_length=300)


# --- Stage 3: topic classification models ---
# (defined here, ahead of Stage 2's deterministic records below, only
# because Question -- a Stage 2 record -- carries a ClassificationStatus
# field and needs the type available above its own definition)

# What the LLM itself may report for one question. Deliberately does NOT
# include "unclassified" -- that value only ever arises deterministically
# (topic_classification/reconciliation.py), for a question the LLM's
# response never covered, or covered with a topic_id that wasn't in the
# list it was given. The LLM is never asked to say "unclassified" itself;
# it always has to pick classified/no_match/ambiguous.
ClassificationVerdict = Literal["classified", "no_match", "ambiguous"]

# The persisted status on a Question record. Superset of
# ClassificationVerdict -- see that type's comment for why "unclassified"
# is deterministic-only.
ClassificationStatus = Literal["classified", "no_match", "ambiguous", "unclassified"]


class QuestionTopicClassification(BaseModel):
    """One question's classification verdict against the textbook's topic
    list. question_id is echoed back (rather than relying on list order)
    so a response can be safely reconciled against the batch of questions
    actually asked about -- see topic_classification/reconciliation.py.

    Doubles as the reconciled/persisted record: reconciliation.py builds
    more of these with status="unclassified" for a question the LLM's
    response omitted, or gave an invented topic_id for. The LLM's own
    prompt only ever asks it to choose classified/no_match/ambiguous (see
    ClassificationVerdict) -- status accepts the full ClassificationStatus
    superset here only so reconciliation can reuse this same model rather
    than needing a parallel one."""

    question_id: str
    status: ClassificationStatus
    topic_id: str | None = Field(
        default=None, description="Must be one of the topic_id values given in the prompt; null unless status=classified"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str | None = Field(default=None, description="Brief justification, especially for no_match/ambiguous")

    @model_validator(mode="after")
    def _topic_id_matches_status(self) -> "QuestionTopicClassification":
        if self.status == "classified" and not self.topic_id:
            raise ValueError("status is classified but topic_id was not provided")
        if self.status != "classified" and self.topic_id:
            raise ValueError(f"status is {self.status!r} but topic_id was provided -- only classified may set it")
        return self


class TopicClassificationResult(BaseModel):
    # One call classifies one exam paper's worth of questions (see
    # topic_classification/classify.py) -- capped at the same ceiling as
    # QuestionExtractionResult since a paper's question count is already
    # bounded there.
    classifications: list[QuestionTopicClassification] = Field(default_factory=list, max_length=300)


# --- Stage 2: deterministic records (never touched by the LLM) ---


class FilenameMetadata(BaseModel):
    """Purely deterministic (regex-based) read of a filename. No exam- or
    institution-specific rules -- just generic year/date pattern matching
    and tokenization, so it works on any filename convention."""

    raw_filename: str
    candidate_years: list[str] = Field(default_factory=list)
    candidate_dates: list[str] = Field(default_factory=list)
    tokens: list[str] = Field(default_factory=list)


class MetadataFieldValue(BaseModel):
    """One metadata field's value(s) from each provenance, kept side by
    side rather than resolved -- see question_extraction/metadata.py.
    Document-side evidence/confidence/pages travel with the value instead
    of living in a separate shared list, mirroring MetadataFieldExtraction."""

    document_value: str | None = None
    document_evidence: str | None = None
    document_source_pages: list[int] = Field(default_factory=list)
    document_confidence: float = 0.0
    filename_value: str | None = None
    conflict: bool = False


class PaperMetadata(BaseModel):
    exam_name: MetadataFieldValue
    institution: MetadataFieldValue
    subject: MetadataFieldValue
    date: MetadataFieldValue
    year: MetadataFieldValue
    paper_identifier: MetadataFieldValue
    other_identifiers_document: list[str] = Field(default_factory=list)
    other_identifiers_filename: list[str] = Field(default_factory=list)


class Section(BaseModel):
    section_id: str
    paper_id: str
    raw_label: str | None
    section_type: SectionType
    start_page: int
    end_page: int
    confidence: float


class Question(BaseModel):
    """The durable, standalone raw-occurrence record. Deliberately carries
    denormalized copies of paper-level identity (source_filename/hash,
    exam_name/institution/subject/paper_year/paper_date) so that later
    stages -- topic classification, then semantic deduplication/frequency
    analysis -- never need to join back through Paper just to group or
    display a question. The full, provenance-preserving metadata (with
    document vs. filename provenance and conflict flags) still lives on
    Paper.metadata; these are a convenience best-available reading (document
    value preferred, filename value as fallback), not a replacement for it.

    topic_id/topic_name/topic_confidence/classification_status are additive
    fields written by Stage 3 (topic_classification/), never by Stage 2 --
    a freshly (re-)extracted question always starts classification_status=
    "unclassified" until Stage 3 is run against it. canonical_question_id
    (deduplication) is reserved for Stage 5 and intentionally NOT added
    yet."""

    question_id: str
    paper_id: str
    section_id: str | None
    source_filename: str
    source_file_hash: str
    exam_name: str | None = None
    institution: str | None = None
    subject: str | None = None
    paper_year: str | None = None
    paper_date: str | None = None
    question_number: str | None
    question_text: str
    options: list[str] = Field(default_factory=list)
    original_text: str | None = None
    correction_applied: bool = False
    question_type: QuestionType
    source_pages: list[int]
    extraction_confidence: float
    ambiguous: bool = False
    ambiguity_note: str | None = None
    topic_id: str | None = None
    topic_name: str | None = None
    topic_confidence: float | None = None
    classification_status: ClassificationStatus = "unclassified"
    # canonical_question_id is deliberately NOT a field here -- Stage 4
    # (semantic deduplication) links a question to its canonical group via a
    # separate question_canonical_links table (see database.py), not by
    # adding a column to this already-frozen record. That keeps Stage 4
    # fully additive: nothing about the `questions` table itself changes.


PaperStatus = Literal["success", "partial", "failed"]


class Paper(BaseModel):
    paper_id: str
    file_hash: str
    start_page: int
    end_page: int
    metadata: PaperMetadata
    sections: list[Section] = Field(default_factory=list)
    boundary_confidence: float
    boundary_evidence: list[EvidenceRef] = Field(default_factory=list)
    status: PaperStatus = "success"
    notes: list[str] = Field(default_factory=list)


class PaperBoundaryIssue(BaseModel):
    code: Literal["overlap_clipped", "gap_filled", "gap_unassigned"]
    message: str
    pages: list[int] = Field(default_factory=list)


class QuestionPaperFileResult(BaseModel):
    file_path: str
    file_hash: str
    total_pages: int
    model_used: str
    papers: list[Paper] = Field(default_factory=list)
    boundary_issues: list[PaperBoundaryIssue] = Field(default_factory=list)
    status: PaperStatus = "success"


# --- Stage 4: semantic deduplication models ---
#
# Two questions are candidates for the LLM judge only after deterministic
# exact-duplicate matching and embedding-based candidate generation (see
# exampapersorter/deduplication/). The LLM's ONLY job for a candidate pair
# is answering "do these ask for substantially the same answer/task?" --
# NOT "are these about the same topic?". See deduplication/semantic_judge.py
# for the prompt that enforces this distinction.


class QuestionEmbedding(BaseModel):
    """One question occurrence's embedding vector, plus enough identity to
    know when it's stale. content_hash is a hash of the exact normalized
    text that was embedded (see deduplication/normalize.py) -- if a
    question's text ever changes (a genuine Stage 2 re-extraction, not a
    resumed run), the cached vector no longer matches and must be
    recomputed; model/version changing has the same effect. See
    database.get_question_embedding for how staleness is detected."""

    question_id: str
    model: str
    version: str
    content_hash: str
    vector: list[float]


# What the LLM itself reports for one candidate pair. Deliberately three-
# valued rather than a bare boolean -- "uncertain" lets the model express
# low confidence directly instead of being forced into a coin-flip
# true/false that dedup_min_merge_confidence would then have to catch after
# the fact. Mirrors topic_classification's ClassificationVerdict pattern
# (classified/no_match/ambiguous).
PairVerdict = Literal["same_question", "different_question", "uncertain"]


class PairEquivalenceVerdict(BaseModel):
    """One candidate pair's semantic-equivalence verdict. pair_id is echoed
    back (built as f"{question_id_a}::{question_id_b}" with a<b, see
    deduplication/semantic_judge.py) so a response can be reconciled against
    the exact batch of pairs actually asked about -- mirrors
    QuestionTopicClassification.question_id's role in Stage 3."""

    pair_id: str
    verdict: PairVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str | None = Field(default=None, description="Brief justification, especially for different_question/uncertain")
    evidence: str | None = Field(
        default=None, description="What in the two questions' text actually grounds this verdict"
    )


class SemanticEquivalenceResult(BaseModel):
    # One call judges a batch of candidate pairs (see
    # Config.dedup_pair_batch_size) -- capped the same way
    # TopicClassificationResult/QuestionExtractionResult are.
    verdicts: list[PairEquivalenceVerdict] = Field(default_factory=list, max_length=300)


# Where a pair decision came from -- kept on the persisted record (not just
# in a log message) so the audit trail can distinguish "these were
# byte-identical after normalization, no LLM involved" from "the LLM judged
# these as equivalent" without re-deriving it.
DedupDecisionSource = Literal["exact_duplicate", "llm_judge"]


class DedupPairDecision(BaseModel):
    """Persisted, resumable record of one pair's dedup verdict -- doubles as
    the LLM-call cache (so a resumed run never re-asks about a pair it
    already has a verdict for, see database.get_dedup_pair_decision) and as
    the audit log (see Config section 27/output_writer's duplicate_groups
    output). question_id_a/question_id_b are always stored with a < b
    (lexicographic) so the same unordered pair always hashes to the same
    cache key regardless of which order it was generated in."""

    question_id_a: str
    question_id_b: str
    decision_source: DedupDecisionSource
    verdict: PairVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str | None = None


# Status of a canonical group as a whole, distinct from PairVerdict (which
# is per-pair). "exact_duplicate": formed purely by deterministic
# normalized-text matching, no LLM involved. "semantic_merge": at least one
# LLM verdict=same_question (at/above dedup_min_merge_confidence) was used
# to union occurrences together. "singleton": exactly one occurrence, no
# candidate pair reached a merge decision. "needs_review": exactly one
# occurrence, but at least one candidate pair involving it came back
# verdict=uncertain, or same_question below the confidence floor -- flagged
# for human inspection rather than silently left indistinguishable from an
# ordinary singleton.
DedupStatus = Literal["exact_duplicate", "semantic_merge", "singleton", "needs_review"]


class CanonicalQuestion(BaseModel):
    """A semantic-identity group over one or more raw question occurrences.
    The `questions` table rows are never deleted or altered by Stage 4 --
    this is a separate, additive record; question_canonical_links (see
    database.py) is what actually associates occurrences with a group.

    occurrence_count/unique_paper_count/years/source_question_ids are
    DERIVED by joining question_canonical_links against `questions` at read
    time (see database.get_canonical_question_detail) -- NOT columns on the
    canonical_questions table itself, so they can never drift out of sync
    with the underlying links."""

    canonical_question_id: str
    canonical_question_text: str
    question_type: QuestionType
    topic_id: str | None = None
    topic_name: str | None = None
    representative_question_id: str = Field(
        description="Which source occurrence's text was chosen as canonical_question_text (see "
        "deduplication/grouping.py's deterministic selection rule) -- never an LLM rewrite/paraphrase"
    )
    dedup_confidence: float | None = Field(
        default=None, description="1.0 for exact_duplicate; min confidence among the merge edges used for "
        "semantic_merge; null for singleton/needs_review (no confirmed merge)"
    )
    dedup_status: DedupStatus
    occurrence_count: int = 1
    unique_paper_count: int = 1
    years: list[str] = Field(default_factory=list)
    source_question_ids: list[str] = Field(default_factory=list)


# --- Stage 5: frequency analysis / aggregation models ---
#
# Purely derived DTOs -- Stage 5 introduces no new source-of-truth table.
# Every field here is computed at read time from Stage 1-4 data (topics,
# `questions`, `canonical_questions` + `question_canonical_links`, `papers`)
# by exampapersorter/frequency_analysis/aggregate.py, never stored as an
# independent counter that could drift out of sync with that data.


class QuestionTypeCounts(BaseModel):
    """Occurrence counts broken down by question_type. Always carries every
    QuestionType key (see schemas.QuestionType) even when a given slice has
    zero of that type, rather than omitting the key -- so a consumer (JSON
    UI or Markdown renderer) can rely on the key always being present
    instead of defaulting missing keys itself."""

    mcq: int = 0
    long_answer: int = 0
    short_answer: int = 0
    essay: int = 0
    short_essay: int = 0
    other: int = 0
    unknown: int = 0

    def total(self) -> int:
        return self.mcq + self.long_answer + self.short_answer + self.essay + self.short_essay + self.other + self.unknown


class DataIntegrityReport(BaseModel):
    """Reconciliation check between raw occurrences and canonical links --
    see database.py's question_canonical_links docstring for why this
    should always be a 1:1 total under normal operation. Surfaced rather
    than silently assumed, per project notes (section 22)."""

    total_raw_occurrences: int
    total_canonical_links: int
    distinct_canonical_question_ids_in_links: int
    orphaned_occurrences: list[str] = Field(
        default_factory=list, description="question_id values present in `questions` with no canonical link"
    )
    duplicate_links: list[str] = Field(
        default_factory=list, description="question_id values that appear in more than one canonical group's members"
    )
    reconciled: bool


class CanonicalQuestionFrequency(BaseModel):
    """One canonical question's full frequency-analysis record. Extends
    CanonicalQuestion's already-hydrated occurrence_count/unique_paper_count/
    years/source_question_ids with the per-occurrence provenance
    (paper_ids/source_filenames) those don't carry, plus an honest
    multi-value question_types (see project notes section 21: a canonical
    group's member occurrences are not pretended to share one type when
    they don't) and best-effort first/latest year ordering (derived only
    from the already-extracted year strings themselves -- never invented,
    see first_and_latest_year)."""

    canonical_question_id: str
    canonical_question_text: str
    question_types: list[str]
    topic_id: str | None = None
    topic_name: str | None = None
    occurrence_count: int
    unique_paper_count: int
    years: list[str]
    first_year: str | None = None
    latest_year: str | None = None
    paper_ids: list[str]
    source_filenames: list[str]
    source_question_ids: list[str]
    dedup_status: str
    dedup_confidence: float | None = None


class RankedCanonicalQuestion(BaseModel):
    """One row of the deterministic most-repeated-questions ranking (project
    notes section 15) -- rank is 1-based and assigned by
    aggregate.rank_canonical_questions's fixed sort key, never LLM judgment."""

    rank: int
    canonical_question_id: str
    canonical_question_text: str
    occurrence_count: int
    unique_paper_count: int
    years: list[str]
    question_types: list[str]
    topic_id: str | None = None
    topic_name: str | None = None


class TopicRepeatedQuestion(BaseModel):
    """One canonical question repeated within a single topic's own
    occurrence set. occurrences_in_topic is a chapter-local recount (only
    occurrences actually classified into THIS topic), deliberately not the
    canonical question's global occurrence_count -- so it always reconciles
    against the enclosing TopicFrequencySummary.total_occurrences even in
    the rare case a canonical group's members span more than one topic.

    source_filenames carries each member occurrence's own source PDF
    filename (never invented -- see Question.source_filename), so a reader
    can still tell repeated occurrences apart by paper even when `years` is
    empty because none of those papers had a resolvable date. A fake/
    placeholder year is deliberately never fabricated to fill that gap."""

    canonical_question_id: str
    canonical_question_text: str
    occurrences_in_topic: int
    question_types: list[str]
    years: list[str]
    source_filenames: list[str] = Field(default_factory=list)


class TopicFrequencySummary(BaseModel):
    """One textbook topic's full frequency-analysis record. Included for
    EVERY topic from Stage 1's topic list, including topics with zero
    questions (project notes section 14/27) -- total_occurrences=0 and an
    empty type_counts/repeated_canonical_questions is the honest
    zero-question representation, not an omitted row. level/parent_id are
    copied from the source Topic so the hierarchy is reconstructable
    without a second lookup."""

    topic_id: str
    topic_name: str
    level: int
    parent_id: str | None = None
    total_occurrences: int
    unique_canonical_questions: int
    type_counts: QuestionTypeCounts
    paper_count: int
    years: list[str]
    repeated_canonical_questions: list[TopicRepeatedQuestion] = Field(default_factory=list)


class RankedTopic(BaseModel):
    """One row of the deterministic most-tested-chapters ranking (project
    notes section 16). Only topics with at least one occurrence are ranked
    here -- see TopicFrequencySummary for the complete, zero-question-
    inclusive topic list."""

    rank: int
    topic_id: str
    topic_name: str
    total_occurrences: int
    unique_canonical_questions: int


class TypeRepetitionStats(BaseModel):
    """Repetition breakdown for one question_type, corpus-wide (project
    notes section 18)."""

    total_occurrences: int
    unique_canonical_questions: int
    repeated_canonical_questions: int


class FrequencyAnalysisSummary(BaseModel):
    """Stage 5's single structured output -- everything summary.json needs
    to drive a future UI without reparsing Markdown (project notes section
    25). Built once per run by frequency_analysis.aggregate.
    build_frequency_analysis, purely from already-persisted Stage 1-4 data;
    running Stage 5 twice against unchanged Stage 1-4 state reproduces this
    object exactly (deterministic, no LLM/embedding calls)."""

    textbook_file_hash: str
    total_question_occurrences: int
    total_canonical_questions: int
    total_papers: int
    topic_count: int
    question_type_distribution: QuestionTypeCounts
    classification_status_distribution: dict[str, int]
    years_represented: list[str]
    year_conflict_paper_ids: list[str] = Field(
        default_factory=list,
        description="paper_id values whose Stage 2 year metadata was flagged as conflicting between document and "
        "filename provenance (Paper.metadata.year.conflict) -- exposed, never silently resolved, per project "
        "notes section 10/28",
    )
    most_repeated_questions: list[RankedCanonicalQuestion] = Field(default_factory=list)
    most_tested_topics: list[RankedTopic] = Field(default_factory=list)
    topic_summaries: list[TopicFrequencySummary] = Field(default_factory=list)
    repetition_by_question_type: dict[str, TypeRepetitionStats] = Field(default_factory=dict)
    no_match_questions: list[str] = Field(default_factory=list, description="question_id values with classification_status=no_match")
    ambiguous_questions: list[str] = Field(default_factory=list, description="question_id values with classification_status=ambiguous")
    unclassified_questions: list[str] = Field(
        default_factory=list, description="question_id values with classification_status=unclassified"
    )
    data_integrity: DataIntegrityReport


# --- Stage 6: final user-facing report models ---
#
# Purely derived DTOs -- Stage 6 introduces no new source-of-truth data and
# makes zero LLM/embedding calls. Every field here is either copied straight
# from an already-computed FrequencyAnalysisSummary, reused verbatim
# (TopicFrequencySummary below is the SAME Stage 5 type, not a redefinition),
# or a different sort/grouping of that same already-computed data (e.g. a
# second topic ranking by unique_canonical_questions instead of
# total_occurrences) -- never a new count, classification, or dedup
# decision. See final_analysis/build.py.


class QuestionTypeShare(BaseModel):
    """One question_type's share of total occurrences. Occurrence-level,
    not canonical-question-level -- the percentage is computed purely for
    display from FrequencyAnalysisSummary.question_type_distribution, which
    remains the source of truth for the counts themselves."""

    question_type: str
    label: str
    occurrence_count: int
    percentage: float = Field(description="Rounded to 1 decimal place; 0.0 if total_occurrences is 0")


class QuestionTypeDistributionReport(BaseModel):
    total_occurrences: int
    by_type: list[QuestionTypeShare]


class RepeatedQuestionReportItem(BaseModel):
    """One repeated canonical question, presentation-ready. Extends Stage
    5's RankedCanonicalQuestion with the paper-level provenance and dedup
    status/confidence that ranking doesn't carry -- both read directly off
    Stage 4's own CanonicalQuestion record (via frequency_analysis.
    aggregate.build_canonical_frequency, reused rather than reimplemented),
    never recomputed. canonical_question_text is copied verbatim from the
    representative source occurrence -- never LLM-rewritten or "cleaned
    up", even if it contains visible OCR corruption."""

    rank: int
    canonical_question_id: str
    canonical_question_text: str
    occurrence_count: int
    unique_paper_count: int
    years: list[str]
    question_types: list[str]
    topic_id: str | None = None
    topic_name: str | None = None
    dedup_status: str
    dedup_confidence: float | None = None
    paper_ids: list[str]
    source_filenames: list[str]
    source_question_ids: list[str]


class TopicRankingItem(BaseModel):
    """One row of a most-tested-chapters ranking. Two separate rankings are
    produced over the SAME already-computed TopicFrequencySummary rows --
    by total occurrences, and separately by unique canonical questions --
    rather than one blended ranking, since they answer different questions
    and can disagree (a topic with heavy repetition can have fewer unique
    questions than a topic with less repetition but more variety)."""

    rank: int
    topic_id: str
    topic_name: str
    total_occurrences: int
    unique_canonical_questions: int


class UnclassifiedQuestionReportItem(BaseModel):
    """One question with classification_status in {no_match, ambiguous},
    hydrated from its full `questions` row -- Stage 5 only carries the bare
    question_id for these. `reason` is deliberately not included: Stage 3
    records a per-verdict reason only in its own call log
    (topic_classification_log), not on the persisted Question row, and
    Stage 6 does not add a new read path into that log just to surface it
    -- classification_status is the honest status signal that IS reliably
    available for every such question."""

    question_id: str
    question_text: str
    question_type: str
    classification_status: str
    source_filename: str
    paper_id: str
    paper_year: str | None = None
    topic_confidence: float | None = None


class UntestedTopicItem(BaseModel):
    """A valid Stage 1 textbook topic that simply had no classified
    questions in this exam-paper collection -- not a sign Stage 1 failed to
    find it."""

    topic_id: str
    topic_name: str
    level: int
    parent_id: str | None = None


class ExecutiveSummary(BaseModel):
    textbook_file_hash: str
    total_papers: int
    total_question_occurrences: int
    total_canonical_questions: int
    total_topics: int
    topics_tested: int
    topics_untested: int
    repeated_canonical_question_count: int
    years_represented: list[str]
    no_match_count: int
    ambiguous_count: int
    unclassified_count: int
    data_integrity_reconciled: bool


class DataQualityNotes(BaseModel):
    """Fixed, honest caveats about known Stage 2-4 limitations. `notes` is
    static text (not derived from the data), except the counts, which are
    copied from the summary so the section stays numerically consistent
    with the rest of the report rather than drifting out of sync."""

    no_match_count: int
    ambiguous_count: int
    unclassified_count: int
    year_conflict_paper_count: int
    data_integrity_reconciled: bool
    notes: list[str]


class AnalysisJob(BaseModel):
    """One `analyze` run's identity and lifecycle state, for GUI pause/resume
    only (see database.py's analysis_jobs table) -- Stage 1-6 data itself
    never lives here, this is purely "which run is this and is it done".
    job_id is a deterministic hash of (topic_source_type, topic_source_path,
    question_papers_dir) -- see analyze_pipeline.compute_job_id -- so
    re-running `analyze` against the exact same inputs always maps back to
    the same row instead of accumulating duplicates.

    status is one of "running" (in progress, or a prior run never reached a
    terminal state -- e.g. a crash -- which is intentionally
    indistinguishable from "still running" here, since both mean
    "incomplete, safe to resume"), "paused" (stopped cleanly because of an
    account-level LLM quota/credit exhaustion), "failed" (stopped cleanly
    for some other reason), or "completed"."""

    job_id: str
    topic_source_type: str  # "textbook" | "index"
    topic_source_path: str
    question_papers_dir: str
    status: str
    pause_reason: str | None = None
    created_at: str
    updated_at: str


class FinalAnalysisReport(BaseModel):
    """Stage 6's single structured output -- everything analysis.json needs
    to drive a future GUI without reparsing Markdown. Built once per run by
    final_analysis.build.build_final_report, purely from an
    already-computed FrequencyAnalysisSummary plus a small amount of
    additional read-only hydration (canonical questions' paper-level
    provenance, no_match/ambiguous questions' full text) -- zero LLM calls,
    zero embedding calls, fully deterministic. topic_analysis reuses Stage
    5's own TopicFrequencySummary type verbatim (tested topics only, i.e.
    total_occurrences > 0) so the textbook's section/group/leaf hierarchy
    (level/parent_id) is reconstructable exactly as Stage 5 computed it."""

    textbook_file_hash: str
    summary: ExecutiveSummary
    question_type_distribution: QuestionTypeDistributionReport
    most_repeated_questions: list[RepeatedQuestionReportItem]
    most_tested_topics_by_occurrences: list[TopicRankingItem]
    most_tested_topics_by_unique_questions: list[TopicRankingItem]
    topic_analysis: list[TopicFrequencySummary]
    untested_topics: list[UntestedTopicItem]
    no_match_questions: list[UnclassifiedQuestionReportItem]
    ambiguous_questions: list[UnclassifiedQuestionReportItem]
    data_quality: DataQualityNotes
