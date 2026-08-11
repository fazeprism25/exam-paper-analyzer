"""Local embedding generation for Stage 4 candidate retrieval.

Uses fastembed (ONNX runtime, already present in this environment) --
entirely local, no external embedding API call, no per-question network
cost, nothing routed through the LLM abstraction. Embeddings are for
candidate retrieval ONLY (see candidates.py); the LLM judge is the final
semantic authority, an embedding similarity score is never treated as a
confirmed duplicate by itself.

Persistent + resumable: every computed vector is cached in the database
(Database.get_question_embedding / save_question_embedding), keyed on
(question_id, embedding_model, embedding_model_version) AND a content_hash
of the exact text that was embedded -- so a crash mid-run, or simply
rerunning the pipeline, never recomputes a vector that's still valid, while
a genuinely changed question (re-extraction) or a deliberately swapped
embedding model/version correctly invalidates the cached one.
"""
from __future__ import annotations

import hashlib
import logging

from exampapersorter.config import Config
from exampapersorter.database import Database
from exampapersorter.schemas import Question, QuestionEmbedding

logger = logging.getLogger(__name__)


def build_embedding_text(question: Question) -> str:
    """The exact text handed to the embedding model. Prefixing with the
    question type is a cheap signal that pulls same-type paraphrases closer
    together and different-type text apart in embedding space, without
    requiring a fine-tuned model -- consistent with treating question_type
    as a strong (not absolute) signal throughout Stage 4 (see
    candidates.py). Options are included for MCQs so two stems that read
    alike but have different answer choices don't score as spuriously
    similar as they would from the stem alone."""
    parts = [f"[{question.question_type}] {question.question_text}"]
    parts.extend(question.options)
    return "\n".join(parts)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _to_float_list(vector) -> list[float]:
    """fastembed yields numpy arrays; a test double may hand back a plain
    list. json-serializing a numpy scalar (e.g. numpy.float32) raises, so
    normalize to native Python floats regardless of which we got."""
    if hasattr(vector, "tolist"):
        return vector.tolist()
    return [float(x) for x in vector]


_model_cache: dict[tuple[str, str], object] = {}


def _get_model(config: Config):
    """Lazily loads (and caches in-process, keyed by model name + cache
    dir) the configured fastembed model. Imported inside the function, not
    at module load, so importing this module doesn't pull in
    fastembed/onnxruntime for any code path that never needs an
    embedding (e.g. every Stage 1-3 test, or a dedup run that resumes
    entirely from cached vectors)."""
    from fastembed import TextEmbedding

    key = (config.embedding_model, str(config.embedding_cache_dir))
    if key not in _model_cache:
        config.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        _model_cache[key] = TextEmbedding(model_name=config.embedding_model, cache_dir=str(config.embedding_cache_dir))
    return _model_cache[key]


def get_or_compute_embeddings(
    questions: list[Question], config: Config, db: Database
) -> dict[str, QuestionEmbedding]:
    """Cache-aware for every question in `questions`: already-cached,
    still-fresh vectors are reused as-is; only the uncached/stale remainder
    is sent to the embedding model, and in a SINGLE batched call (fastembed
    batches internally) rather than one model invocation per question."""
    result: dict[str, QuestionEmbedding] = {}
    to_compute: list[tuple[Question, str, str]] = []  # (question, embedding_text, content_hash)

    for q in questions:
        text = build_embedding_text(q)
        current_hash = content_hash(text)
        cached = db.get_question_embedding(q.question_id, config.embedding_model, config.embedding_model_version)
        if cached is not None and cached.content_hash == current_hash:
            result[q.question_id] = cached
        else:
            to_compute.append((q, text, current_hash))

    if to_compute:
        model = _get_model(config)
        vectors = list(model.embed([t for _, t, _ in to_compute]))
        for (q, _text, h), vector in zip(to_compute, vectors):
            embedding = QuestionEmbedding(
                question_id=q.question_id, model=config.embedding_model, version=config.embedding_model_version,
                content_hash=h, vector=_to_float_list(vector),
            )
            db.save_question_embedding(embedding)
            result[q.question_id] = embedding

    logger.info(
        "Embeddings: %d reused from cache, %d newly computed (model=%s, version=%s)",
        len(questions) - len(to_compute), len(to_compute), config.embedding_model, config.embedding_model_version,
    )
    return result
