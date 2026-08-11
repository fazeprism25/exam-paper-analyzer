"""Central configuration for the pipeline.

Every tunable that affects behavior or output lives here, not scattered
through the modules that use it. Nothing here is specific to any one
textbook or exam-paper set -- textbook_path/question_papers_dir are set
per-run (CLI args), not hard-coded defaults.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ReloadPolicy = Literal["persistent", "reload_per_paper", "reload_every_n_calls", "reload_on_failure"]
Provider = Literal["openrouter", "ollama"]

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OPENROUTER_MODELS_FILE = _REPO_ROOT / "config" / "openrouter_models.json"

# Centralized fallback pool, used only if DEFAULT_OPENROUTER_MODELS_FILE is
# missing or unreadable -- the file (not this constant) is the intended
# place to edit the pool. Kept in sync with config/openrouter_models.json;
# see that file's _comment for selection criteria. The SAME pool is used
# for every LLM task (extraction, classification, dedup, ...) -- prompts
# and Pydantic schemas differ by task, the model pool does not.
_BUILTIN_OPENROUTER_MODELS: tuple[str, ...] = (
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-31b-it:free",
    "inclusionai/ling-3.0-tiny:free",
    "poolside/laguna-s-2.1:free",
    "poolside/laguna-xs-2.1:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
)


def load_openrouter_models(path: Path = DEFAULT_OPENROUTER_MODELS_FILE) -> tuple[str, ...]:
    """Ordered model-fallback pool, read from `path` so it can be updated
    without a code change. Falls back to _BUILTIN_OPENROUTER_MODELS if the
    file is missing/malformed, so a deleted or corrupted config file
    degrades gracefully instead of crashing startup."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        models = tuple(data["models"])
        if models:
            return models
        logger.warning("%s has an empty \"models\" list; using built-in default pool", path)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Could not load OpenRouter model pool from %s (%s); using built-in default pool", path, exc)
    return _BUILTIN_OPENROUTER_MODELS


@dataclass(frozen=True)
class Config:
    # --- Provider selection ---
    # "openrouter" is the normal/default backend -- a user only needs their
    # own OpenRouter API key, no local install. "ollama" remains available
    # as an optional local dev/testing backend (see llm_providers/ollama.py)
    # but is never required for normal use.
    llm_provider: Provider = "openrouter"

    # --- OpenRouter ---
    # Never hard-code a key here -- this is None until the user supplies
    # their own (CLI flag / env var / config file), see cli.py.
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_models: tuple[str, ...] = field(default_factory=load_openrouter_models)
    # Optional attribution headers OpenRouter's docs recommend (used for
    # their own leaderboards, not required for calls to function).
    openrouter_http_referer: str | None = None
    openrouter_app_title: str | None = "exampapersorter"
    # Bump this when the pool composition changes meaningfully -- it's part
    # of Config.model_identifier, the DB cache key, so bumping it forces
    # fresh verdicts instead of silently reusing results cached under a
    # since-changed pool.
    openrouter_pool_version: str = "v1"
    # Per-HTTP-request timeout, in seconds. Generous because free-tier
    # models can be slow/queued; call_structured's own llm_retry_count is
    # the bound on total wall-clock time, not this.
    openrouter_timeout_seconds: float = 120.0

    # --- LLM call behavior shared by every provider (call_structured) ---
    # Bounded retry count for one call_structured() invocation -- covers
    # both providers' failure modes (network/HTTP errors, malformed/
    # truncated structured output). For OpenRouter this ALSO determines how
    # many times the model-fallback rotation advances (see
    # llm_providers/openrouter.py), since each retry after a validation
    # failure rotates which pool model is tried first.
    llm_retry_count: int = 3
    llm_think: bool = False  # disable qwen3 thinking mode for structured extraction calls (Ollama only)
    # Hard backstop on response length -- Ollama's `num_predict` / OpenRouter's
    # `max_tokens`, whichever provider is active. Without this, a
    # JSON-schema-constrained generation that isn't converging (e.g. the
    # model padding a list with near-duplicate entries instead of stopping)
    # can run for many minutes with no natural end -- observed directly on
    # Ollama: one topic-extraction call hit 3500+ output tokens at ~3.4
    # tok/s with no sign of stopping, for a book with ~43 chapters that
    # should need a few hundred.
    llm_num_predict: int = 6000
    # If a response looks truncated (see llm_client._looks_truncated) rather
    # than merely malformed, the retry strategy is different from a plain
    # retry: bump num_predict for the next attempt instead of hoping the
    # same ceiling produces a shorter response. Multiplier is applied once
    # per truncated attempt, capped at llm_num_predict_max so a genuinely
    # non-converging generation can't be retried into an unbounded runtime.
    # This does NOT replace the num_predict ceiling's purpose (a backstop
    # against non-converging generation, see llm_num_predict's own comment)
    # -- it only widens it specifically in response to observed evidence
    # that THIS call's output didn't fit, not as a blanket increase.
    llm_truncation_retry_multiplier: float = 1.5
    llm_num_predict_max: int = 16000
    # `seed` fixes the RNG used for sampling. At temperature=0 the sampler is
    # greedy (argmax) regardless of seed, so this is NOT expected to fully
    # eliminate run-to-run variance -- observed directly: the same evidence
    # and prompt at temperature=0 produced materially different topic
    # extractions across runs (clean names vs. garbled vs. numeric
    # placeholders). The likely remaining cause is floating-point
    # non-associativity in batched/multi-threaded matrix ops (a known
    # llama.cpp/Ollama characteristic, independent of sampling RNG), which a
    # seed cannot fix. Kept configurable so the reproducibility experiment
    # (see scripts/reproducibility_experiment.py) can measure seed's actual
    # effect rather than assuming it. None = omit the option entirely
    # (Ollama's own default RNG seeding), preserving prior behavior.
    llm_seed: int | None = None

    # --- Ollama-only knobs (no effect when llm_provider == "openrouter") ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    # Ollama defaults to a 4096-token context window regardless of what a model
    # supports -- it does NOT auto-scale to the model's max. Our evidence prompts
    # can run several thousand tokens, so this must be requested explicitly or
    # the model silently loses the start of its input. Tune alongside
    # evidence_format.MAX_EVIDENCE_CHARS and watch system RAM: a larger context
    # window means a larger KV cache.
    # Raised from 8192 once Topic gained 4 more fields (resolution_status,
    # name_evidence_source, name_source_pages, declared_page_number) -- a
    # ~65-topic response now needs several thousand more output tokens, and
    # num_ctx must cover prompt + response combined. Kept well short of
    # doubling to 16384 given this machine's tight free RAM (~2GB) and the
    # KV-cache cost of a larger context.
    llm_num_ctx: int = 12000

    # `ollama stop <model>` unload/reload policy for call_structured. The
    # Stage 1 reproducibility investigation (see
    # scripts/reload_isolated_trials.py and
    # output/reproducibility_experiment/) found that on this machine, a
    # freshly-loaded model gives a clean deterministic result on its FIRST
    # call, then reliably drifts into a persistent degraded state
    # (deterministic truncated/invalid JSON) on every subsequent call in
    # the same session -- and that reloading before a call restores clean
    # output. That was measured with Stage 1's single call per run; it is
    # NOT yet measured at Stage 2's call volume (many calls per paper), and
    # reload itself is not free (the first post-reload call in that
    # experiment took ~1450s on this hardware, which is presumably mostly
    # model load, not generation). So:
    #   - "persistent": never reload. Stage 1's tested behavior, kept as
    #     the default here so nothing about Stage 1 changes.
    #   - "reload_per_paper": unload before starting each new paper.
    #   - "reload_every_n_calls": unload after every llm_reload_every_n_calls
    #     successful calls, regardless of paper boundaries.
    #   - "reload_on_failure": only unload when a call's structured output
    #     fails to validate, before the next retry attempt. This is the
    #     cheapest policy that still uses the Stage 1 finding -- it never
    #     pays a reload's cost on a call that would have succeeded anyway,
    #     and it directly implements the "detect malformed output -> reload
    #     -> retry" recovery Stage 2 requires.
    # Stage 2's CLI entry point defaults to "reload_on_failure" (see
    # cli.py) rather than changing this default, precisely so this
    # decision is visible/overridable per-run instead of silently baked in.
    llm_reload_policy: ReloadPolicy = "persistent"
    llm_reload_every_n_calls: int = 5  # only consulted when llm_reload_policy == "reload_every_n_calls"

    # --- Embedding model (Stage 4: semantic deduplication) ---
    # Runs entirely locally via fastembed (ONNX runtime, already present in
    # this environment) -- no external embedding API, no per-question
    # network cost. bge-small-en-v1.5 is a small (~130MB) general-purpose
    # English sentence embedding model, confirmed working in this
    # environment; swap via config if a different local fastembed model is
    # preferred. Never touches llm_provider/openrouter_models -- embedding
    # generation is entirely separate from the LLM abstraction.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Bump when embedding_model is swapped for a materially different
    # model/weights. Part of the embedding cache key (see
    # database.get_question_embedding / save_question_embedding), so
    # bumping this forces recomputation instead of silently reusing vectors
    # from a different embedding space under the same model name.
    embedding_model_version: str = "v1"
    # Local on-disk cache for downloaded embedding model weights (fastembed's
    # own cache directory, separate from database_path). Kept inside the
    # repo's own data/ directory, not a system temp dir, so it survives
    # reboots/cleanup and doesn't get silently wiped between runs.
    embedding_cache_dir: Path = Path("data/embedding_cache")

    # --- Deduplication (Stage 4) ---
    # Candidate-generation threshold ONLY: a pair scoring at or above this
    # cosine similarity is worth asking the LLM about. It is never treated
    # as a confirmed duplicate by itself -- see
    # exampapersorter/deduplication/candidates.py.
    embedding_similarity_threshold: float = 0.85
    # A candidate pair whose question_type differs is only offered to the
    # LLM if similarity clears this stricter bar. question_type is a strong
    # signal (an MCQ and a long-answer question about the same fact are
    # usually NOT the same question -- see project notes) but not an
    # absolute filter, so a same-worded question that Stage 2 typed
    # differently isn't silently excluded from consideration.
    embedding_cross_type_similarity_threshold: float = 0.95
    # How many candidate pairs are batched into one semantic-equivalence LLM
    # call -- mirrors Stage 3's per-paper batching (see
    # topic_classification/classify.py). Bounded well under
    # SemanticEquivalenceResult's schema cap so a large candidate set is
    # chunked across multiple calls rather than one oversized one.
    dedup_pair_batch_size: int = 40
    # Deterministic floor applied on top of the LLM's own reported
    # confidence: a verdict="same_question" is only unioned into a canonical
    # group if confidence is at or above this. Defense in depth against a
    # single overconfident-but-wrong LLM judgment silently merging two
    # distinct questions -- see deduplication/reconciliation.py.
    dedup_min_merge_confidence: float = 0.6

    # --- Storage ---
    database_path: Path = Path("data/pipeline.db")
    output_directory: Path = Path("output")

    # --- Stage 1: adaptive TOC/index search ---
    initial_textbook_search_range: int = 25   # pages per search window
    textbook_search_step: int = 25            # pages to advance per iteration if no TOC found
    maximum_textbook_search_pages: int = 150  # hard cap so a run can't scan an entire huge book
    toc_detection_confidence_threshold: float = 0.6

    # --- Topic validation thresholds ---
    min_expected_topics: int = 3
    max_expected_topics: int = 500

    # --- Stage 2 (not yet implemented, reserved) ---
    question_papers_directory: Path = Path("question_papers")
    question_types_to_process: tuple[str, ...] = (
        "long_answer",
        "short_answer",
        "essay",
        "short_essay",
    )
    filename_metadata_convention: str | None = None

    # --- Logging ---
    log_level: str = "INFO"

    @property
    def model_identifier(self) -> str:
        """Stable label used as the DB cache key for LLM verdicts (and for
        display) -- NOT necessarily the literal model that answers any one
        call. Ollama has exactly one configured model, so its own name is
        already a stable per-call identity. OpenRouter's pool has automatic
        per-call model fallback (see llm_providers/openrouter.py), so the
        actual responding model varies call to call; keying the cache on
        that would mean a fallback response could never be found on a
        later lookup even when the pool hasn't changed. The pool's identity
        only changes when the pool itself changes (see
        openrouter_pool_version) -- that's what this key tracks. The
        specific model that actually answered a given call is recorded
        separately, see llm_client.get_last_call_metadata()."""
        if self.llm_provider == "ollama":
            return self.ollama_model
        return f"openrouter-pool-{self.openrouter_pool_version}"


DEFAULT_CONFIG = Config()
