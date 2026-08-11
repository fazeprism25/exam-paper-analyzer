"""Structured-output wrapper around whichever LLM backend is configured
(see Config.llm_provider) -- OpenRouter by default, Ollama as an optional
local dev/testing backend. This is the ONLY module the rest of the
pipeline calls into for LLM access; it has no idea which backend answered
a given call. Every call:

  - asks the backend to constrain generation to a Pydantic model's JSON
    schema, so we're not parsing free-form text, and
  - validates the response against that same Pydantic model before
    returning it.

If the model returns something that doesn't validate (or the call fails
outright -- backend down/unreachable, rate-limited, network hiccup), we
retry a bounded number of times and then raise LLMCallFailed. Callers are
expected to catch that, log it, and mark the affected item as
failed/needs-review -- never fabricate a result to paper over a failure.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, replace
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from exampapersorter.config import Config
from exampapersorter.llm_providers import ollama as ollama_provider
from exampapersorter.llm_providers import openrouter as openrouter_provider
from exampapersorter.llm_providers.base import ProviderCallError, RawCompletion

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMCallFailed(Exception):
    """Raised when no valid structured response could be obtained."""

    def __init__(self, message: str, truncated: bool = False):
        super().__init__(message)
        # True only if EVERY attempt's failure looked like truncation (see
        # _looks_truncated) rather than some other kind of malformed output.
        # Callers (paper_extraction.py's per-section/page split fallback)
        # use this to decide whether "generate a smaller response" is even
        # a plausible fix, as opposed to e.g. a reload-worthy session issue.
        self.truncated = truncated


def unload_current_model(config: Config) -> bool:
    """Best-effort model unload for the Ollama backend's reload policy (see
    Config.llm_reload_policy). A no-op for any other provider -- OpenRouter
    has no local model to unload, so callers (e.g. pipeline.py's
    reload_per_paper wiring) can call this unconditionally regardless of
    which provider is configured."""
    if config.llm_provider != "ollama":
        return True
    return ollama_provider.unload_current_model(config)


@dataclass
class CallMetrics:
    """Observability counters for the reload-policy investigation -- see
    Config.llm_reload_policy. Accumulates across calls in this process
    until reset_call_metrics() is called; callers that want per-run numbers
    (e.g. Stage 2's pipeline) reset at the start of a run and read
    get_call_metrics() at the end."""

    total_calls: int = 0
    total_attempts: int = 0
    total_retries: int = 0
    total_reloads: int = 0
    total_failures: int = 0
    total_truncations: int = 0
    total_elapsed_seconds: float = 0.0


@dataclass
class CallAttemptMetadata:
    """Which model actually produced the most recently *successful*
    call_structured() result, for auditing/debugging (see project notes on
    recording requested/actual model + fallback level). Intentionally NOT
    threaded through call_structured's return value -- every call site
    treats that return value as just the validated Pydantic model, and
    changing that would ripple through the whole pipeline for a
    secondary/debugging concern. Callers that want to persist this
    alongside a cached verdict (see database.py's call_metadata_json
    columns) read it via get_last_call_metadata() right after the
    call_structured() call it corresponds to."""

    provider: str
    requested_model: str
    actual_model_used: str
    fallback_level: int
    attempt_count: int
    upstream_provider: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


_metrics = CallMetrics()
_calls_since_reload = 0
_last_call_metadata: CallAttemptMetadata | None = None


def get_call_metrics() -> CallMetrics:
    return replace(_metrics)


def reset_call_metrics() -> None:
    global _metrics, _calls_since_reload
    _metrics = CallMetrics()
    _calls_since_reload = 0


def get_last_call_metadata() -> CallAttemptMetadata | None:
    """Metadata for the most recent successful call_structured() call in
    this process (None before any call has succeeded). Single global like
    CallMetrics above -- the pipeline drives calls sequentially, never
    concurrently, so there's no ambiguity about which call it refers to at
    the point a caller reads it."""
    return _last_call_metadata


# Matches content that is ENTIRELY a single ```json ... ``` / ``` ... ```
# fence (fullmatch, not search) -- deliberately does not try to extract JSON
# embedded in surrounding prose, since that would be the kind of aggressive
# cleanup that could turn genuinely malformed output into something that
# looks valid.
_MARKDOWN_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)


def _strip_markdown_json_fence(content: str) -> str:
    """Some models wrap an otherwise-complete structured response in a
    Markdown code fence (```json ... ``` or ``` ... ```) despite being asked
    for raw JSON. Strip only that surrounding fence -- observed directly on
    a real Stage 4 run: a fully valid, complete JSON array came back fenced,
    got misclassified as truncated (see _looks_truncated: the fence's
    closing ``` means the content doesn't end in '}') and then failed
    Pydantic validation outright. If the content isn't a single complete
    fenced block (e.g. no closing fence because generation was genuinely cut
    off mid-stream), this returns it unchanged so truncation detection and
    validation still see the real, unmodified content."""
    stripped = content.strip()
    match = _MARKDOWN_JSON_FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _looks_truncated(content: str) -> bool:
    """Cheap, format-agnostic truncation signature: valid JSON from a
    schema-constrained generation always ends with the closing brace of the
    top-level object. If generation stopped mid-value (hit the max output
    token ceiling, or any other early cutoff) the content is very likely
    non-empty but does NOT end with '}'. This does not attempt to parse the
    JSON -- it only distinguishes "the model ran out of room" from "the
    model wrote something that parses but doesn't satisfy the schema",
    which call for different recoveries (see the max-tokens bump below vs.
    plain retry)."""
    stripped = content.strip()
    return bool(stripped) and not stripped.endswith("}")


def _call_once(
    config: Config, system_prompt: str, user_prompt: str, schema: dict, schema_name: str, max_tokens: int, attempt: int
) -> RawCompletion:
    if config.llm_provider == "ollama":
        return ollama_provider.complete(config, system_prompt, user_prompt, schema, max_tokens)
    if config.llm_provider == "openrouter":
        return openrouter_provider.complete(config, system_prompt, user_prompt, schema, schema_name, max_tokens, attempt)
    raise ValueError(f"Unknown llm_provider: {config.llm_provider!r}")


def call_structured(
    config: Config,
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    call_label: str | None = None,
) -> T:
    global _calls_since_reload, _last_call_metadata
    label = call_label or response_model.__name__

    schema = response_model.model_json_schema()
    schema_name = response_model.__name__
    last_error: Exception | None = None
    last_was_truncated = False
    all_failures_were_truncations = True
    t_start = time.time()
    _metrics.total_calls += 1

    if (
        config.llm_provider == "ollama"
        and config.llm_reload_policy == "reload_every_n_calls"
        and _calls_since_reload >= config.llm_reload_every_n_calls
    ):
        logger.info(
            "Reload policy reload_every_n_calls: unloading %s after %d calls since last reload",
            config.ollama_model, _calls_since_reload,
        )
        if unload_current_model(config):
            _metrics.total_reloads += 1
        _calls_since_reload = 0

    current_max_tokens = config.llm_num_predict

    for attempt in range(1, config.llm_retry_count + 1):
        _metrics.total_attempts += 1
        try:
            raw = _call_once(config, system_prompt, user_prompt, schema, schema_name, current_max_tokens, attempt)
        except ProviderCallError as exc:
            last_error = exc
            all_failures_were_truncations = False
            logger.warning(
                "%s call failed (%s, attempt %d/%d): %s",
                config.llm_provider, label, attempt, config.llm_retry_count, exc,
            )
            if not exc.retryable:
                break
            if attempt < config.llm_retry_count:
                _metrics.total_retries += 1
            continue

        content = raw.content
        logger.debug(
            "LLM raw response (%s, attempt %d, model=%s): %s", label, attempt, raw.actual_model, content
        )
        normalized_content = _strip_markdown_json_fence(content)

        try:
            result = response_model.model_validate_json(normalized_content)
        except ValidationError as exc:
            last_error = exc
            last_was_truncated = _looks_truncated(normalized_content)
            all_failures_were_truncations = all_failures_were_truncations and last_was_truncated
            logger.warning(
                "LLM returned schema-invalid JSON for %s (%s, attempt %d/%d, model=%s, truncated=%s): %s",
                response_model.__name__, label, attempt, config.llm_retry_count, raw.actual_model, last_was_truncated, exc,
            )
            if attempt < config.llm_retry_count:
                _metrics.total_retries += 1
                if last_was_truncated:
                    _metrics.total_truncations += 1
                    bumped = min(int(current_max_tokens * config.llm_truncation_retry_multiplier), config.llm_num_predict_max)
                    if bumped > current_max_tokens:
                        logger.info(
                            "Truncation detected for %s (attempt %d/%d): retrying with max output tokens %d -> %d",
                            label, attempt, config.llm_retry_count, current_max_tokens, bumped,
                        )
                        current_max_tokens = bumped
                # Reload policy is orthogonal to the truncation-specific
                # max-tokens bump above: reload_on_failure still fires on
                # ANY validation failure per its own contract, even though
                # the Stage 2 real-run finding was that reload does NOT fix
                # a genuine max-output-tokens ceiling (paper5: two
                # post-reload attempts produced byte-identical truncated
                # output). It's kept here anyway because a caller may have
                # chosen reload_on_failure for other calls in the same run
                # where it DOES help (session drift), and this call has no
                # way to know in advance which cause it's facing. Ollama
                # only -- OpenRouter has no local model to reload.
                if config.llm_provider == "ollama" and config.llm_reload_policy == "reload_on_failure":
                    logger.info(
                        "Reload policy reload_on_failure: unloading %s after malformed output (%s, attempt %d/%d)",
                        config.ollama_model, label, attempt, config.llm_retry_count,
                    )
                    if unload_current_model(config):
                        _metrics.total_reloads += 1
                        _calls_since_reload = 0
            continue

        _calls_since_reload += 1
        _metrics.total_elapsed_seconds += time.time() - t_start
        _last_call_metadata = CallAttemptMetadata(
            provider=config.llm_provider,
            requested_model=raw.requested_model,
            actual_model_used=raw.actual_model,
            fallback_level=raw.fallback_level,
            attempt_count=attempt,
            upstream_provider=raw.provider_name,
        )
        if raw.fallback_level > 0 or raw.requested_model != raw.actual_model:
            logger.info(
                "%s (%s): served by %s (requested %s, fallback_level=%d, attempt=%d)",
                label, config.llm_provider, raw.actual_model, raw.requested_model, raw.fallback_level, attempt,
            )
        return result

    _metrics.total_failures += 1
    _metrics.total_elapsed_seconds += time.time() - t_start
    raise LLMCallFailed(
        f"Failed to get valid {response_model.__name__} via {config.llm_provider} "
        f"(model_identifier={config.model_identifier}) after {config.llm_retry_count} attempts: {last_error}",
        truncated=all_failures_were_truncations and last_was_truncated,
    )
