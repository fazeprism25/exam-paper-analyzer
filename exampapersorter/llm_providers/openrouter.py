"""OpenRouter backend -- the default/normal backend (see Config.llm_provider).

Two levels of fallback, per the project's architecture:

  Level 1 (provider fallback): within a single HTTP request, OpenRouter
  itself fails over across upstream compute providers for whichever model
  is chosen -- we don't implement this, it's inherent to calling
  OpenRouter rather than a single model provider directly.

  Level 2 (model fallback): we send an ORDERED list of Config.openrouter_models
  as the request's `models` field, OpenRouter's own model-fallback
  mechanism (tried in order if an earlier entry errors/rate-limits within
  that one request). On top of that, `complete()` rotates which model is
  *first* in the list based on call_structured's attempt number -- so a
  request that came back with schema-invalid JSON (a failure OpenRouter
  itself can't see, since HTTP-wise that request succeeded) gets retried
  against a different primary model rather than the same one again.

This module never implements ten independent provider SDKs -- OpenRouter's
own HTTP API is the only thing spoken here.
"""
from __future__ import annotations

import logging

import requests

from exampapersorter.config import Config
from exampapersorter.llm_providers.base import ProviderCallError, RawCompletion

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_PATH = "/chat/completions"
AUTH_KEY_PATH = "/auth/key"

# OpenRouter rejects a `models` fallback array with more than this many
# entries (HTTP 400 "'models' array must have N items or fewer" --
# confirmed against the live API 2026-08-11). This is why call_structured's
# OWN attempt-based rotation matters: OpenRouter's per-request fallback
# only ever covers a short window of the pool, so cycling the window's
# start point across retries is what actually gets through all
# ~10 configured models over a call's bounded retry count, not one request.
MAX_MODELS_PER_REQUEST = 3


def _redact(key: str | None) -> str:
    """Never put a usable key fragment in logs -- last 4 chars is enough to
    tell two configured keys apart without being replayable."""
    if not key:
        return "<missing>"
    return f"...{key[-4:]}" if len(key) > 4 else "***"


def _headers(config: Config) -> dict[str, str]:
    if not config.openrouter_api_key:
        raise ProviderCallError("No OpenRouter API key configured", retryable=False)
    headers = {
        "Authorization": f"Bearer {config.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    if config.openrouter_http_referer:
        headers["HTTP-Referer"] = config.openrouter_http_referer
    if config.openrouter_app_title:
        headers["X-Title"] = config.openrouter_app_title
    return headers


def _rotate_models(models: tuple[str, ...], attempt: int) -> list[str]:
    """Rotates the pool so attempt N starts from a different model than
    attempt N-1, wrapping around. attempt is 1-based (call_structured's
    convention), so attempt 1 leaves the pool in its configured order."""
    if not models:
        raise ProviderCallError("Config.openrouter_models is empty -- no model pool configured", retryable=False)
    start = (attempt - 1) % len(models)
    return list(models[start:]) + list(models[:start])


# Substrings that reliably distinguish an ACCOUNT-level quota exhaustion
# from a plain per-minute rate limit, both of which arrive as HTTP 429. Per
# OpenRouter's own docs (openrouter.ai/docs/api_reference/limits) and
# observed real responses, a per-minute limit's message is just "Rate limit
# exceeded" with no day/credit language, while the free-tier daily cap names
# it explicitly, e.g. "Rate limit exceeded: free-models-per-day. Add 10
# credits to unlock 1000 free model requests per day." Matching on the
# message (rather than trying to parse X-RateLimit-Reset and guess whether
# the wait is "long") keeps this deterministic and testable.
_QUOTA_EXHAUSTED_429_MARKERS = ("per-day", "per day", "daily", "add credits", "add 10 credits", "credits to unlock")


def _is_quota_exhausted_429(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _QUOTA_EXHAUSTED_429_MARKERS)


def _error_message(response: requests.Response) -> str:
    try:
        body = response.json()
        msg = body.get("error", {}).get("message")
        if msg:
            return str(msg)
    except (ValueError, AttributeError):
        pass
    return response.text[:500]


def complete(
    config: Config,
    system_prompt: str,
    user_prompt: str,
    schema: dict,
    schema_name: str,
    max_tokens: int,
    attempt: int,
) -> RawCompletion:
    rotated = _rotate_models(config.openrouter_models, attempt)
    primary = rotated[0]
    request_models = rotated[:MAX_MODELS_PER_REQUEST]

    # Structured output is requested via response_format (best-effort --
    # not every free model honors strict JSON-schema constrained decoding),
    # AND the schema is restated in the prompt as a second line of defense.
    # Either way, Pydantic validation back in call_structured is the real
    # gate: a model that ignores both still just gets treated as a
    # malformed response and retried/rotated, never accepted unvalidated.
    schema_reminder = (
        f"\n\nRespond with ONLY a single JSON object matching this JSON Schema "
        f"(no markdown fences, no commentary):\n{schema}"
    )

    payload: dict = {
        "model": primary,
        "models": request_models,
        "messages": [
            {"role": "system", "content": system_prompt + schema_reminder},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
    }
    if config.llm_seed is not None:
        payload["seed"] = config.llm_seed

    url = config.openrouter_base_url.rstrip("/") + CHAT_COMPLETIONS_PATH
    try:
        response = requests.post(
            url, headers=_headers(config), json=payload, timeout=config.openrouter_timeout_seconds
        )
    except requests.RequestException as exc:
        raise ProviderCallError(f"OpenRouter request failed: {exc}") from exc

    if response.status_code == 401:
        raise ProviderCallError(
            f"OpenRouter rejected the API key ({_redact(config.openrouter_api_key)}): {_error_message(response)}",
            retryable=False,
        )
    if response.status_code == 402:
        # Always account-level (insufficient credits/limit exhausted for
        # this key) -- never a single-model problem, so never worth cycling
        # the pool for. See ProviderCallError.quota_exhausted.
        raise ProviderCallError(
            f"OpenRouter: insufficient credits/quota for this key: {_error_message(response)}. "
            "Free-tier capacity may be exhausted -- try again later.",
            retryable=False, quota_exhausted=True,
        )
    if response.status_code == 429:
        message = _error_message(response)
        if _is_quota_exhausted_429(message):
            raise ProviderCallError(
                f"OpenRouter: daily/account request quota exhausted: {message}",
                retryable=False, quota_exhausted=True,
            )
        raise ProviderCallError(f"OpenRouter rate-limited this request: {message}")
    if response.status_code >= 500:
        raise ProviderCallError(f"OpenRouter upstream error ({response.status_code}): {_error_message(response)}")
    if response.status_code != 200:
        raise ProviderCallError(f"OpenRouter returned HTTP {response.status_code}: {_error_message(response)}")

    try:
        body = response.json()
    except ValueError as exc:
        raise ProviderCallError(f"OpenRouter returned non-JSON response body: {exc}") from exc

    choices = body.get("choices") or []
    if not choices:
        raise ProviderCallError(f"OpenRouter response had no choices: {body}")

    content = (choices[0].get("message") or {}).get("content")
    if not content:
        raise ProviderCallError("OpenRouter response had an empty message content")

    actual_model = body.get("model") or primary
    fallback_level = rotated.index(actual_model) if actual_model in rotated else 0
    provider_name = body.get("provider")

    return RawCompletion(
        content=content,
        requested_model=primary,
        actual_model=actual_model,
        fallback_level=fallback_level,
        provider_name=provider_name,
    )


def validate_api_key(config: Config) -> tuple[bool, str]:
    """Lightweight, cheap check that the configured key is accepted by
    OpenRouter -- does NOT spend model-inference quota (hits /auth/key, not
    /chat/completions). Returns (ok, message); message is a short
    human-readable reason on failure, or a short confirmation on success."""
    if not config.openrouter_api_key:
        return False, "No OpenRouter API key configured"

    url = config.openrouter_base_url.rstrip("/") + AUTH_KEY_PATH
    try:
        response = requests.get(url, headers=_headers(config), timeout=config.openrouter_timeout_seconds)
    except requests.RequestException as exc:
        return False, f"Could not reach OpenRouter: {exc}"

    if response.status_code == 200:
        return True, "API key OK"
    if response.status_code == 401:
        return False, f"OpenRouter rejected the API key ({_redact(config.openrouter_api_key)}): {_error_message(response)}"
    return False, f"OpenRouter key check returned HTTP {response.status_code}: {_error_message(response)}"
