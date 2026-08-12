"""Provider-agnostic interface that decouples the pipeline from any one LLM
backend. call_structured() (llm_client.py) is the only caller of
ProviderClient.complete() -- it owns the retry/truncation/validation loop
that is the same regardless of backend; each provider only knows how to
turn one (system_prompt, user_prompt, schema) into one raw text completion,
or raise ProviderCallError describing why it couldn't.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ProviderCallError(Exception):
    """Raised by a provider's complete() for anything that isn't a
    schema/validation problem: network error, HTTP error, rate limit, empty
    response, etc. call_structured retries these the same as a validation
    failure, EXCEPT when retryable=False (e.g. an invalid API key) -- that
    can never succeed by retrying, so call_structured stops immediately
    instead of burning the rest of its bounded attempts.

    quota_exhausted=True marks an ACCOUNT-level cap (OpenRouter 402
    insufficient credits, or a 429 that names a daily/credit-based limit
    rather than a plain per-minute one -- see llm_providers/openrouter.py's
    classification) as opposed to a per-request/per-model problem. Cycling
    to a different pool model can never bypass an account-level cap, so a
    quota_exhausted error is always also retryable=False: the 10-model pool
    is for model/provider fallback, not quota multiplication. Callers above
    call_structured (see LLMCallFailed.quota_exhausted) use this to stop an
    entire multi-paper run rather than treating it like an ordinary
    per-paper failure."""

    def __init__(self, message: str, *, retryable: bool = True, quota_exhausted: bool = False):
        super().__init__(message)
        self.retryable = retryable
        self.quota_exhausted = quota_exhausted


@dataclass
class RawCompletion:
    """One successful raw completion, before JSON/schema validation.

    requested_model/actual_model/fallback_level/provider_name exist so
    call_structured can record which model really answered a call (see
    llm_client.get_last_call_metadata()) even though every call site above
    it just sees the validated Pydantic result. For a backend with no
    concept of model fallback (Ollama), actual_model == requested_model and
    fallback_level is always 0.
    """

    content: str
    requested_model: str
    actual_model: str
    fallback_level: int = 0
    provider_name: str | None = None


class ProviderClient(Protocol):
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict,
        schema_name: str,
        max_tokens: int,
        attempt: int,
    ) -> RawCompletion:
        """Perform exactly one raw completion request. Raises
        ProviderCallError on any failure below the JSON-validation layer
        (call_structured handles JSON/schema validation itself, uniformly
        across providers). `attempt` is call_structured's 1-based retry
        counter -- a provider that supports multiple models (OpenRouter)
        uses it to rotate which model is tried first, so a validation
        failure doesn't just retry the exact same model."""
        ...
