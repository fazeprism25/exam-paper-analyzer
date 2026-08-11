import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from exampapersorter import llm_client
from exampapersorter.config import DEFAULT_CONFIG
from exampapersorter.llm_client import LLMCallFailed, call_structured, get_last_call_metadata
from exampapersorter.llm_providers import ollama as ollama_provider
from exampapersorter.llm_providers import openrouter as openrouter_provider


class Widget(BaseModel):
    name: str
    count: int


OLLAMA_CONFIG = replace(DEFAULT_CONFIG, llm_provider="ollama")


# =====================================================================
# Ollama backend (optional local dev/testing path -- see Config.llm_provider)
# =====================================================================


def fake_response(content: str):
    return SimpleNamespace(message=SimpleNamespace(content=content))


class FakeOllamaClient:
    """Returns queued responses/exceptions in order, one per .chat() call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.last_kwargs = None

    def chat(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return fake_response(item)


def install_fake_client(monkeypatch, responses):
    fake = FakeOllamaClient(responses)
    monkeypatch.setattr(ollama_provider.ollama_sdk, "Client", lambda host: fake)
    return fake


def test_valid_response_on_first_try(monkeypatch):
    install_fake_client(monkeypatch, ['{"name": "widget", "count": 3}'])
    result = call_structured(OLLAMA_CONFIG, "sys", "user", Widget)
    assert result == Widget(name="widget", count=3)


def test_invalid_json_then_valid_json_retries_and_succeeds(monkeypatch):
    install_fake_client(
        monkeypatch,
        ["not valid json at all", '{"name": "widget", "count": 3}'],
    )
    config = replace(OLLAMA_CONFIG, llm_retry_count=3)
    result = call_structured(config, "sys", "user", Widget)
    assert result == Widget(name="widget", count=3)


def test_schema_invalid_json_counts_as_failure_and_retries(monkeypatch):
    # valid JSON but missing required field "count" -> Pydantic validation error
    install_fake_client(
        monkeypatch,
        ['{"name": "widget"}', '{"name": "widget", "count": 5}'],
    )
    config = replace(OLLAMA_CONFIG, llm_retry_count=3)
    result = call_structured(config, "sys", "user", Widget)
    assert result == Widget(name="widget", count=5)


def test_exhausting_retries_raises_llm_call_failed_never_fabricates(monkeypatch):
    config = replace(OLLAMA_CONFIG, llm_retry_count=2)
    install_fake_client(monkeypatch, ["garbage", "still garbage"])
    with pytest.raises(LLMCallFailed):
        call_structured(config, "sys", "user", Widget)


def test_connection_error_is_retried_then_raises(monkeypatch):
    config = replace(OLLAMA_CONFIG, llm_retry_count=2)
    install_fake_client(monkeypatch, [ConnectionError("ollama unreachable"), ConnectionError("still down")])
    with pytest.raises(LLMCallFailed):
        call_structured(config, "sys", "user", Widget)


def test_respects_configured_retry_count(monkeypatch):
    config = replace(OLLAMA_CONFIG, llm_retry_count=4)
    fake = install_fake_client(monkeypatch, ["bad"] * 4)
    with pytest.raises(LLMCallFailed):
        call_structured(config, "sys", "user", Widget)
    assert fake.calls == 4


def test_seed_omitted_from_options_by_default(monkeypatch):
    fake = install_fake_client(monkeypatch, ['{"name": "widget", "count": 3}'])
    call_structured(OLLAMA_CONFIG, "sys", "user", Widget)
    assert "seed" not in fake.last_kwargs["options"]


def test_seed_passed_through_when_configured(monkeypatch):
    fake = install_fake_client(monkeypatch, ['{"name": "widget", "count": 3}'])
    config = replace(OLLAMA_CONFIG, llm_seed=42)
    call_structured(config, "sys", "user", Widget)
    assert fake.last_kwargs["options"]["seed"] == 42


# --- reload policy (Ollama only) ---


def install_fake_unload(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_client, "unload_current_model", lambda config: calls.append(1) or True)
    return calls


def test_persistent_policy_never_reloads_even_on_failure(monkeypatch):
    """Default policy: Stage 1's exact tested behavior, no reload side effects."""
    unload_calls = install_fake_unload(monkeypatch)
    install_fake_client(monkeypatch, ["garbage", '{"name": "widget", "count": 3}'])
    config = replace(OLLAMA_CONFIG, llm_retry_count=3, llm_reload_policy="persistent")
    call_structured(config, "sys", "user", Widget)
    assert unload_calls == []


def test_reload_on_failure_unloads_before_the_retry_that_succeeds(monkeypatch):
    unload_calls = install_fake_unload(monkeypatch)
    install_fake_client(monkeypatch, ["garbage", '{"name": "widget", "count": 3}'])
    config = replace(OLLAMA_CONFIG, llm_retry_count=3, llm_reload_policy="reload_on_failure")
    result = call_structured(config, "sys", "user", Widget)
    assert result == Widget(name="widget", count=3)
    assert unload_calls == [1]


def test_reload_on_failure_does_not_reload_on_a_clean_first_attempt(monkeypatch):
    unload_calls = install_fake_unload(monkeypatch)
    install_fake_client(monkeypatch, ['{"name": "widget", "count": 3}'])
    config = replace(OLLAMA_CONFIG, llm_reload_policy="reload_on_failure")
    call_structured(config, "sys", "user", Widget)
    assert unload_calls == []


def test_reload_every_n_calls_reloads_after_threshold(monkeypatch):
    llm_client.reset_call_metrics()
    unload_calls = install_fake_unload(monkeypatch)
    responses = ['{"name": "widget", "count": 3}'] * 5
    install_fake_client(monkeypatch, responses)
    config = replace(OLLAMA_CONFIG, llm_reload_policy="reload_every_n_calls", llm_reload_every_n_calls=2)

    for _ in range(5):
        call_structured(config, "sys", "user", Widget)

    # calls 1,2 -> no reload yet; reload triggers at the START of call 3 (after 2
    # successful calls since the last reload), then again at call 5.
    assert unload_calls == [1, 1]
    llm_client.reset_call_metrics()


def test_reload_policy_is_a_no_op_for_openrouter(monkeypatch):
    """reload_per_paper etc. only make sense for a locally-loaded model --
    unload_current_model must no-op (not crash, not try to shell out to
    `ollama stop`) when the configured provider isn't ollama."""
    calls = []
    monkeypatch.setattr(ollama_provider, "unload_current_model", lambda config: calls.append(1) or True)
    config = replace(DEFAULT_CONFIG, llm_provider="openrouter")
    assert llm_client.unload_current_model(config) is True
    assert calls == []


def test_call_metrics_track_calls_retries_and_failures(monkeypatch):
    llm_client.reset_call_metrics()
    install_fake_client(monkeypatch, ["garbage", '{"name": "widget", "count": 3}'])
    config = replace(OLLAMA_CONFIG, llm_retry_count=3)
    call_structured(config, "sys", "user", Widget)

    metrics = llm_client.get_call_metrics()
    assert metrics.total_calls == 1
    assert metrics.total_attempts == 2
    assert metrics.total_retries == 1
    assert metrics.total_failures == 0
    llm_client.reset_call_metrics()


def test_call_metrics_count_a_fully_exhausted_call_as_a_failure(monkeypatch):
    llm_client.reset_call_metrics()
    config = replace(OLLAMA_CONFIG, llm_retry_count=2)
    install_fake_client(monkeypatch, ["garbage", "still garbage"])
    with pytest.raises(LLMCallFailed):
        call_structured(config, "sys", "user", Widget)

    metrics = llm_client.get_call_metrics()
    assert metrics.total_calls == 1
    assert metrics.total_failures == 1
    llm_client.reset_call_metrics()


def test_ollama_call_metadata_has_no_fallback(monkeypatch):
    install_fake_client(monkeypatch, ['{"name": "widget", "count": 3}'])
    call_structured(OLLAMA_CONFIG, "sys", "user", Widget)
    meta = get_last_call_metadata()
    assert meta.provider == "ollama"
    assert meta.requested_model == meta.actual_model_used == OLLAMA_CONFIG.ollama_model
    assert meta.fallback_level == 0
    assert meta.attempt_count == 1


# =====================================================================
# OpenRouter backend (default -- see Config.llm_provider)
# =====================================================================

OR_CONFIG = replace(
    DEFAULT_CONFIG,
    llm_provider="openrouter",
    openrouter_api_key="sk-or-v1-testkey0000000000000000000000000000000000000000",
    openrouter_models=("model-a", "model-b", "model-c"),
)


class FakeResponse:
    def __init__(self, status_code, json_data=None, text=None):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text if text is not None else (json.dumps(json_data) if json_data is not None else "")

    def json(self):
        if self._json_data is None:
            raise ValueError("no JSON body")
        return self._json_data


def openrouter_success_body(content: str, model: str = "model-a", provider: str | None = None) -> dict:
    body = {"model": model, "choices": [{"message": {"content": content}}]}
    if provider:
        body["provider"] = provider
    return body


def install_fake_post(monkeypatch, responses):
    """Queues FakeResponse (or Exception) objects, one per requests.post
    call, and records every call's kwargs for assertions."""
    calls = []
    queue = list(responses)

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "payload": json, "timeout": timeout})
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(openrouter_provider.requests, "post", fake_post)
    return calls


def test_openrouter_successful_call_returns_validated_model(monkeypatch):
    install_fake_post(monkeypatch, [FakeResponse(200, openrouter_success_body('{"name": "widget", "count": 3}'))])
    result = call_structured(OR_CONFIG, "sys", "user", Widget)
    assert result == Widget(name="widget", count=3)


def test_openrouter_request_shape_includes_ordered_pool_and_schema(monkeypatch):
    calls = install_fake_post(
        monkeypatch, [FakeResponse(200, openrouter_success_body('{"name": "widget", "count": 3}'))]
    )
    call_structured(OR_CONFIG, "sys", "user", Widget)

    payload = calls[0]["payload"]
    assert payload["model"] == "model-a"
    assert payload["models"] == ["model-a", "model-b", "model-c"]
    assert payload["temperature"] == 0
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["name"] == "Widget"
    assert "count" in payload["response_format"]["json_schema"]["schema"]["properties"]
    assert calls[0]["headers"]["Authorization"] == f"Bearer {OR_CONFIG.openrouter_api_key}"


def test_openrouter_caps_models_array_at_three_entries(monkeypatch):
    """OpenRouter's live API rejects a `models` fallback array with more
    than 3 entries (HTTP 400 "'models' array must have 3 items or fewer" --
    confirmed against the real API). The full ~10-model pool must still
    only ever be spread across call_structured's own attempt-based
    rotation, never sent in one oversized request."""
    calls = install_fake_post(
        monkeypatch, [FakeResponse(200, openrouter_success_body('{"name": "widget", "count": 3}'))]
    )
    big_pool_config = replace(DEFAULT_CONFIG, llm_provider="openrouter", openrouter_api_key="k", openrouter_models=tuple(f"model-{i}" for i in range(10)))
    call_structured(big_pool_config, "sys", "user", Widget)
    assert len(calls[0]["payload"]["models"]) <= 3
    assert calls[0]["payload"]["models"] == ["model-0", "model-1", "model-2"]


def test_openrouter_rotates_primary_model_on_schema_validation_retry(monkeypatch):
    calls = install_fake_post(
        monkeypatch,
        [
            FakeResponse(200, openrouter_success_body("not valid json")),
            FakeResponse(200, openrouter_success_body('{"name": "widget", "count": 3}', model="model-b")),
        ],
    )
    config = replace(OR_CONFIG, llm_retry_count=3)
    result = call_structured(config, "sys", "user", Widget)
    assert result == Widget(name="widget", count=3)

    assert calls[0]["payload"]["model"] == "model-a"
    assert calls[1]["payload"]["model"] == "model-b"  # rotated after the first attempt's bad JSON
    assert calls[1]["payload"]["models"] == ["model-b", "model-c", "model-a"]


def test_openrouter_401_is_not_retried(monkeypatch):
    calls = install_fake_post(monkeypatch, [FakeResponse(401, {"error": {"message": "User not found.", "code": 401}})])
    config = replace(OR_CONFIG, llm_retry_count=3)
    with pytest.raises(LLMCallFailed):
        call_structured(config, "sys", "user", Widget)
    assert len(calls) == 1  # never burned the rest of llm_retry_count on a key that can't ever succeed


def test_openrouter_key_never_appears_in_failure_message(monkeypatch):
    install_fake_post(monkeypatch, [FakeResponse(401, {"error": {"message": "User not found.", "code": 401}})])
    config = replace(OR_CONFIG, llm_retry_count=1)
    with pytest.raises(LLMCallFailed) as exc_info:
        call_structured(config, "sys", "user", Widget)
    assert OR_CONFIG.openrouter_api_key not in str(exc_info.value)


def test_openrouter_missing_api_key_fails_without_any_request(monkeypatch):
    calls = install_fake_post(monkeypatch, [])
    config = replace(OR_CONFIG, openrouter_api_key=None, llm_retry_count=3)
    with pytest.raises(LLMCallFailed):
        call_structured(config, "sys", "user", Widget)
    assert calls == []


def test_openrouter_429_is_retried_against_next_attempt(monkeypatch):
    calls = install_fake_post(
        monkeypatch,
        [
            FakeResponse(429, {"error": {"message": "rate limited", "code": 429}}),
            FakeResponse(200, openrouter_success_body('{"name": "widget", "count": 3}')),
        ],
    )
    config = replace(OR_CONFIG, llm_retry_count=3)
    result = call_structured(config, "sys", "user", Widget)
    assert result == Widget(name="widget", count=3)
    assert len(calls) == 2


def test_openrouter_network_error_is_retried(monkeypatch):
    import requests as requests_module

    install_fake_post(
        monkeypatch,
        [
            requests_module.ConnectionError("dns failure"),
            FakeResponse(200, openrouter_success_body('{"name": "widget", "count": 3}')),
        ],
    )
    config = replace(OR_CONFIG, llm_retry_count=3)
    result = call_structured(config, "sys", "user", Widget)
    assert result == Widget(name="widget", count=3)


def test_openrouter_empty_content_is_treated_as_a_retryable_failure(monkeypatch):
    calls = install_fake_post(
        monkeypatch,
        [
            FakeResponse(200, {"model": "model-a", "choices": [{"message": {"content": ""}}]}),
            FakeResponse(200, openrouter_success_body('{"name": "widget", "count": 3}')),
        ],
    )
    config = replace(OR_CONFIG, llm_retry_count=3)
    result = call_structured(config, "sys", "user", Widget)
    assert result == Widget(name="widget", count=3)
    assert len(calls) == 2


def test_openrouter_records_actual_model_and_fallback_level_metadata(monkeypatch):
    install_fake_post(
        monkeypatch, [FakeResponse(200, openrouter_success_body('{"name": "widget", "count": 3}', model="model-b"))]
    )
    call_structured(OR_CONFIG, "sys", "user", Widget)
    meta = get_last_call_metadata()
    assert meta.provider == "openrouter"
    assert meta.requested_model == "model-a"
    assert meta.actual_model_used == "model-b"
    assert meta.fallback_level == 1  # model-b is index 1 in the rotated pool [a, b, c]
    assert meta.attempt_count == 1


def test_openrouter_upstream_provider_recorded_when_present(monkeypatch):
    install_fake_post(
        monkeypatch,
        [FakeResponse(200, openrouter_success_body('{"name": "widget", "count": 3}', provider="SomeCompute"))],
    )
    call_structured(OR_CONFIG, "sys", "user", Widget)
    meta = get_last_call_metadata()
    assert meta.upstream_provider == "SomeCompute"


# --- validate_api_key ---


def install_fake_get(monkeypatch, response):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append({"url": url, "headers": headers})
        return response

    monkeypatch.setattr(openrouter_provider.requests, "get", fake_get)
    return calls


def test_validate_api_key_success(monkeypatch):
    install_fake_get(monkeypatch, FakeResponse(200, {"data": {"label": "test key"}}))
    ok, message = openrouter_provider.validate_api_key(OR_CONFIG)
    assert ok is True


def test_validate_api_key_rejects_invalid_key(monkeypatch):
    install_fake_get(monkeypatch, FakeResponse(401, {"error": {"message": "User not found.", "code": 401}}))
    ok, message = openrouter_provider.validate_api_key(OR_CONFIG)
    assert ok is False
    assert "401" in message or "rejected" in message.lower()
    assert OR_CONFIG.openrouter_api_key not in message


def test_validate_api_key_missing_key_fails_without_request(monkeypatch):
    calls = install_fake_get(monkeypatch, FakeResponse(200, {}))
    config = replace(OR_CONFIG, openrouter_api_key=None)
    ok, message = openrouter_provider.validate_api_key(config)
    assert ok is False
    assert calls == []


# =====================================================================
# Markdown JSON code fence normalization (see _strip_markdown_json_fence)
#
# Some models wrap an otherwise-complete structured response in a
# ```json ... ``` / ``` ... ``` fence despite being asked for raw JSON.
# Observed directly on a real Stage 4 run: a complete, valid 36-verdict
# JSON array came back fenced, was misclassified as truncated (its last
# character is the closing fence's "`", not "}"), and then failed Pydantic
# validation outright -- discarding 36 real LLM judgments.
# =====================================================================


def test_strip_markdown_json_fence_leaves_unfenced_json_unchanged():
    raw = '{"name": "widget", "count": 3}'
    assert llm_client._strip_markdown_json_fence(raw) == raw


def test_strip_markdown_json_fence_removes_json_tagged_fence():
    fenced = '```json\n{"name": "widget", "count": 3}\n```'
    assert llm_client._strip_markdown_json_fence(fenced) == '{"name": "widget", "count": 3}'


def test_strip_markdown_json_fence_removes_untagged_fence():
    fenced = '```\n{"name": "widget", "count": 3}\n```'
    assert llm_client._strip_markdown_json_fence(fenced) == '{"name": "widget", "count": 3}'


def test_strip_markdown_json_fence_tolerates_surrounding_whitespace():
    fenced = '  \n\n```json\n\n{"name": "widget", "count": 3}\n\n```\n  \n'
    assert llm_client._strip_markdown_json_fence(fenced) == '{"name": "widget", "count": 3}'


def test_strip_markdown_json_fence_preserves_content_exactly():
    # The captured JSON body itself must come back byte-identical -- this is
    # fence removal only, never a JSON-reformatting/cleanup step.
    body = '{"name":  "widget",\n  "count": 3}'
    fenced = f"```json\n{body}\n```"
    assert llm_client._strip_markdown_json_fence(fenced) == body


def test_strip_markdown_json_fence_leaves_unclosed_fence_unchanged():
    # No closing ``` -- generation was genuinely cut off mid-stream. Must
    # NOT strip the opening fence marker, so truncation detection still
    # sees the real (still-truncated-looking) content.
    unclosed = '```json\n{"name": "widget", "count": 3'
    assert llm_client._strip_markdown_json_fence(unclosed) == unclosed


def test_looks_truncated_treats_fenced_complete_json_as_truncated_before_stripping():
    # Documents WHY normalization must run before _looks_truncated: the raw
    # fenced content ends in "```", not "}", so checking it directly (as
    # the old code path did) misclassifies complete output as truncated.
    fenced = '```json\n{"name": "widget", "count": 3}\n```'
    assert llm_client._looks_truncated(fenced) is True
    normalized = llm_client._strip_markdown_json_fence(fenced)
    assert llm_client._looks_truncated(normalized) is False


def test_looks_truncated_still_flags_genuinely_incomplete_json():
    incomplete = '{"name": "widget", "count": '
    assert llm_client._looks_truncated(llm_client._strip_markdown_json_fence(incomplete)) is True


def test_fenced_valid_json_is_accepted_via_call_structured(monkeypatch):
    """End-to-end: a fenced-but-complete response now succeeds on the first
    attempt instead of being discarded as truncated/invalid."""
    fenced = '```json\n{"name": "widget", "count": 3}\n```'
    install_fake_client(monkeypatch, [fenced])
    result = call_structured(OLLAMA_CONFIG, "sys", "user", Widget)
    assert result == Widget(name="widget", count=3)


def test_fenced_valid_json_without_json_tag_is_accepted_via_call_structured(monkeypatch):
    fenced = '```\n{"name": "widget", "count": 3}\n```'
    install_fake_client(monkeypatch, [fenced])
    result = call_structured(OLLAMA_CONFIG, "sys", "user", Widget)
    assert result == Widget(name="widget", count=3)


def test_fenced_json_with_extra_whitespace_is_accepted_via_call_structured(monkeypatch):
    fenced = '\n\n```json\n\n{"name": "widget", "count": 3}\n\n```\n\n'
    install_fake_client(monkeypatch, [fenced])
    result = call_structured(OLLAMA_CONFIG, "sys", "user", Widget)
    assert result == Widget(name="widget", count=3)


def test_genuinely_truncated_fenced_json_still_retries_as_truncated(monkeypatch):
    """An unclosed fence around incomplete JSON must still be recognized as
    truncated (bumping max tokens for the retry), not silently swallowed."""
    truncated = '```json\n{"name": "widget", "count":'
    install_fake_client(monkeypatch, [truncated, '{"name": "widget", "count": 3}'])
    config = replace(OLLAMA_CONFIG, llm_retry_count=3)
    result = call_structured(config, "sys", "user", Widget)
    assert result == Widget(name="widget", count=3)

    metrics = llm_client.get_call_metrics()
    llm_client.reset_call_metrics()
    assert metrics.total_truncations >= 1


def test_fenced_but_schema_invalid_json_remains_a_failure(monkeypatch):
    """Fence-stripping must not paper over genuinely malformed/invalid
    content -- missing the required "count" field is still a validation
    error after the fence is removed."""
    fenced = '```json\n{"name": "widget"}\n```'
    config = replace(OLLAMA_CONFIG, llm_retry_count=1)
    install_fake_client(monkeypatch, [fenced])
    with pytest.raises(LLMCallFailed):
        call_structured(config, "sys", "user", Widget)


def test_fenced_garbage_never_becomes_valid_through_stripping(monkeypatch):
    """Stripping the fence off non-JSON content must still leave it
    unparsable -- no aggressive cleanup that could turn malformed output
    into something that looks valid."""
    fenced = "```json\nthis is not json at all\n```"
    config = replace(OLLAMA_CONFIG, llm_retry_count=1)
    install_fake_client(monkeypatch, [fenced])
    with pytest.raises(LLMCallFailed):
        call_structured(config, "sys", "user", Widget)
