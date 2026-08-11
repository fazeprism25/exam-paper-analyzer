"""Ollama backend -- optional local dev/testing path (see
Config.llm_provider). Not required for normal use; OpenRouter is the
default (see openrouter.py). Behavior here is unchanged from the
pre-migration implementation: one model, `format=<json schema>` for
constrained decoding, and the unload/reload policy investigated during
Stage 1 (see Config.llm_reload_policy).
"""
from __future__ import annotations

import logging
import subprocess

import ollama as ollama_sdk

from exampapersorter.config import Config
from exampapersorter.llm_providers.base import ProviderCallError, RawCompletion

logger = logging.getLogger(__name__)


def unload_current_model(config: Config) -> bool:
    """Best-effort `ollama stop <model>`. Returns True if the command ran
    without raising (not whether Ollama actually had it loaded -- stopping
    an already-unloaded model is harmless). Never raises: a failed unload
    attempt should not itself take down a pipeline run, it should just mean
    the next call proceeds without the (unconfirmed) benefit of a reload."""
    try:
        subprocess.run(["ollama", "stop", config.ollama_model], capture_output=True, timeout=30)
        return True
    except Exception as exc:
        logger.warning("Failed to unload model %s: %s", config.ollama_model, exc)
        return False


def complete(
    config: Config,
    system_prompt: str,
    user_prompt: str,
    schema: dict,
    max_tokens: int,
) -> RawCompletion:
    client = ollama_sdk.Client(host=config.ollama_base_url)
    options = {"temperature": 0, "num_ctx": config.llm_num_ctx, "num_predict": max_tokens}
    if config.llm_seed is not None:
        options["seed"] = config.llm_seed

    try:
        response = client.chat(
            model=config.ollama_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format=schema,
            think=config.llm_think,
            options=options,
        )
    except Exception as exc:
        raise ProviderCallError(str(exc)) from exc

    content = response.message.content or ""
    return RawCompletion(
        content=content, requested_model=config.ollama_model, actual_model=config.ollama_model, fallback_level=0
    )
