"""Central logging configuration.

INFO by default: stage, current file/page-range, progress, retries,
completions. DEBUG additionally logs raw LLM prompts/responses (via
llm_client's logger.debug calls) -- opt in with --debug on the CLI, since
those payloads are large and not useful noise for a normal run.
"""
from __future__ import annotations

import logging


class _SecretRedactionFilter(logging.Filter):
    """Scrubs any registered secret (e.g. the OpenRouter API key) out of
    every log record before it's formatted. Defense in depth on top of
    llm_providers/openrouter.py never logging the raw key itself -- this
    catches the case where some exception message from a dependency
    happens to echo it back (e.g. a request library including headers in
    an error repr)."""

    def __init__(self) -> None:
        super().__init__()
        self._secrets: list[str] = []

    def register(self, secret: str | None) -> None:
        if secret and secret not in self._secrets:
            self._secrets.append(secret)

    def filter(self, record: logging.LogRecord) -> bool:
        if self._secrets:
            msg = record.getMessage()
            for secret in self._secrets:
                if secret in msg:
                    msg = msg.replace(secret, "<redacted>")
            record.msg = msg
            record.args = ()
        return True


_redaction_filter = _SecretRedactionFilter()


def redact_secret_from_logs(secret: str | None) -> None:
    """Registers `secret` (e.g. a user-supplied OpenRouter API key) so it's
    scrubbed from every subsequent log record, regardless of which logger
    or level emits it. Safe to call before setup_logging()."""
    _redaction_filter.register(secret)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    if _redaction_filter not in root.filters:
        root.addFilter(_redaction_filter)
    # Docling/torch/transformers are chatty at INFO; keep them at WARNING
    # unless we're specifically debugging the pipeline itself. httpx/urllib3
    # are additionally kept off DEBUG specifically so a lower-level HTTP
    # client never has the chance to log request headers (Authorization)
    # even when --debug is passed -- see redact_secret_from_logs() above
    # for the complementary content-based scrub.
    for noisy in ("docling", "transformers", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
