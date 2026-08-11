"""Optional local persistence for the user's OpenRouter API key, via the OS
credential store (Windows Credential Manager / macOS Keychain / Linux
Secret Service) through the `keyring` package -- never a plain-text file in
the project directory, never the database, never a Stage output (see
project notes section 25).

`keyring` is treated as an optional dependency here: if it isn't installed,
or no OS credential backend is available in the current environment, every
function below degrades to a no-op / None rather than raising. The app
still works fine with just --openrouter-api-key / OPENROUTER_API_KEY --
the user just has to supply it every run, exactly as before this module
existed.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SERVICE_NAME = "exampapersorter"
_USERNAME = "openrouter_api_key"


def _keyring_module():
    try:
        import keyring
    except ImportError:
        return None
    return keyring


def load_saved_key() -> str | None:
    kr = _keyring_module()
    if kr is None:
        return None
    try:
        return kr.get_password(_SERVICE_NAME, _USERNAME)
    # Deliberately broad: any OS credential-backend failure (locked keyring,
    # no backend configured, headless session, ...) must degrade to "no
    # saved key" rather than crash a run over a convenience feature.
    except Exception as exc:
        logger.debug("Could not read saved OpenRouter API key from OS credential store: %s", exc)
        return None


def save_key(api_key: str) -> bool:
    kr = _keyring_module()
    if kr is None:
        return False
    try:
        kr.set_password(_SERVICE_NAME, _USERNAME, api_key)
        return True
    except Exception as exc:
        logger.debug("Could not save OpenRouter API key to OS credential store: %s", exc)
        return False


def clear_saved_key() -> bool:
    kr = _keyring_module()
    if kr is None:
        return False
    try:
        kr.delete_password(_SERVICE_NAME, _USERNAME)
        return True
    except Exception as exc:
        logger.debug("Could not clear saved OpenRouter API key: %s", exc)
        return False
