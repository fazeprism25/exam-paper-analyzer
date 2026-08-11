import sys

import exampapersorter.api_key_store as api_key_store


class _FakeKeyring:
    def __init__(self):
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service, username):
        return self._store.get((service, username))

    def set_password(self, service, username, password):
        self._store[(service, username)] = password

    def delete_password(self, service, username):
        del self._store[(service, username)]


class _RaisingKeyring:
    def get_password(self, service, username):
        raise RuntimeError("no OS backend available")

    def set_password(self, service, username, password):
        raise RuntimeError("no OS backend available")

    def delete_password(self, service, username):
        raise RuntimeError("no OS backend available")


def test_load_returns_none_when_nothing_saved(monkeypatch):
    monkeypatch.setattr(api_key_store, "_keyring_module", lambda: _FakeKeyring())
    assert api_key_store.load_saved_key() is None


def test_save_then_load_round_trips(monkeypatch):
    fake = _FakeKeyring()
    monkeypatch.setattr(api_key_store, "_keyring_module", lambda: fake)

    assert api_key_store.save_key("sk-test-123") is True
    assert api_key_store.load_saved_key() == "sk-test-123"


def test_clear_removes_saved_key(monkeypatch):
    fake = _FakeKeyring()
    monkeypatch.setattr(api_key_store, "_keyring_module", lambda: fake)

    api_key_store.save_key("sk-test-123")
    assert api_key_store.clear_saved_key() is True
    assert api_key_store.load_saved_key() is None


def test_missing_keyring_package_degrades_to_noop(monkeypatch):
    monkeypatch.setattr(api_key_store, "_keyring_module", lambda: None)
    assert api_key_store.load_saved_key() is None
    assert api_key_store.save_key("sk-test-123") is False
    assert api_key_store.clear_saved_key() is False


def test_backend_errors_never_propagate(monkeypatch):
    monkeypatch.setattr(api_key_store, "_keyring_module", lambda: _RaisingKeyring())
    assert api_key_store.load_saved_key() is None
    assert api_key_store.save_key("sk-test-123") is False
    assert api_key_store.clear_saved_key() is False


def test_real_import_path_is_used_when_keyring_installed():
    # Confirms _keyring_module() actually reaches for the real `keyring`
    # package rather than something hard-coded, when it's importable.
    if "keyring" not in sys.modules:
        import importlib
        try:
            importlib.import_module("keyring")
        except ImportError:
            return  # keyring genuinely not installed in this environment -- nothing to confirm
    module = api_key_store._keyring_module()
    assert module is not None
    assert module.__name__ == "keyring"
