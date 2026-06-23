import pytest

from core.dotenv import load_dotenv
from eval.runner_chutes import RunnerError, _load_api_key


# --- API key loading -------------------------------------------------------------

def test_env_var_used(monkeypatch):
    monkeypatch.setenv("CHUTES_API_KEY", "cpk_from_env")
    assert _load_api_key(None) == "cpk_from_env"


def test_explicit_key_file_overrides_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CHUTES_API_KEY", "cpk_from_env")
    key_file = tmp_path / "key.txt"
    key_file.write_text("cpk_from_file\n")
    assert _load_api_key(str(key_file)) == "cpk_from_file"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("CHUTES_API_KEY", raising=False)
    with pytest.raises(RunnerError):
        _load_api_key(None)


def test_missing_key_file_raises(monkeypatch):
    monkeypatch.delenv("CHUTES_API_KEY", raising=False)
    with pytest.raises(RunnerError):
        _load_api_key("/nonexistent/key.txt")


# --- .env loader -----------------------------------------------------------------

def test_load_dotenv_sets_unset_vars(monkeypatch, tmp_path):
    monkeypatch.delenv("CHUTES_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('# comment\nexport CHUTES_API_KEY="cpk_dotenv"\nGLYPH_CHUTE_URL=https://x\n')
    assert load_dotenv(env) is True
    assert _load_api_key(None) == "cpk_dotenv"
    import os

    assert os.environ["GLYPH_CHUTE_URL"] == "https://x"


def test_load_dotenv_does_not_override_existing(monkeypatch, tmp_path):
    monkeypatch.setenv("CHUTES_API_KEY", "cpk_already")
    env = tmp_path / ".env"
    env.write_text("CHUTES_API_KEY=cpk_dotenv\n")
    load_dotenv(env)
    assert _load_api_key(None) == "cpk_already"


def test_load_dotenv_missing_file_returns_false(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") is False
