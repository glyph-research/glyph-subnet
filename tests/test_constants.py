"""Launch-constant configuration contract (issue #2).

Two invariants the subnet relies on:

* ``CHUTE_USERNAME`` is *deployment-specific* and overridable via ``GLYPH_CHUTE_USERNAME``.
* ``WINDOW_ANCHOR_BLOCK`` is *consensus-critical* and must be a fixed in-source constant —
  no environment variable may shift it, or validators' burn windows would diverge.
"""

import importlib

import core.constants as constants


def _reload(monkeypatch, **env):
    """Reload core.constants with a patched environment, restoring it afterwards."""
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    return importlib.reload(constants)


def test_chute_username_defaults_to_glyph(monkeypatch):
    mod = _reload(monkeypatch, GLYPH_CHUTE_USERNAME=None)
    assert mod.CHUTE_USERNAME == "glyph"


def test_chute_username_honours_env_override(monkeypatch):
    mod = _reload(monkeypatch, GLYPH_CHUTE_USERNAME="acme")
    assert mod.CHUTE_USERNAME == "acme"
    # the derived default chute URL follows the override
    importlib.reload(importlib.import_module("eval.runner_chutes"))
    from eval.runner_chutes import DEFAULT_BASE_URL

    assert DEFAULT_BASE_URL == "https://acme-glyph-runner.chutes.ai"


def test_window_anchor_is_a_fixed_int_not_env_configurable(monkeypatch):
    baseline = constants.WINDOW_ANCHOR_BLOCK
    assert isinstance(baseline, int)
    # No env var may shift the anchor — it is committed in source for network-wide determinism.
    mod = _reload(
        monkeypatch,
        GLYPH_WINDOW_ANCHOR_BLOCK="12345",
        WINDOW_ANCHOR_BLOCK="12345",
    )
    assert mod.WINDOW_ANCHOR_BLOCK == baseline


def teardown_module(_module):
    """Leave the imported modules in their committed-default state for other tests."""
    importlib.reload(constants)
    importlib.reload(importlib.import_module("eval.runner_chutes"))
