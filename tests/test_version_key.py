from argparse import Namespace

import pytest

import core
from core.state import ValidatorState
from core.version import assert_weights_version_matches, local_version_key
from weight_setter.service import run as run_weight_setter


class FakeChain:
    def __init__(self, version: int):
        self.version = version
        self.config = type("Config", (), {"netuid": 488})()

    def get_weights_version(self) -> int:
        return self.version


def test_local_version_key_reads_core_value(monkeypatch):
    monkeypatch.setattr(core, "__version_key__", 11)
    assert local_version_key() == 11


def test_shared_version_key_gate_fails_closed(monkeypatch):
    monkeypatch.setattr(core, "__version_key__", 11)
    with pytest.raises(SystemExit) as exc:
        assert_weights_version_matches(FakeChain(12))
    assert "version key mismatch" in str(exc.value)
    assert "netuid 488" in str(exc.value)


def test_weight_setter_checks_weights_version_before_chain_reads(monkeypatch, tmp_path):
    class MismatchedChain(FakeChain):
        def __init__(self, _config):
            super().__init__(12)

        def current_block(self):  # pragma: no cover - must not be reached
            raise AssertionError("weight setter read chain state after version mismatch")

    monkeypatch.setattr(core, "__version_key__", 11)
    monkeypatch.setattr("weight_setter.service.load_state", lambda _path: ValidatorState())
    monkeypatch.setattr("chain.chain.BittensorChain", MismatchedChain)

    args = Namespace(
        netuid=488,
        network="test",
        wallet_name="validator",
        hotkey_name="default",
        wallet_path=None,
        state_dir=str(tmp_path),
        burn_uid=0,
        window_anchor=0,
        dry_run=True,
    )
    with pytest.raises(SystemExit) as exc:
        run_weight_setter(args)
    assert "version key mismatch" in str(exc.value)


def test_auto_update_tracks_version_key():
    script = "scripts/run_auto_validator.sh"
    with open(script, encoding="utf-8") as handle:
        content = handle.read()
    assert 'VERSION_VAR="__version_key__"' in content
    assert "__version__" not in content
