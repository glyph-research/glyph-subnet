import pytest

import core
from core.commitments import CodecCommitment, ParsedCommitment
from validation.precheck import PrecheckResult
from core.state import ValidatorState
from validator.service import _apply_precheck, _assert_version_key_matches, decide_weights
from core.weights import WinnerEntry

TEMPO = 360
ANCHOR = 0
HOTKEYS = ["uid0_burn", "hkA", "hkB"]


class FakeChain:
    def __init__(self, version: int):
        self.version = version
        self.config = type("Config", (), {"netuid": 488})()

    def get_weights_version(self) -> int:
        return self.version


# --- version-key gate (fail closed) ---------------------------------------------

def test_version_key_match_allows(monkeypatch):
    monkeypatch.setattr(core, "__version_key__", 7)
    assert _assert_version_key_matches(FakeChain(7)) == 7


def test_version_key_mismatch_stops(monkeypatch):
    monkeypatch.setattr(core, "__version_key__", 7)
    with pytest.raises(SystemExit) as exc:
        _assert_version_key_matches(FakeChain(8))
    assert "version key mismatch" in str(exc.value)


# --- duplicate artifact-hash disqualification -----------------------------------

def test_apply_precheck_disqualifies_duplicate_hash(monkeypatch):
    def fake_precheck(repo, revision, *, max_artifact_bytes, download=True):
        return PrecheckResult(repo=repo, revision=revision, ok=True, artifact_hash="same", artifact_bytes=10)

    monkeypatch.setattr("validator.service.precheck_codec", fake_precheck)
    state = ValidatorState()
    parsed = [
        ParsedCommitment("hotkey-a", CodecCommitment(repo="a/codec", rev="abc123"), "raw-a"),
        ParsedCommitment("hotkey-b", CodecCommitment(repo="b/codec", rev="def456"), "raw-b"),
    ]
    _apply_precheck(state, parsed, max_artifact_bytes=100, block=10)

    first = state.commitments["hotkey-a:a/codec@abc123"]
    second = state.commitments["hotkey-b:b/codec@def456"]
    assert first.valid is True
    assert second.valid is False
    assert "duplicate artifact" in second.disqualification_reason


# --- temporal burn weights ------------------------------------------------------

def _window_blocks():
    return [i * TEMPO for i in range(4)]


def test_exactly_one_burn_tempo_per_window_in_decide_weights():
    history = [WinnerEntry("hkA", "a/c", "rev123456", 0.5, 1)]
    outputs = [("s0", 100, "hash0")]
    flags = []
    for block in _window_blocks():
        weights, burn = decide_weights(
            HOTKEYS, history, block=block, tempo=TEMPO, last_round_outputs=outputs, anchor=ANCHOR
        )
        flags.append(burn)
        if burn:
            assert weights[0] == 1.0  # all to burn UID
            assert sum(weights[1:]) == 0.0
        else:
            assert weights[1] == 1.0  # sole winner takes everything
    assert sum(flags) == 1


def test_idle_empty_history_burns_on_normal_tempo():
    # Find a non-burn tempo for empty-history outputs and assert it burns anyway.
    outputs = []
    for block in _window_blocks():
        weights, burn = decide_weights(
            HOTKEYS, [], block=block, tempo=TEMPO, last_round_outputs=outputs, anchor=ANCHOR
        )
        # No winner yet -> emission is always burned, burn tempo or not.
        assert weights[0] == 1.0


def test_two_winner_split_on_normal_tempo():
    history = [WinnerEntry("hkA", "a/c", "rev123456", 0.4, 1), WinnerEntry("hkB", "b/c", "rev654321", 0.5, 2)]
    outputs = [("s0", 100, "h")]
    # find a non-burn tempo
    for block in _window_blocks():
        weights, burn = decide_weights(
            HOTKEYS, history, block=block, tempo=TEMPO, last_round_outputs=outputs, anchor=ANCHOR
        )
        if not burn:
            assert weights[1] == pytest.approx(0.70)
            assert weights[2] == pytest.approx(0.30)
            break
