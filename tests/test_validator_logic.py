import pytest

import core
from core.commitments import CodecCommitment, ParsedCommitment
from validation.precheck import PrecheckResult
from core.state import ValidatorState
from validator.service import _apply_precheck, _assert_version_key_matches, _make_provider, decide_weights
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


# --- corpus default -------------------------------------------------------------

def test_make_provider_defaults_to_mixed_corpus_env(monkeypatch, tmp_path):
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    (mixed / "chunk_00_fineweb.txt").write_bytes(b"fineweb")
    (mixed / "provenance.json").write_text("{}")
    monkeypatch.setenv("GLYPH_MIXED_CORPUS_DIR", str(mixed))

    args = type("Args", (), {"corpus_dir": None, "corpus_url": None})()
    provider = _make_provider(args)

    assert provider.directory == mixed
    assert provider.total_bytes == len(b"fineweb")


def test_make_provider_explicit_corpus_dir_wins(monkeypatch, tmp_path):
    default = tmp_path / "default"
    explicit = tmp_path / "explicit"
    default.mkdir()
    explicit.mkdir()
    (default / "chunk_default.txt").write_bytes(b"default")
    (explicit / "chunk_explicit.txt").write_bytes(b"explicit")
    monkeypatch.setenv("GLYPH_MIXED_CORPUS_DIR", str(default))

    args = type("Args", (), {"corpus_dir": str(explicit), "corpus_url": None})()
    provider = _make_provider(args)

    assert provider.directory == explicit
    assert provider.total_bytes == len(b"explicit")


def test_make_provider_preserves_corpus_url_on_default(monkeypatch, tmp_path):
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    (mixed / "chunk_00_fineweb.txt").write_bytes(b"fineweb")
    monkeypatch.setenv("GLYPH_MIXED_CORPUS_DIR", str(mixed))

    args = type("Args", (), {"corpus_dir": None, "corpus_url": "https://host/corpus.bin"})()
    provider = _make_provider(args)

    source = provider.stream_source(type("Spec", (), {"offset": 2, "length": 3})())
    assert source.url == "https://host/corpus.bin"


def test_make_provider_missing_default_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("GLYPH_MIXED_CORPUS_DIR", str(tmp_path / "missing"))

    args = type("Args", (), {"corpus_dir": None, "corpus_url": None})()
    with pytest.raises(SystemExit) as exc:
        _make_provider(args)

    assert "default mixed corpus directory not found" in str(exc.value)


def test_make_provider_empty_corpus_fails(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    args = type("Args", (), {"corpus_dir": str(empty), "corpus_url": None})()
    with pytest.raises(SystemExit) as exc:
        _make_provider(args)

    assert "no benchmark data files" in str(exc.value)


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


# --- commit-reveal tie-break block (exploit vector #9) --------------------------

def _ok_precheck(repo, revision, *, max_artifact_bytes, download=True):
    return PrecheckResult(repo=repo, revision=revision, ok=True, artifact_hash=repo, artifact_bytes=10)


def test_reveal_tie_breaks_off_observed_commit_phase_block(monkeypatch):
    from core.commitments import commitment_digest

    monkeypatch.setattr("validator.service.precheck_codec", _ok_precheck)
    salt = "00112233"
    digest = commitment_digest("a/codec", "abc123", salt)
    state = ValidatorState()
    # Validator observed this hotkey's commit-phase digest at block 5.
    state.commit_phase_seen = {"hotkey-a": {digest: 5}}
    reveal = ParsedCommitment(
        "hotkey-a", CodecCommitment(repo="a/codec", rev="abc123"), "raw", salt=salt, digest=digest
    )
    # Reveal is processed much later, at block 50.
    _apply_precheck(state, [reveal], max_artifact_bytes=100, block=50)
    # commit_block keys off the commit-phase block, not the reveal-observation block.
    assert state.commitments["hotkey-a:a/codec@abc123"].block == 5
    # Reveal resolved -> the commit-phase digest is dropped so the map stays bounded (#21).
    assert "hotkey-a" not in state.commit_phase_seen


def test_reveal_without_observed_commit_phase_falls_back_to_current_block(monkeypatch):
    from core.commitments import commitment_digest

    monkeypatch.setattr("validator.service.precheck_codec", _ok_precheck)
    salt = "00112233"
    digest = commitment_digest("a/codec", "abc123", salt)
    state = ValidatorState()  # commit phase never observed
    reveal = ParsedCommitment(
        "hotkey-a", CodecCommitment(repo="a/codec", rev="abc123"), "raw", salt=salt, digest=digest
    )
    _apply_precheck(state, [reveal], max_artifact_bytes=100, block=50)
    assert state.commitments["hotkey-a:a/codec@abc123"].block == 50


# --- temporal burn weights ------------------------------------------------------

def _window_blocks():
    return [i * TEMPO for i in range(4)]


def test_burn_disabled_never_burns_in_decide_weights():
    """issue #43: BURN_ENABLED = False is the shipped default -- no tempo ever burns."""

    history = [WinnerEntry("hkA", "a/c", "rev123456", 0.5, 1)]
    outputs = [("s0", 100, "hash0")]
    for block in _window_blocks():
        weights, burn = decide_weights(
            HOTKEYS, history, block=block, tempo=TEMPO, last_round_outputs=outputs, anchor=ANCHOR
        )
        assert burn is False
        assert weights[1] == 1.0  # sole winner takes everything, every tempo


def test_burn_reenabled_gives_exactly_one_burn_tempo_per_window(monkeypatch):
    """Flipping BURN_ENABLED back to True must restore the original schedule exactly."""

    import weight_setter.service as weight_setter_service

    monkeypatch.setattr(weight_setter_service, "BURN_ENABLED", True)
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
