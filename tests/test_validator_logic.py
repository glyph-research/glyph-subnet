import sys

import pytest
from bittensor.utils.btlogging import logging as bt_logging

import core
from core.commitments import CodecCommitment, ParsedCommitment
from validation.precheck import PrecheckResult
from core.state import CommitmentState, ValidatorState
from validator.service import (
    DEFAULT_DEMO_CORPUS_DIR,
    _apply_precheck,
    _assert_version_key_matches,
    _make_demo_provider,
    _make_runner,
    decide_weights,
    run_once,
)
from core.weights import WinnerEntry

# issue #96: CodecCommitment.rev must be a pinned 40-char git commit SHA now, not any
# non-empty string -- stand-ins for what HfApi().repo_info(...).sha actually returns.
_REV_A = "a" * 40
_REV_B = "b" * 40

TEMPO = 360
ANCHOR = 0
HOTKEYS = ["uid0_burn", "hkA", "hkB"]


class FakeChain:
    def __init__(self, version: int):
        self.version = version
        self.config = type("Config", (), {"netuid": 488})()

    def get_weights_version(self) -> int:
        return self.version


# --- offline-demo corpus (issue #71: a real round never uses this) --------------------

def test_make_demo_provider_defaults_to_bundled_sample_corpus():
    args = type("Args", (), {"corpus_dir": None})()
    provider = _make_demo_provider(args)

    assert provider.directory == DEFAULT_DEMO_CORPUS_DIR
    assert provider.total_bytes > 0


def test_make_demo_provider_explicit_corpus_dir_wins(tmp_path):
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    (explicit / "chunk_explicit.txt").write_bytes(b"explicit")

    args = type("Args", (), {"corpus_dir": str(explicit)})()
    provider = _make_demo_provider(args)

    assert provider.directory == explicit
    assert provider.total_bytes == len(b"explicit")


def test_make_demo_provider_missing_explicit_dir_fails(tmp_path):
    args = type("Args", (), {"corpus_dir": str(tmp_path / "missing")})()
    with pytest.raises(SystemExit) as exc:
        _make_demo_provider(args)

    assert "corpus directory not found" in str(exc.value)


def test_make_demo_provider_empty_corpus_fails(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    args = type("Args", (), {"corpus_dir": str(empty)})()
    with pytest.raises(SystemExit) as exc:
        _make_demo_provider(args)

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
        ParsedCommitment("hotkey-a", CodecCommitment(repo="a/codec", rev=_REV_A), "raw-a"),
        ParsedCommitment("hotkey-b", CodecCommitment(repo="b/codec", rev=_REV_B), "raw-b"),
    ]
    _apply_precheck(state, parsed, max_artifact_bytes=100, block=10)

    first = state.commitments[f"hotkey-a:a/codec@{_REV_A}"]
    second = state.commitments[f"hotkey-b:b/codec@{_REV_B}"]
    assert first.valid is True
    assert second.valid is False
    assert "duplicate artifact" in second.disqualification_reason


# --- precheck/round visibility logging (issue #81) ------------------------------


def test_apply_precheck_logs_valid_and_invalid_per_hotkey(monkeypatch, caplog):
    bt_logging.set_info()

    def fake_precheck(repo, revision, *, max_artifact_bytes, download=True):
        ok = repo == "a/codec"
        return PrecheckResult(
            repo=repo, revision=revision, ok=ok, artifact_hash="a" if ok else None,
            artifact_bytes=10 if ok else None, errors=[] if ok else ["too big"],
        )

    monkeypatch.setattr("validator.service.precheck_codec", fake_precheck)
    state = ValidatorState()
    parsed = [
        ParsedCommitment("hotkey-a", CodecCommitment(repo="a/codec", rev=_REV_A), "raw-a"),
        ParsedCommitment("hotkey-b", CodecCommitment(repo="b/codec", rev=_REV_B), "raw-b"),
    ]
    _apply_precheck(state, parsed, max_artifact_bytes=100, block=10)

    out = caplog.text
    assert f"precheck: hotkey-a a/codec@{_REV_A} valid" in out
    assert f"precheck: hotkey-b b/codec@{_REV_B} invalid: too big" in out


# --- precheck full re-check cadence: not skipped forever (issue #96) ------------


def _recording_precheck(calls):
    def fake_precheck(repo, revision, *, max_artifact_bytes, download=True):
        calls.append(download)
        return PrecheckResult(repo=repo, revision=revision, ok=True, artifact_hash="same-hash", artifact_bytes=10)

    return fake_precheck


def test_first_sight_always_does_a_full_check(monkeypatch):
    calls = []
    monkeypatch.setattr("validator.service.precheck_codec", _recording_precheck(calls))
    state = ValidatorState()
    parsed = [ParsedCommitment("hotkey-a", CodecCommitment(repo="a/codec", rev=_REV_A), "raw-a")]

    _apply_precheck(state, parsed, max_artifact_bytes=100, block=1000)

    assert calls == [True]
    assert state.commitments[f"hotkey-a:a/codec@{_REV_A}"].last_full_check_block == 1000


def test_soon_after_a_full_check_skips_the_next_one(monkeypatch):
    calls = []
    monkeypatch.setattr("validator.service.precheck_codec", _recording_precheck(calls))
    state = ValidatorState()
    parsed = [ParsedCommitment("hotkey-a", CodecCommitment(repo="a/codec", rev=_REV_A), "raw-a")]

    _apply_precheck(state, parsed, max_artifact_bytes=100, block=1000)
    _apply_precheck(state, parsed, max_artifact_bytes=100, block=1001)

    assert calls == [True, False]
    # last_full_check_block doesn't move on the skipped (manifest-only) round.
    assert state.commitments[f"hotkey-a:a/codec@{_REV_A}"].last_full_check_block == 1000


def test_full_check_is_forced_again_once_the_interval_elapses(monkeypatch):
    from core.constants import PRECHECK_FULL_RECHECK_INTERVAL_BLOCKS

    calls = []
    monkeypatch.setattr("validator.service.precheck_codec", _recording_precheck(calls))
    state = ValidatorState()
    parsed = [ParsedCommitment("hotkey-a", CodecCommitment(repo="a/codec", rev=_REV_A), "raw-a")]

    _apply_precheck(state, parsed, max_artifact_bytes=100, block=1000)
    later_block = 1000 + PRECHECK_FULL_RECHECK_INTERVAL_BLOCKS
    _apply_precheck(state, parsed, max_artifact_bytes=100, block=later_block)

    assert calls == [True, True]
    assert state.commitments[f"hotkey-a:a/codec@{_REV_A}"].last_full_check_block == later_block


def test_persisted_state_without_last_full_check_block_forces_a_full_check(monkeypatch):
    # Migration case: state persisted before this field existed has last_full_check_block=None
    # even though artifact_hash is already set -- must not be trusted as "already fully
    # checked" forever.
    calls = []
    monkeypatch.setattr("validator.service.precheck_codec", _recording_precheck(calls))
    state = ValidatorState()
    state.commitments[f"hotkey-a:a/codec@{_REV_A}"] = CommitmentState(
        hotkey="hotkey-a", repo="a/codec", revision=_REV_A, block=1, artifact_hash="same-hash",
        valid=True, last_full_check_block=None,
    )
    parsed = [ParsedCommitment("hotkey-a", CodecCommitment(repo="a/codec", rev=_REV_A), "raw-a")]

    _apply_precheck(state, parsed, max_artifact_bytes=100, block=1000)

    assert calls == [True]
    assert state.commitments[f"hotkey-a:a/codec@{_REV_A}"].last_full_check_block == 1000


# --- duplicate-artifact ownership: earliest commit_block wins, not hotkey order (#58) -------


def _same_hash_precheck(repo, revision, *, max_artifact_bytes, download=True):
    return PrecheckResult(repo=repo, revision=revision, ok=True, artifact_hash="same-hash", artifact_bytes=10)


def _seed_existing_commitment(state, hotkey, repo, rev, block):
    key = f"{hotkey}:{repo}@{rev}"
    state.commitments[key] = CommitmentState(
        hotkey=hotkey, repo=repo, revision=rev, block=block, artifact_hash=None, valid=False
    )


def test_duplicate_owner_prefers_earlier_block_when_earlier_hotkey_sorts_first(monkeypatch):
    # hotkey-a committed earlier (block 5) AND sorts first -- both signals agree, so this
    # alone wouldn't distinguish the fix from the old (buggy) hotkey-sort behavior.
    monkeypatch.setattr("validator.service.precheck_codec", _same_hash_precheck)
    state = ValidatorState()
    _seed_existing_commitment(state, "hotkey-a", "a/codec", _REV_A, block=5)
    _seed_existing_commitment(state, "hotkey-z", "z/codec", _REV_B, block=20)
    parsed = [
        ParsedCommitment("hotkey-a", CodecCommitment(repo="a/codec", rev=_REV_A), "raw-a"),
        ParsedCommitment("hotkey-z", CodecCommitment(repo="z/codec", rev=_REV_B), "raw-z"),
    ]
    _apply_precheck(state, parsed, max_artifact_bytes=100, block=999)
    a = state.commitments[f"hotkey-a:a/codec@{_REV_A}"]
    z = state.commitments[f"hotkey-z:z/codec@{_REV_B}"]
    assert a.valid is True
    assert z.valid is False
    assert "hotkey-a" in z.disqualification_reason


def test_duplicate_owner_prefers_earlier_block_when_later_hotkey_sorts_first(monkeypatch):
    # hotkey-a sorts first lexicographically but committed LATER (block 20); hotkey-z sorts
    # last but committed EARLIER (block 5). The old hotkey-sort-order logic would have made
    # hotkey-a the owner here -- this is the exact copy-cat exploit scenario from the issue.
    monkeypatch.setattr("validator.service.precheck_codec", _same_hash_precheck)
    state = ValidatorState()
    _seed_existing_commitment(state, "hotkey-a", "a/codec", _REV_A, block=20)
    _seed_existing_commitment(state, "hotkey-z", "z/codec", _REV_B, block=5)
    parsed = [
        ParsedCommitment("hotkey-a", CodecCommitment(repo="a/codec", rev=_REV_A), "raw-a"),
        ParsedCommitment("hotkey-z", CodecCommitment(repo="z/codec", rev=_REV_B), "raw-z"),
    ]
    _apply_precheck(state, parsed, max_artifact_bytes=100, block=999)
    a = state.commitments[f"hotkey-a:a/codec@{_REV_A}"]
    z = state.commitments[f"hotkey-z:z/codec@{_REV_B}"]
    assert z.valid is True
    assert a.valid is False
    assert "hotkey-z" in a.disqualification_reason


def test_duplicate_owner_equal_commit_block_ties_break_by_hotkey(monkeypatch):
    monkeypatch.setattr("validator.service.precheck_codec", _same_hash_precheck)
    state = ValidatorState()
    _seed_existing_commitment(state, "hotkey-a", "a/codec", _REV_A, block=10)
    _seed_existing_commitment(state, "hotkey-z", "z/codec", _REV_B, block=10)
    parsed = [
        ParsedCommitment("hotkey-z", CodecCommitment(repo="z/codec", rev=_REV_B), "raw-z"),
        ParsedCommitment("hotkey-a", CodecCommitment(repo="a/codec", rev=_REV_A), "raw-a"),
    ]
    _apply_precheck(state, parsed, max_artifact_bytes=100, block=999)
    a = state.commitments[f"hotkey-a:a/codec@{_REV_A}"]
    z = state.commitments[f"hotkey-z:z/codec@{_REV_B}"]
    assert a.valid is True
    assert z.valid is False


def test_duplicate_owner_stays_sticky_across_rounds(monkeypatch):
    # Once decided, a hash's owner doesn't get re-litigated by a later round just because a
    # new contender happens to have an earlier commit_block -- first-decided sticks.
    monkeypatch.setattr("validator.service.precheck_codec", _same_hash_precheck)
    state = ValidatorState()
    state.duplicate_hash_owner = {"same-hash": "hotkey-a"}
    _seed_existing_commitment(state, "hotkey-z", "z/codec", _REV_B, block=0)  # earlier than any real block
    parsed = [ParsedCommitment("hotkey-z", CodecCommitment(repo="z/codec", rev=_REV_B), "raw-z")]
    _apply_precheck(state, parsed, max_artifact_bytes=100, block=999)
    z = state.commitments[f"hotkey-z:z/codec@{_REV_B}"]
    assert z.valid is False
    assert "hotkey-a" in z.disqualification_reason


# --- local runner defaults to strict sandbox (issue #56) ------------------------


def _args(**overrides):
    defaults = {"runner": "local", "unsafe_local_no_sandbox": False}
    return type("Args", (), {**defaults, **overrides})()


def test_make_runner_local_defaults_to_strict_sandbox():
    from eval.runner import LocalSubprocessRunner

    runner = _make_runner(_args())
    assert isinstance(runner, LocalSubprocessRunner)
    assert runner.strict_sandbox is True
    assert runner.require_network_isolation is True


def test_make_runner_local_unsafe_flag_is_refused():
    with pytest.raises(SystemExit, match="refused"):
        _make_runner(_args(unsafe_local_no_sandbox=True))


# --- commit-reveal tie-break block (exploit vector #9) --------------------------

def _ok_precheck(repo, revision, *, max_artifact_bytes, download=True):
    return PrecheckResult(repo=repo, revision=revision, ok=True, artifact_hash=repo, artifact_bytes=10)


def test_reveal_tie_breaks_off_observed_commit_phase_block(monkeypatch):
    from core.commitments import commitment_digest

    monkeypatch.setattr("validator.service.precheck_codec", _ok_precheck)
    salt = "00112233"
    digest = commitment_digest("a/codec", _REV_A, salt)
    state = ValidatorState()
    # Validator observed this hotkey's commit-phase digest at block 5.
    state.commit_phase_seen = {"hotkey-a": {digest: 5}}
    reveal = ParsedCommitment(
        "hotkey-a", CodecCommitment(repo="a/codec", rev=_REV_A), "raw", salt=salt, digest=digest
    )
    # Reveal is processed much later, at block 50.
    _apply_precheck(state, [reveal], max_artifact_bytes=100, block=50)
    # commit_block keys off the commit-phase block, not the reveal-observation block.
    assert state.commitments[f"hotkey-a:a/codec@{_REV_A}"].block == 5
    # Reveal resolved -> the commit-phase digest is dropped so the map stays bounded (#21).
    assert "hotkey-a" not in state.commit_phase_seen


def test_reveal_without_observed_commit_phase_falls_back_to_current_block(monkeypatch):
    from core.commitments import commitment_digest

    monkeypatch.setattr("validator.service.precheck_codec", _ok_precheck)
    salt = "00112233"
    digest = commitment_digest("a/codec", _REV_A, salt)
    state = ValidatorState()  # commit phase never observed
    reveal = ParsedCommitment(
        "hotkey-a", CodecCommitment(repo="a/codec", rev=_REV_A), "raw", salt=salt, digest=digest
    )
    _apply_precheck(state, [reveal], max_artifact_bytes=100, block=50)
    assert state.commitments[f"hotkey-a:a/codec@{_REV_A}"].block == 50


# --- temporal burn weights ------------------------------------------------------

def _window_blocks():
    return [i * TEMPO for i in range(4)]


def test_burn_enabled_by_default_gives_exactly_one_burn_tempo_per_window():
    """issue #88: BURN_ENABLED = True is the shipped default -- the schedule applies."""

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


def test_burn_disabled_via_monkeypatch_never_burns_in_decide_weights(monkeypatch):
    """The kill-switch direction: flipping BURN_ENABLED to False must restore the pure
    rolling-winner distribution with no tempo ever burning."""

    import weight_setter.service as weight_setter_service

    monkeypatch.setattr(weight_setter_service, "BURN_ENABLED", False)
    history = [WinnerEntry("hkA", "a/c", "rev123456", 0.5, 1)]
    outputs = [("s0", 100, "hash0")]
    for block in _window_blocks():
        weights, burn = decide_weights(
            HOTKEYS, history, block=block, tempo=TEMPO, last_round_outputs=outputs, anchor=ANCHOR
        )
        assert burn is False
        assert weights[1] == 1.0  # sole winner takes everything, every tempo


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


# --- run_once console logging (issue #77) ---------------------------------------


class _FakeWandb:
    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


class _FakeChain:
    def __init__(self, rate_limit_remaining=None):
        self._rate_limit_remaining = rate_limit_remaining
        self.set_weights_called = False

    def commit_reveal_enabled(self):
        return True

    def tempo(self):
        return 360

    def metagraph(self):
        return type("Metagraph", (), {"hotkeys": ["hk0"], "uids": [0]})()

    def set_weights(self, uids, weights, version_key):
        self.set_weights_called = True
        return type("Response", (), {"success": True, "error": None, "message": None})()

    def blocks_until_weights_allowed(self):
        return self._rate_limit_remaining


def test_run_once_prints_champion_and_concise_set_weights_on_a_quiet_round(monkeypatch, tmp_path, caplog):
    # A quiet round (no new challengers) must still report champion/ratio state, and
    # set_weights must print a short success/failure summary, not the full response dump.
    # Console output goes through bt.logging (issue #80), not print/capsys -- caplog captures it.
    bt_logging.set_info()
    state = ValidatorState()
    state.winner_history = [WinnerEntry("champ", "r/c", "rev123456", 0.42, 1)]

    monkeypatch.setattr("validator.service.load_state", lambda path: state)
    monkeypatch.setattr("validator.service.save_state", lambda path, s: None)
    monkeypatch.setattr("validator.service._make_chain", lambda args: _FakeChain())
    monkeypatch.setattr("validator.service._local_version_key", lambda: 1)
    monkeypatch.setattr("validator.service._assert_version_key_matches", lambda chain: 1)
    monkeypatch.setattr("validator.service._evaluate_round", lambda args, state, chain, salt: (999, None))
    monkeypatch.setattr("validator.service.decide_weights", lambda *a, **k: ([1.0], False))

    args = type(
        "Args", (),
        {"state_dir": str(tmp_path), "salt_file": None, "dry_run": False, "burn_uid": 0, "window_anchor": 0},
    )()

    run_once(args, wandb_logger=_FakeWandb())

    out = caplog.text
    assert "round: block=999 champion=champ ratio=0.4200 0 challengers" in out
    assert "set_weights: success=True" in out
    assert "set_weights response:" not in out


def test_run_once_prints_no_champion_when_history_empty(monkeypatch, tmp_path, caplog):
    bt_logging.set_info()
    state = ValidatorState()

    monkeypatch.setattr("validator.service.load_state", lambda path: state)
    monkeypatch.setattr("validator.service.save_state", lambda path, s: None)
    monkeypatch.setattr("validator.service._make_chain", lambda args: _FakeChain())
    monkeypatch.setattr("validator.service._local_version_key", lambda: 1)
    monkeypatch.setattr("validator.service._assert_version_key_matches", lambda chain: 1)
    monkeypatch.setattr("validator.service._evaluate_round", lambda args, state, chain, salt: (999, None))
    monkeypatch.setattr("validator.service.decide_weights", lambda *a, **k: ([1.0], False))

    args = type(
        "Args", (),
        {"state_dir": str(tmp_path), "salt_file": None, "dry_run": False, "burn_uid": 0, "window_anchor": 0},
    )()

    run_once(args, wandb_logger=_FakeWandb())

    out = caplog.text
    assert "champion=none" in out


def test_run_once_skips_set_weights_when_rate_limited(monkeypatch, tmp_path, caplog):
    # issue #79: below the subnet's weights-rate-limit, chain.set_weights would return a bare
    # contentless failure (no error, no message) by construction in the real SDK -- detect
    # and log this explicitly instead of attempting and reporting a silent failure.
    bt_logging.set_info()
    state = ValidatorState()
    fake_chain = _FakeChain(rate_limit_remaining=42)

    monkeypatch.setattr("validator.service.load_state", lambda path: state)
    monkeypatch.setattr("validator.service.save_state", lambda path, s: None)
    monkeypatch.setattr("validator.service._make_chain", lambda args: fake_chain)
    monkeypatch.setattr("validator.service._local_version_key", lambda: 1)
    monkeypatch.setattr("validator.service._assert_version_key_matches", lambda chain: 1)
    monkeypatch.setattr("validator.service._evaluate_round", lambda args, state, chain, salt: (999, None))
    monkeypatch.setattr("validator.service.decide_weights", lambda *a, **k: ([1.0], False))

    args = type(
        "Args", (),
        {"state_dir": str(tmp_path), "salt_file": None, "dry_run": False, "burn_uid": 0, "window_anchor": 0},
    )()

    run_once(args, wandb_logger=_FakeWandb())

    assert fake_chain.set_weights_called is False
    assert "set_weights: skipped, rate-limited (42 blocks remaining)" in caplog.text


# --- --loop/--once default (issue #79) ------------------------------------------


def test_validator_loops_by_default_and_once_opts_out():
    from validator.service import build_parser as validator_build_parser

    default_args = validator_build_parser().parse_args([])
    assert default_args.once is False  # continuous looping is the default now
    assert default_args.loop is False  # deprecated no-op, still accepted

    once_args = validator_build_parser().parse_args(["--once"])
    assert once_args.once is True

    # An existing invocation that already passes --loop must still parse without error.
    legacy_args = validator_build_parser().parse_args(["--loop"])
    assert legacy_args.once is False


def test_weight_setter_loops_by_default_and_once_opts_out():
    from weight_setter.service import build_parser as weight_setter_build_parser

    default_args = weight_setter_build_parser().parse_args([])
    assert default_args.once is False
    assert default_args.loop is False

    once_args = weight_setter_build_parser().parse_args(["--once"])
    assert once_args.once is True

    legacy_args = weight_setter_build_parser().parse_args(["--loop"])
    assert legacy_args.once is False


# --- wandb defaults (issue #102) --------------------------------------------------


def test_wandb_defaults_to_glyph_research_org_text_compression():
    from validator.service import build_parser as validator_build_parser

    default_args = validator_build_parser().parse_args([])
    assert default_args.wandb_project == "text-compression"
    assert default_args.wandb_entity == "glyph-research-org"

    override_args = validator_build_parser().parse_args(
        ["--wandb.project", "my-proj", "--wandb.entity", "my-team"]
    )
    assert override_args.wandb_project == "my-proj"
    assert override_args.wandb_entity == "my-team"


def test_main_loops_continuously_without_once_or_loop_flag(monkeypatch):
    # issue #79: this used to run exactly one round and exit unless --loop was passed, with no
    # indication anywhere that this was expected -- prove main() now keeps going by default.
    import validator.service as vs

    monkeypatch.setattr(sys, "argv", ["glyph-validator", "--netuid", "117"])
    call_count = {"n": 0}

    def fake_run_once(args, wandb_logger=None):
        call_count["n"] += 1
        if call_count["n"] >= 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(vs, "run_once", fake_run_once)
    monkeypatch.setattr(vs, "make_wandb_logger", lambda args: _FakeWandb())
    monkeypatch.setattr(vs, "_make_chain", lambda args: _FakeChain())
    monkeypatch.setattr(vs, "_sleep_with_commit_polls", lambda args, chain: None)
    monkeypatch.setattr("core.dotenv.load_dotenv", lambda: None)

    vs.main()

    assert call_count["n"] >= 3
