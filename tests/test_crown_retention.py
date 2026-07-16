"""issue #135 (live incident 2026-07-16): a ~2-minute HF outage 504'd the champion's
precheck, the eligibility-based history compaction permanently dethroned it, and every
round since burned 100%. The crown must survive transient unreachability and be droppable
only by a genuine re-eval failure, a confirmed-permanent 404 (the #128 streak), or a
content disqualification -- and a vacant crown must be able to refill from persisted
scores instead of burning forever."""

from bittensor.utils.btlogging import logging as bt_logging

from core.commitments import CodecCommitment, serialize_commitment
from core.constants import REPO_NOT_FOUND_EXCLUDE_STREAK, SCORING_VERSION
from core.state import ScoreState, ValidatorState
from core.weights import WinnerEntry, compute_weights
from validation.precheck import PrecheckResult
from validator.service import _evaluate_round

_REV_A = "a" * 40
_CHAMP = "champ-hk"
_KEY = f"{_CHAMP}:champ/codec@{_REV_A}"

_OK = PrecheckResult(
    repo="champ/codec", revision=_REV_A, ok=True, artifact_hash="champ-hash", artifact_bytes=10,
)
_TRANSIENT = PrecheckResult(
    repo="champ/codec", revision=_REV_A, ok=False,
    errors=["repo unavailable: Server error '504 Gateway Timeout'"],
    repo_not_found=False, repo_unreachable=True,
)
_GONE = PrecheckResult(
    repo="champ/codec", revision=_REV_A, ok=False,
    errors=["repo unavailable: 404 Client Error ... Repository Not Found"],
    repo_not_found=True, repo_unreachable=True,
)


class _FakeChain:
    def __init__(self, raw_commitments: dict, block: int = 8632806):
        self._raw_commitments = raw_commitments
        self._block = block

    def current_block(self) -> int:
        return self._block

    def block_hash(self, block: int) -> str:
        return "0xbeacon"

    def get_all_commitments(self) -> dict:
        return self._raw_commitments

    def metagraph(self):
        return type("Metagraph", (), {"hotkeys": ["uid0-burn", _CHAMP], "uids": [0, 1]})()


def _args():
    return type(
        "Args", (),
        {"window_anchor": None, "max_artifact_bytes": 10_000, "corpus_dir": None},
    )()


def _champion_state() -> ValidatorState:
    """A validator that already crowned the champion and persisted its score -- so the
    champion is never re-selected as a challenger (one-shot scoring)."""

    state = ValidatorState()
    state.winner_history = [
        WinnerEntry(hotkey=_CHAMP, repo="champ/codec", revision=_REV_A, ratio=0.0763, commit_block=100)
    ]
    state.scores[_KEY] = ScoreState(
        hotkey=_CHAMP, repo="champ/codec", revision=_REV_A, ratio=0.0763, roundtrip_ok=True,
        throughput_bps=50_000.0, valid=True, commit_block=100, scoring_version=SCORING_VERSION,
    )
    return state


def _scripted_precheck(monkeypatch, results):
    def fake_precheck(repo, revision, *, max_artifact_bytes, download=True):
        return results.pop(0)

    monkeypatch.setattr("validator.service.precheck_codec", fake_precheck)


def _chain() -> _FakeChain:
    return _FakeChain({_CHAMP: serialize_commitment(CodecCommitment(repo="champ/codec", rev=_REV_A))})


def test_transient_precheck_failure_leaves_crown_and_weights_unchanged(monkeypatch, caplog):
    # The exact live sequence: champion prechecks valid, next round HF 504s on everything,
    # 0 challengers either round. Before the fix the second round emitted champion=none and
    # weights=[(0, 1.0)] -- permanently.
    bt_logging.set_info()
    state = _champion_state()
    _scripted_precheck(monkeypatch, [_OK, _TRANSIENT])

    _evaluate_round(_args(), state, _chain(), "salt")
    history_after_healthy_round = list(state.winner_history)
    weights_before = compute_weights(["uid0-burn", _CHAMP], state.winner_history, is_burn_tempo=False)

    _evaluate_round(_args(), state, _chain(), "salt")

    assert state.winner_history == history_after_healthy_round  # crown kept
    assert state.commitments[_KEY].transiently_unreachable is True
    weights_after = compute_weights(["uid0-burn", _CHAMP], state.winner_history, is_burn_tempo=False)
    assert weights_after == weights_before  # no 100% burn
    assert weights_after[1] > 0  # champion still earning


def test_definitive_404_dethrones_only_after_the_confirmation_streak(monkeypatch):
    state = _champion_state()
    _scripted_precheck(monkeypatch, [_GONE] * REPO_NOT_FOUND_EXCLUDE_STREAK)

    for _ in range(REPO_NOT_FOUND_EXCLUDE_STREAK - 1):
        _evaluate_round(_args(), state, _chain(), "salt")
        assert [w.hotkey for w in state.winner_history] == [_CHAMP]  # not yet confirmed gone

    _evaluate_round(_args(), state, _chain(), "salt")

    assert _CHAMP in state.excluded_hotkeys  # #128 streak crossed
    assert state.winner_history == []  # only now is the dethronement real
    weights = compute_weights(["uid0-burn", _CHAMP], state.winner_history, is_burn_tempo=False)
    assert weights == [1.0, 0.0]  # legitimate burn once definitively gone


def test_vacant_crown_refills_from_the_best_persisted_score(monkeypatch, caplog):
    # The recovery gap: history already wiped (e.g. by the pre-fix bug, or a legitimate pop
    # whose cause later healed), champion's score persisted, repo reachable again. One-shot
    # challenger selection would never re-evaluate it -- the crown must refill directly.
    bt_logging.set_info()
    state = _champion_state()
    state.winner_history = []
    _scripted_precheck(monkeypatch, [_OK])

    _evaluate_round(_args(), state, _chain(), "salt")

    assert [w.hotkey for w in state.winner_history] == [_CHAMP]
    assert state.winner_history[0].ratio == 0.0763
    assert "vacant crown: re-promoting" in caplog.text


def test_vacant_crown_never_refills_from_excluded_or_stale_scores(monkeypatch):
    state = _champion_state()
    state.winner_history = []

    # Excluded hotkey (one-shot loser / confirmed 404): never re-crowned.
    state.excluded_hotkeys.add(_CHAMP)
    _scripted_precheck(monkeypatch, [_OK])
    _evaluate_round(_args(), state, _chain(), "salt")
    assert state.winner_history == []

    # Stale scoring version: not comparable, never re-crowned from it.
    state.excluded_hotkeys.clear()
    state.scores[_KEY].scoring_version = SCORING_VERSION - 1
    _scripted_precheck(monkeypatch, [_OK])
    _evaluate_round(_args(), state, _chain(), "salt")
    assert state.winner_history == []


def test_vacant_crown_does_not_refill_while_the_repo_is_still_unreachable(monkeypatch):
    state = _champion_state()
    state.winner_history = []
    _scripted_precheck(monkeypatch, [_TRANSIENT])

    _evaluate_round(_args(), state, _chain(), "salt")

    # Commitment invalid this round -> not a recovery candidate yet; refills once it heals.
    assert state.winner_history == []
    _scripted_precheck(monkeypatch, [_OK])
    _evaluate_round(_args(), state, _chain(), "salt")
    assert [w.hotkey for w in state.winner_history] == [_CHAMP]
