"""issue #104: ScoreState had no version stamp, so a change to the scoring surfaces
(eval/scoring.py's aggregation formula, BASELINE_LEVEL, the corpus source list/sampling, the
validity gates) left every already-recorded score -- including the reigning champion's own
and already-excluded losers' -- trusted forever. A SCORING_VERSION bump must drop stale
scores and clear the exclusions decided against them, so every hotkey competes fresh under
the current rules."""

from core.constants import SCORING_VERSION, SCORING_VERSION_START_BLOCKS
from core.state import (
    CommitmentState,
    ScoreState,
    ValidatorState,
    load_state,
    save_state,
    score_is_comparable,
)
from core.weights import WinnerEntry


def _score(hotkey, *, revision="rev0", version, evaluated_at_block=None):
    return ScoreState(
        hotkey=hotkey, repo="r/c", revision=revision, ratio=0.5, roundtrip_ok=True,
        throughput_bps=50_000.0, valid=True, commit_block=1, scoring_version=version,
        evaluated_at_block=evaluated_at_block,
    )


def _no_start_blocks(monkeypatch):
    """Pin the wipe-all path: the current bump has NO start-block entry (a genuine
    scoring-surface change, issue #104's original and still-default semantics)."""

    monkeypatch.setattr("core.constants.SCORING_VERSION_START_BLOCKS", {})


def test_stale_scoring_version_is_dropped_on_load(tmp_path, monkeypatch):
    _no_start_blocks(monkeypatch)
    state = ValidatorState()
    state.scores["hk-a:r/c@rev0"] = _score("hk-a", version=SCORING_VERSION - 1)
    state.scores["hk-b:r/c@rev1"] = _score("hk-b", revision="rev1", version=SCORING_VERSION)
    path = tmp_path / "state.json"
    save_state(path, state)

    reloaded = load_state(path)

    # Stale -> dropped, which alone re-admits it to the challenger filter next round
    # (c.key not in state.scores in validator.service._evaluate_round).
    assert "hk-a:r/c@rev0" not in reloaded.scores
    # Current version -> kept, still counted as "already scored."
    assert "hk-b:r/c@rev1" in reloaded.scores


def test_version_bump_clears_matching_excluded_hotkeys(tmp_path, monkeypatch):
    _no_start_blocks(monkeypatch)
    state = ValidatorState()
    state.scores["hk-loser:r/c@rev0"] = _score("hk-loser", version=SCORING_VERSION - 1)
    state.excluded_hotkeys = {"hk-loser"}
    path = tmp_path / "state.json"
    save_state(path, state)

    reloaded = load_state(path)

    assert reloaded.excluded_hotkeys == set()  # "everyone competes fresh"


def test_excluded_hotkeys_untouched_when_nothing_is_stale(tmp_path):
    state = ValidatorState()
    state.scores["hk-loser:r/c@rev0"] = _score("hk-loser", version=SCORING_VERSION)
    state.excluded_hotkeys = {"hk-loser"}
    path = tmp_path / "state.json"
    save_state(path, state)

    reloaded = load_state(path)

    assert reloaded.excluded_hotkeys == {"hk-loser"}  # nothing stale -> no reason to touch it


def test_champion_with_stale_score_becomes_eligible_as_a_challenger_again(tmp_path, monkeypatch):
    # The reigning champion's own commitment + score, both recorded under the old regime.
    _no_start_blocks(monkeypatch)
    state = ValidatorState()
    commitment = CommitmentState(
        hotkey="champ", repo="r/c", revision="rev0", block=1, artifact_hash="hash", valid=True,
    )
    state.commitments[commitment.key] = commitment
    state.scores[commitment.key] = _score("champ", revision="rev0", version=SCORING_VERSION - 1)
    state.winner_history = [WinnerEntry("champ", "r/c", "rev0", 0.5, 1)]
    path = tmp_path / "state.json"
    save_state(path, state)

    reloaded = load_state(path)

    # Matches the exact challenger filter in validator.service._evaluate_round.
    challengers = [
        c for c in reloaded.commitments.values()
        if c.valid and c.key not in reloaded.scores and c.hotkey not in reloaded.excluded_hotkeys
    ]
    assert [c.hotkey for c in challengers] == ["champ"]


def test_run_round_stamps_the_current_scoring_version(monkeypatch):
    # Regression guard for the actual production write path: reign_worker.service.run_round
    # must stamp every freshly-recorded score with the current SCORING_VERSION, or every
    # score would look "stale" forever regardless of this whole mechanism.
    from eval.evaluator import EvalOutcome
    from eval.runner import ResourceCaps
    from eval.scoring import CodecScore
    from reign_worker.service import run_round

    state = ValidatorState()
    commitment = CommitmentState(
        hotkey="hk-a", repo="a/codec", revision="rev0", block=1, artifact_hash="hash-a", valid=True,
    )
    state.commitments[commitment.key] = commitment
    monkeypatch.setattr(
        "reign_worker.service.paired_eval",
        lambda *a, **k: {
            "hk-a": EvalOutcome(
                hotkey="hk-a",
                score=CodecScore(valid=True, ratio=0.5, throughput_bps_min=99_999.0, reasons=[]),
                results=[],
            )
        },
    )

    run_round(
        state, runner=object(), challengers=[commitment], provider=object(), stream_specs=[],
        caps=ResourceCaps(), floor_bps=1.0, budget_secs=60.0, margin=0.05, block=100,
        eligible_hotkeys={"hk-a"},
    )

    assert state.scores[commitment.key].scoring_version == SCORING_VERSION


# --- score-preserving transitions via per-version start blocks (issue #143) -------------


V3_START = SCORING_VERSION_START_BLOCKS[SCORING_VERSION]  # shipped entry for the v3 bump


def test_score_compatible_bump_retains_scores_and_exclusions(tmp_path):
    # The exact live incident, inverted: v2 scores evaluated before the v3 start block plus
    # the shipped `3:` entry -> retained, exclusions retained, and challenger selection
    # yields ONLY the genuinely-unscored commitment instead of the whole board.
    state = ValidatorState()
    scored = CommitmentState(
        hotkey="hk-scored", repo="r/c", revision="rev0", block=1, artifact_hash="h1", valid=True,
    )
    fresh = CommitmentState(
        hotkey="hk-new", repo="n/c", revision="rev9", block=2, artifact_hash="h2", valid=True,
    )
    state.commitments[scored.key] = scored
    state.commitments[fresh.key] = fresh
    state.scores[scored.key] = _score(
        "hk-scored", version=SCORING_VERSION - 1, evaluated_at_block=V3_START - 100,
    )
    state.excluded_hotkeys = {"hk-defeated"}
    path = tmp_path / "state.json"
    save_state(path, state)

    reloaded = load_state(path)

    assert scored.key in reloaded.scores  # retained, still "already scored"
    assert reloaded.excluded_hotkeys == {"hk-defeated"}  # defeats really happened; kept
    challengers = [
        c for c in reloaded.commitments.values()
        if c.valid and c.key not in reloaded.scores and c.hotkey not in reloaded.excluded_hotkeys
    ]
    assert [c.hotkey for c in challengers] == ["hk-new"]  # only the new commitment


def test_old_version_score_evaluated_after_the_start_block_is_dropped(tmp_path):
    # A mixed-version validator writing v2-stamped scores after the transition began: not
    # covered by the compatibility declaration, so it drops -- but this is not a surface
    # wipe, so exclusions survive.
    state = ValidatorState()
    state.scores["hk-a:r/c@rev0"] = _score(
        "hk-a", version=SCORING_VERSION - 1, evaluated_at_block=V3_START + 10,
    )
    state.excluded_hotkeys = {"hk-defeated"}
    path = tmp_path / "state.json"
    save_state(path, state)

    reloaded = load_state(path)

    assert reloaded.scores == {}
    assert reloaded.excluded_hotkeys == {"hk-defeated"}


def test_compatibility_is_chained_not_single_step(tmp_path):
    # A v1 score predates the surface-changing v1->v2 overhaul; the v3 entry alone must not
    # let it ride through -- every step of the transition has to be declared compatible.
    state = ValidatorState()
    state.scores["hk-a:r/c@rev0"] = _score("hk-a", version=1, evaluated_at_block=V3_START - 100)
    state.excluded_hotkeys = {"hk-defeated"}
    path = tmp_path / "state.json"
    save_state(path, state)

    reloaded = load_state(path)

    assert reloaded.scores == {}
    assert reloaded.excluded_hotkeys == set()  # genuine surface wipe: board competes fresh


def test_score_is_comparable_matches_the_retention_rule():
    assert score_is_comparable(_score("hk", version=SCORING_VERSION))
    assert score_is_comparable(_score("hk", version=SCORING_VERSION - 1, evaluated_at_block=V3_START - 1))
    assert not score_is_comparable(_score("hk", version=SCORING_VERSION - 1, evaluated_at_block=V3_START))
    assert not score_is_comparable(_score("hk", version=1, evaluated_at_block=V3_START - 1))
    assert not score_is_comparable(_score("hk", version=SCORING_VERSION + 1))  # future/unknown


def test_vacant_crown_recovery_accepts_a_retained_compatible_score(monkeypatch):
    # issue #135's recovery must treat a retained v2 score as current (its ratio means the
    # same thing) -- otherwise a post-transition vacant crown burns despite a perfectly
    # comparable persisted score.
    from validator.service import _recover_vacant_crown

    state = ValidatorState()
    commitment = CommitmentState(
        hotkey="champ", repo="r/c", revision="rev0", block=1, artifact_hash="h", valid=True,
    )
    state.commitments[commitment.key] = commitment
    state.scores[commitment.key] = _score(
        "champ", version=SCORING_VERSION - 1, evaluated_at_block=V3_START - 100,
    )

    _recover_vacant_crown(state)

    assert [w.hotkey for w in state.winner_history] == ["champ"]
