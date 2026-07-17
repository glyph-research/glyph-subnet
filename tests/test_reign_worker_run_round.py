"""issue #67: when the incumbent fails its own re-evaluation, the crown must actually be
vacated -- rolling_weights_for_hotkeys only looks at winner_history's presence, never
re-checks state.scores[...].valid, so a stale entry would keep earning weight indefinitely
whenever no challenger happens to appear in a given round."""

from bittensor.utils.btlogging import logging as bt_logging

from eval.evaluator import EvalOutcome
from eval.runner import ResourceCaps
from eval.scoring import CodecScore
from core.state import CommitmentState, ValidatorState
from core.weights import WinnerEntry
from reign_worker.service import run_round

CAPS = ResourceCaps()


def _commitment(hotkey, repo, revision, block):
    return CommitmentState(
        hotkey=hotkey, repo=repo, revision=revision, block=block, artifact_hash=f"hash-{hotkey}", valid=True
    )


def _outcome(hotkey, *, valid, ratio, error=None):
    return EvalOutcome(
        hotkey=hotkey,
        score=CodecScore(valid=valid, ratio=ratio, throughput_bps_min=99_999.0, reasons=[] if valid else ["broke"]),
        results=[],
        error=error,
    )


def test_incumbent_reeval_failure_vacates_crown_when_no_challenger(monkeypatch):
    state = ValidatorState()
    incumbent_commitment = _commitment("incumbent", "inc/codec", "rev123456", block=1)
    state.commitments[incumbent_commitment.key] = incumbent_commitment
    state.winner_history = [
        WinnerEntry(hotkey="incumbent", repo="inc/codec", revision="rev123456", ratio=0.5, commit_block=1)
    ]
    monkeypatch.setattr(
        "reign_worker.service.paired_eval",
        lambda *a, **k: {"incumbent": _outcome("incumbent", valid=False, ratio=999.0)},
    )

    run_round(
        state, runner=object(), challengers=[], provider=object(), stream_specs=[],
        caps=CAPS, floor_bps=1.0, budget_secs=60.0, margin=0.05, block=100,
        eligible_hotkeys={"incumbent"},
    )

    assert state.winner_history == []  # vacated, not left stale
    assert state.scores[incumbent_commitment.key].valid is False


def test_incumbent_reeval_failure_lets_challenger_take_vacant_crown(monkeypatch):
    # Regression guard: the existing same-round vacant-crown promotion must not break.
    state = ValidatorState()
    incumbent_commitment = _commitment("incumbent", "inc/codec", "rev123456", block=1)
    challenger_commitment = _commitment("challenger", "chal/codec", "rev654321", block=2)
    state.commitments[incumbent_commitment.key] = incumbent_commitment
    state.commitments[challenger_commitment.key] = challenger_commitment
    state.winner_history = [
        WinnerEntry(hotkey="incumbent", repo="inc/codec", revision="rev123456", ratio=0.5, commit_block=1)
    ]
    monkeypatch.setattr(
        "reign_worker.service.paired_eval",
        lambda *a, **k: {
            "incumbent": _outcome("incumbent", valid=False, ratio=999.0),
            "challenger": _outcome("challenger", valid=True, ratio=0.3),
        },
    )

    run_round(
        state, runner=object(), challengers=[challenger_commitment], provider=object(), stream_specs=[],
        caps=CAPS, floor_bps=1.0, budget_secs=60.0, margin=0.05, block=100,
        eligible_hotkeys={"incumbent", "challenger"},
    )

    assert [w.hotkey for w in state.winner_history] == ["challenger"]


def test_incumbent_reeval_failure_promotes_hot_standby_to_index_zero(monkeypatch):
    # A two-entry history: the standby (index 1) must naturally become the effective
    # incumbent for the NEXT round once the current one (index 0) is popped -- no
    # special-casing needed here, per the code's own "promotes later" comment.
    state = ValidatorState()
    incumbent_commitment = _commitment("incumbent", "inc/codec", "rev123456", block=1)
    state.commitments[incumbent_commitment.key] = incumbent_commitment
    state.winner_history = [
        WinnerEntry(hotkey="incumbent", repo="inc/codec", revision="rev123456", ratio=0.5, commit_block=1),
        WinnerEntry(hotkey="standby", repo="standby/codec", revision="rev999999", ratio=0.6, commit_block=0),
    ]
    monkeypatch.setattr(
        "reign_worker.service.paired_eval",
        lambda *a, **k: {"incumbent": _outcome("incumbent", valid=False, ratio=999.0)},
    )

    run_round(
        state, runner=object(), challengers=[], provider=object(), stream_specs=[],
        caps=CAPS, floor_bps=1.0, budget_secs=60.0, margin=0.05, block=100,
        eligible_hotkeys={"incumbent", "standby"},
    )

    assert [w.hotkey for w in state.winner_history] == ["standby"]


def test_run_round_logs_every_candidate_result_not_only_the_winner(monkeypatch, caplog):
    # issue #81: a challenger that loses (beats baseline but not the incumbent's margin, or
    # fails validation) must have its actual ratio/reason printed -- not silently excluded
    # with nothing in the log beyond the eventual "new winner" line.
    bt_logging.set_info()
    state = ValidatorState()
    incumbent_commitment = _commitment("incumbent", "inc/codec", "rev123456", block=1)
    winner_commitment = _commitment("winner", "win/codec", "rev654321", block=2)
    loser_commitment = _commitment("loser", "lose/codec", "rev777777", block=3)
    for c in (incumbent_commitment, winner_commitment, loser_commitment):
        state.commitments[c.key] = c
    state.winner_history = [
        WinnerEntry(hotkey="incumbent", repo="inc/codec", revision="rev123456", ratio=0.5, commit_block=1)
    ]
    monkeypatch.setattr(
        "reign_worker.service.paired_eval",
        lambda *a, **k: {
            "incumbent": _outcome("incumbent", valid=True, ratio=0.5),
            "winner": _outcome("winner", valid=True, ratio=0.1),
            "loser": _outcome("loser", valid=False, ratio=999.0),
        },
    )

    run_round(
        state, runner=object(), challengers=[winner_commitment, loser_commitment], provider=object(),
        stream_specs=[], caps=CAPS, floor_bps=1.0, budget_secs=60.0, margin=0.05, block=100,
        eligible_hotkeys={"incumbent", "winner", "loser"},
    )

    out = caplog.text
    assert "candidate incumbent: ratio=0.5000 valid" in out
    assert "candidate winner: ratio=0.1000 valid" in out
    assert "candidate loser: invalid" in out
    assert "broke" in out  # the actual reason, not just "invalid"


def test_invalid_candidate_summary_includes_the_runner_error_when_the_codec_never_ran(monkeypatch, caplog):
    # issue #127 (observed live): a docker pull failure was summarized as "round-trip failed
    # on streams: [...]", implying the codec produced wrong output when it never ran at all.
    # The summary line -- where the candidate's fate is decided -- must carry the real error,
    # not force the operator to scroll back to an earlier per-stream warning.
    bt_logging.set_info()
    state = ValidatorState()
    pull_denied = _commitment("pulldenied", "gone/codec", "rev999999", block=3)
    state.commitments[pull_denied.key] = pull_denied
    monkeypatch.setattr(
        "reign_worker.service.paired_eval",
        lambda *a, **k: {
            "pulldenied": _outcome(
                "pulldenied", valid=False, ratio=999.0,
                error="docker: pull access denied for gone/codec, repository does not exist",
            ),
        },
    )

    run_round(
        state, runner=object(), challengers=[pull_denied], provider=object(), stream_specs=[],
        caps=CAPS, floor_bps=1.0, budget_secs=60.0, margin=0.05, block=100,
        eligible_hotkeys={"pulldenied"},
    )

    out = caplog.text
    assert "candidate pulldenied: invalid" in out
    assert "runner error: docker: pull access denied" in out


def test_incumbent_reeval_failure_warning_includes_the_runner_error(monkeypatch, caplog):
    bt_logging.set_info()
    state = ValidatorState()
    incumbent_commitment = _commitment("incumbent", "inc/codec", "rev123456", block=1)
    state.commitments[incumbent_commitment.key] = incumbent_commitment
    state.winner_history = [
        WinnerEntry(hotkey="incumbent", repo="inc/codec", revision="rev123456", ratio=0.5, commit_block=1)
    ]
    monkeypatch.setattr(
        "reign_worker.service.paired_eval",
        lambda *a, **k: {
            "incumbent": _outcome("incumbent", valid=False, ratio=999.0, error="docker daemon unreachable"),
        },
    )

    run_round(
        state, runner=object(), challengers=[], provider=object(), stream_specs=[],
        caps=CAPS, floor_bps=1.0, budget_secs=60.0, margin=0.05, block=100,
        eligible_hotkeys={"incumbent"},
    )

    assert "failed re-eval" in caplog.text
    assert "runner error: docker daemon unreachable" in caplog.text


def test_unevaluated_incumbent_still_defends_with_its_last_recorded_ratio(monkeypatch):
    # issue #135: the incumbent's commitment is invalid this round (transiently unreachable
    # repo), so it can't be re-evaluated -- but the crown is not vacant. A challenger must
    # still beat the incumbent's last recorded ratio by the full margin.
    state = ValidatorState()
    challenger = _commitment("challenger", "chal/codec", "rev654321", block=2)
    state.commitments[challenger.key] = challenger
    unreachable_incumbent = CommitmentState(
        hotkey="incumbent", repo="inc/codec", revision="rev123456", block=1,
        artifact_hash="inc-hash", valid=False, transiently_unreachable=True,
    )
    state.commitments[unreachable_incumbent.key] = unreachable_incumbent
    state.winner_history = [
        WinnerEntry(hotkey="incumbent", repo="inc/codec", revision="rev123456", ratio=0.5, commit_block=1)
    ]
    # 0.49 is better than 0.5 but NOT by the 5% margin (needs <= 0.475).
    monkeypatch.setattr(
        "reign_worker.service.paired_eval",
        lambda *a, **k: {"challenger": _outcome("challenger", valid=True, ratio=0.49)},
    )

    run_round(
        state, runner=object(), challengers=[challenger], provider=object(), stream_specs=[],
        caps=CAPS, floor_bps=1.0, budget_secs=60.0, margin=0.05, block=100,
        eligible_hotkeys={"incumbent", "challenger"},
    )

    assert [w.hotkey for w in state.winner_history] == ["incumbent"]  # crown defended


def test_unevaluated_incumbent_is_dethroned_by_a_full_margin_beat(monkeypatch):
    # Same setup, but the challenger genuinely clears the epsilon against the last recorded
    # ratio -- promotion proceeds and the unreachable incumbent rolls to the previous slot.
    state = ValidatorState()
    challenger = _commitment("challenger", "chal/codec", "rev654321", block=2)
    state.commitments[challenger.key] = challenger
    unreachable_incumbent = CommitmentState(
        hotkey="incumbent", repo="inc/codec", revision="rev123456", block=1,
        artifact_hash="inc-hash", valid=False, transiently_unreachable=True,
    )
    state.commitments[unreachable_incumbent.key] = unreachable_incumbent
    state.winner_history = [
        WinnerEntry(hotkey="incumbent", repo="inc/codec", revision="rev123456", ratio=0.5, commit_block=1)
    ]
    monkeypatch.setattr(
        "reign_worker.service.paired_eval",
        lambda *a, **k: {"challenger": _outcome("challenger", valid=True, ratio=0.47)},
    )

    run_round(
        state, runner=object(), challengers=[challenger], provider=object(), stream_specs=[],
        caps=CAPS, floor_bps=1.0, budget_secs=60.0, margin=0.05, block=100,
        eligible_hotkeys={"incumbent", "challenger"},
    )

    assert [w.hotkey for w in state.winner_history] == ["challenger", "incumbent"]
