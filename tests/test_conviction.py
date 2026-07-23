"""issue #141: Miner Conviction -- winners must keep earnings staked to receive incentive.

Covers the lock formula's boundaries, deterministic ledger accounting (gap backfill ==
uninterrupted live tracking), gating semantics (pay the two most recent compliant winners
in the retained history, burn only when none qualifies -- issue #170; reversible; crown
untouched), activation gating, persistence, and observability.
"""

from core.constants import (
    CONVICTION_ACTIVATION_BLOCK,
    CONVICTION_LOCK_CHECK_START_BLOCK,
    CONVICTION_TRACKING_START_BLOCK,
    SCORING_VERSION,
)
from core.conviction import (
    ConvictionLedger,
    conviction_report,
    is_compliant,
    ledger_catchup,
    ledger_grid,
    required_conviction,
)
from core.state import ValidatorState, load_state, save_state
from core.wandb_logger import build_weights_metrics
from core.weights import WinnerEntry, compute_weights
from weight_setter.service import decide_weights

START = CONVICTION_TRACKING_START_BLOCK
TEMPO = 360


# --- required_conviction boundaries (the 1000 / 5000 regimes) --------------------------------


def test_required_conviction_regimes_and_exact_boundaries():
    assert required_conviction(0) == 0.0
    assert required_conviction(999) == 0.0  # young reign: everything liquid
    assert required_conviction(1000) == 0.0  # exact boundary: still nothing locked
    assert required_conviction(1001) == 1.0  # flat 1000-free plateau begins
    assert required_conviction(5000) == 4000.0  # regimes meet exactly: 0.8*5000 == 5000-1000
    assert required_conviction(10000) == 8000.0  # 20% allowance: 2000 free, growing with reign
    assert required_conviction(20000) == 16000.0
    assert required_conviction(100000) == 80000.0


def test_free_fraction_is_the_owner_set_80_20_split():
    # Issue #158: owner reduced the lock requirement from 90% to 80% of earned. Pinned so
    # a silent revert of CONVICTION_FREE_FRACTION fails loudly, not just numerically.
    from core.constants import CONVICTION_FREE_FRACTION

    assert CONVICTION_FREE_FRACTION == 0.20


def test_is_compliant_is_alpha_vs_alpha():
    assert is_compliant(earned=999, staked=0)  # nothing required yet
    assert is_compliant(earned=5000, staked=4000)  # exactly at the line
    assert not is_compliant(earned=5000, staked=3999.99)


# --- ledger: fixed grid, deterministic backfill ------------------------------------------


def _payouts(tempos, per_tempo=None):
    per_tempo = per_tempo or {"champ": 100.0, "prev": 40.0}
    return {START + TEMPO * k: dict(per_tempo) for k in range(1, tempos + 1)}


def _emissions_at(payouts):
    return lambda block: payouts.get(block, {})


def test_ledger_grid_is_anchored_at_the_protocol_constant():
    # Fresh ledger: first sample is one tempo past the tracking start, never "now".
    assert ledger_grid(0, START + 3 * TEMPO, TEMPO) == [START + TEMPO, START + 2 * TEMPO, START + 3 * TEMPO]
    # Nothing to do before the start block or between grid points.
    assert ledger_grid(0, START, TEMPO) == []
    assert ledger_grid(START + TEMPO, START + TEMPO + 100, TEMPO) == []
    # A mid-stride last_block still resumes on the same grid, not a shifted one.
    assert ledger_grid(START + 500, START + 3 * TEMPO, TEMPO) == [START + 2 * TEMPO, START + 3 * TEMPO]


def test_gap_backfill_produces_the_identical_ledger_to_uninterrupted_tracking():
    payouts = _payouts(10)

    live = ConvictionLedger()
    for k in range(1, 11):  # one catchup per tempo, like an always-up validator
        ledger_catchup(live, current_block=START + TEMPO * k, tempo=TEMPO, emissions_at=_emissions_at(payouts))

    gapped = ConvictionLedger()  # validator down the whole time, then one backfill
    ledger_catchup(gapped, current_block=START + TEMPO * 10, tempo=TEMPO, emissions_at=_emissions_at(payouts))

    assert live == gapped
    assert gapped.earned == {"champ": 1000.0, "prev": 400.0}
    assert gapped.last_block == START + TEMPO * 10


def test_ledger_catchup_failure_resumes_exactly_where_it_stopped():
    payouts = _payouts(4)
    calls = {"n": 0}

    def flaky(block):
        calls["n"] += 1
        if calls["n"] == 3:  # third grid block: archive hiccup
            raise ConnectionError("archive unavailable")
        return payouts.get(block, {})

    ledger = ConvictionLedger()
    try:
        ledger_catchup(ledger, current_block=START + TEMPO * 4, tempo=TEMPO, emissions_at=flaky)
    except ConnectionError:
        pass
    assert ledger.last_block == START + TEMPO * 2  # only fully-applied blocks recorded

    ledger_catchup(ledger, current_block=START + TEMPO * 4, tempo=TEMPO, emissions_at=_emissions_at(payouts))
    assert ledger.earned == {"champ": 400.0, "prev": 160.0}  # nothing double-counted


def test_ledger_totals_are_permanent_per_hotkey():
    # Dethrone-and-return never resets: earnings keep accumulating on the same key.
    payouts = {START + TEMPO: {"champ": 100.0}, START + 2 * TEMPO: {"other": 50.0}, START + 3 * TEMPO: {"champ": 10.0}}
    ledger = ConvictionLedger()
    ledger_catchup(ledger, current_block=START + 3 * TEMPO, tempo=TEMPO, emissions_at=_emissions_at(payouts))
    assert ledger.earned["champ"] == 110.0


def test_ledger_survives_a_state_round_trip(tmp_path):
    state = ValidatorState()
    state.conviction_ledger.earned["champ"] = 1234.5
    state.conviction_ledger.last_block = START + TEMPO
    path = tmp_path / "state.json"
    save_state(path, state)
    reloaded = load_state(path)
    assert reloaded.conviction_ledger == state.conviction_ledger


# --- report + activation gate ------------------------------------------------------------


def test_report_gates_only_after_the_activation_block():
    ledger = ConvictionLedger(earned={"champ": 5000.0})
    staked = {"champ": 100.0}  # far below the 4000 required

    before = conviction_report(ledger, ["champ"], staked, block=CONVICTION_ACTIVATION_BLOCK - 1)
    assert before["champ"]["compliant"] is True  # tracking only, no gating yet
    assert before["champ"]["required_conviction"] == 4000.0  # ledger already warm and reported

    after = conviction_report(ledger, ["champ"], staked, block=CONVICTION_ACTIVATION_BLOCK)
    assert after["champ"]["compliant"] is False


def test_report_covers_both_winner_slots_independently():
    ledger = ConvictionLedger(earned={"champ": 20000.0, "prev": 20000.0})
    staked = {"champ": 16000.0, "prev": 15999.0}
    report = conviction_report(ledger, ["champ", "prev"], staked, block=CONVICTION_ACTIVATION_BLOCK)
    assert report["champ"]["compliant"] is True
    assert report["prev"]["compliant"] is False


# --- weights: pay the two most recent compliant winners, burn only if none qualify -------


HOTKEYS = ["burn-sink", "champ", "prev", "bystander"]
HISTORY = [
    WinnerEntry(hotkey="champ", repo="c/r", revision="rev1", ratio=0.4, commit_block=10),
    WinnerEntry(hotkey="prev", repo="p/r", revision="rev2", ratio=0.5, commit_block=5),
]


def test_two_slot_history_pays_the_compliant_ones_and_burns_only_when_none_qualify():
    # With exactly two retained entries the ladder degenerates to the #166 table.
    ungated = compute_weights(HOTKEYS, HISTORY, is_burn_tempo=False)
    assert ungated == [0.0, 0.7, 0.3, 0.0]

    prev_gated = compute_weights(HOTKEYS, HISTORY, is_burn_tempo=False, gated_hotkeys={"prev"})
    assert prev_gated == [0.0, 1.0, 0.0, 0.0]  # champ is the only qualifier -> whole pot

    champ_gated = compute_weights(HOTKEYS, HISTORY, is_burn_tempo=False, gated_hotkeys={"champ"})
    assert champ_gated == [0.0, 0.0, 1.0, 0.0]  # ladder shifts up: prev takes 1.0

    both = compute_weights(HOTKEYS, HISTORY, is_burn_tempo=False, gated_hotkeys={"champ", "prev"})
    assert both == [1.0, 0.0, 0.0, 0.0]  # nothing retained qualifies -> burn


LADDER_HOTKEYS = ["burn-sink", "w0", "w1", "w2", "w3", "bystander"]
LADDER = [
    WinnerEntry(hotkey=f"w{i}", repo=f"r{i}/c", revision=f"rev{i}", ratio=0.4 + i / 100, commit_block=100 - i)
    for i in range(4)
]


def test_the_ladder_walks_past_gated_winners_to_the_next_compliant_pair():
    # issue #170: a gated current winner shifts the WHOLE ladder up rather than handing
    # its share to the slot beside it.
    assert compute_weights(LADDER_HOTKEYS, LADDER, is_burn_tempo=False) == [0.0, 0.7, 0.3, 0.0, 0.0, 0.0]

    top_gated = compute_weights(LADDER_HOTKEYS, LADDER, is_burn_tempo=False, gated_hotkeys={"w0"})
    assert top_gated == [0.0, 0.0, 0.7, 0.3, 0.0, 0.0]  # w1 -> 0.7, w2 -> 0.3

    two_gated = compute_weights(LADDER_HOTKEYS, LADDER, is_burn_tempo=False, gated_hotkeys={"w0", "w1"})
    assert two_gated == [0.0, 0.0, 0.0, 0.7, 0.3, 0.0]  # w2 -> 0.7, w3 -> 0.3

    # A gap in the middle is skipped, not collapsed onto one payee.
    middle_gated = compute_weights(LADDER_HOTKEYS, LADDER, is_burn_tempo=False, gated_hotkeys={"w1"})
    assert middle_gated == [0.0, 0.7, 0.0, 0.3, 0.0, 0.0]

    all_gated = compute_weights(
        LADDER_HOTKEYS, LADDER, is_burn_tempo=False, gated_hotkeys={"w0", "w1", "w2", "w3"}
    )
    assert all_gated == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # only now does the pot burn


def test_deregistered_fallback_entries_are_never_paid():
    # Eligibility does real work here: a retained winner absent from the metagraph (or
    # excluded) must be skipped exactly like a gated one, never selected as a payee.
    without_w1 = ["burn-sink", "w0", "w2", "w3", "bystander"]
    weights = compute_weights(without_w1, LADDER, is_burn_tempo=False, gated_hotkeys={"w0"})
    assert weights == [0.0, 0.0, 0.7, 0.3, 0.0]  # w1 gone -> w2/w3 paid


def test_every_payout_row_sums_to_exactly_one():
    rows = [
        compute_weights(LADDER_HOTKEYS, LADDER, is_burn_tempo=False),
        compute_weights(LADDER_HOTKEYS, LADDER, is_burn_tempo=False, gated_hotkeys={"w0"}),
        compute_weights(LADDER_HOTKEYS, LADDER, is_burn_tempo=False, gated_hotkeys={"w0", "w1", "w2"}),
        compute_weights(LADDER_HOTKEYS, LADDER, is_burn_tempo=True),
        compute_weights(LADDER_HOTKEYS, [], is_burn_tempo=False),
    ]
    for row in rows:
        assert sum(row) == 1.0  # exact, not approx: validators must agree bit-for-bit


def test_single_retained_winner_takes_the_pot_or_burns():
    solo = [HISTORY[0]]
    compliant = compute_weights(HOTKEYS, solo, is_burn_tempo=False, gated_hotkeys=set())
    assert compliant == [0.0, 1.0, 0.0, 0.0]
    gated = compute_weights(HOTKEYS, solo, is_burn_tempo=False, gated_hotkeys={"champ"})
    assert gated == [1.0, 0.0, 0.0, 0.0]  # nothing further down the ladder -> burn


def test_a_dethroned_but_locked_winner_keeps_earning():
    # The intended incentive (issue #170): staying compliant keeps paying you after you
    # lose the crown -- w3 sits at the bottom of the ladder and is paid the moment the
    # more recent winners lapse.
    idle = compute_weights(LADDER_HOTKEYS, LADDER, is_burn_tempo=False)
    assert idle[4] == 0.0  # w3 earns nothing while the top of the ladder is compliant
    lapsed = compute_weights(
        LADDER_HOTKEYS, LADDER, is_burn_tempo=False, gated_hotkeys={"w0", "w1", "w2"}
    )
    assert lapsed[4] == 1.0  # ... and takes the whole pot once they all lapse


def test_gating_is_reversible_and_never_touches_the_crown():
    history = list(HISTORY)
    gated_round, _ = decide_weights(
        HOTKEYS, history, block=CONVICTION_ACTIVATION_BLOCK, tempo=TEMPO,
        last_round_outputs=[], anchor=0, burn_uid=0, force_burn=False, gated_hotkeys={"champ"},
    )
    restaked_round, _ = decide_weights(
        HOTKEYS, history, block=CONVICTION_ACTIVATION_BLOCK, tempo=TEMPO,
        last_round_outputs=[], anchor=0, burn_uid=0, force_burn=False, gated_hotkeys=set(),
    )
    assert history == HISTORY  # winner_history byte-identical throughout
    if gated_round[0] < 1.0:  # not a scheduled burn tempo: champ's share visibly moved
        assert gated_round[1] == 0.0
        assert restaked_round[1] > 0.0  # incentive restored at the next weight-setting


def test_burn_tempo_and_no_winner_paths_are_unaffected_by_gating():
    burn = compute_weights(HOTKEYS, HISTORY, is_burn_tempo=True, gated_hotkeys={"champ"})
    assert burn == [1.0, 0.0, 0.0, 0.0]
    no_winners = compute_weights(HOTKEYS, [], is_burn_tempo=False, gated_hotkeys={"champ"})
    assert no_winners == [1.0, 0.0, 0.0, 0.0]


# --- observability -----------------------------------------------------------------------


def test_build_weights_metrics_flattens_the_conviction_report():
    conviction = {
        "champ": {"earned": 5000.0, "staked": 100.0, "required_conviction": 4000.0, "compliant": False},
    }
    metrics = build_weights_metrics(
        block=1, tempo=TEMPO, is_burn_tempo=False, uids=[0, 1], weights=[1.0, 0.0],
        conviction=conviction,
    )
    assert metrics["conviction/champ/earned"] == 5000.0
    assert metrics["conviction/champ/staked"] == 100.0
    assert metrics["conviction/champ/required_conviction"] == 4000.0
    assert metrics["conviction/champ/compliant"] is False
    # And without a report the metric shape is exactly the pre-#141 one.
    bare = build_weights_metrics(block=1, tempo=TEMPO, is_burn_tempo=False, uids=[0], weights=[1.0])
    assert all(not k.startswith("conviction/") for k in bare)


# --- run_once wiring: report computed from ledger + metagraph stake, gate passed through --


def test_run_once_gates_a_noncompliant_winner_after_activation(monkeypatch, tmp_path):
    from core.state import ValidatorState as VS
    from validator.service import run_once

    state = VS()
    state.winner_history = [WinnerEntry(hotkey="champ", repo="c/r", revision="rev1", ratio=0.4, commit_block=10)]
    state.conviction_ledger.earned["champ"] = 5000.0
    state.conviction_ledger.last_block = CONVICTION_ACTIVATION_BLOCK  # ledger already caught up

    class _Chain:
        def commit_reveal_enabled(self):
            return True

        def tempo(self):
            return TEMPO

        def metagraph(self):
            return type("Metagraph", (), {"hotkeys": ["hk0", "champ"], "uids": [0, 1], "S": [0.0, 100.0]})()

        def get_all_commitments(self):
            return {}

    chain = _Chain()
    monkeypatch.setattr("validator.service.load_state", lambda path: state)
    monkeypatch.setattr("validator.service.save_state", lambda path, s: None)
    monkeypatch.setattr("validator.service._make_chain", lambda args: chain)
    monkeypatch.setattr("validator.service._local_version_key", lambda: 1)
    monkeypatch.setattr("validator.service._assert_version_key_matches", lambda chain: 1)
    monkeypatch.setattr(
        "validator.service._evaluate_round",
        lambda args, state, chain, salt: (CONVICTION_ACTIVATION_BLOCK, None, {}, chain.metagraph()),
    )

    captured = {}

    def fake_decide_weights(*a, **kwargs):
        captured["gated"] = kwargs.get("gated_hotkeys")
        return [1.0, 0.0], False

    monkeypatch.setattr("validator.service.decide_weights", fake_decide_weights)

    class _FakeWandb:
        def log(self, *a, **k):
            pass

        def finish(self):
            pass

    args = type(
        "Args", (),
        {"state_dir": str(tmp_path), "salt_file": None, "dry_run": True, "burn_uid": 0, "window_anchor": 0},
    )()
    run_once(args, wandb_logger=_FakeWandb())

    # earned 5000 -> required 4000; staked 100 -> gated (block == activation block).
    assert captured["gated"] == {"champ"}
    assert state.scores == {} and state.winner_history[0].hotkey == "champ"  # crown untouched


def test_run_once_does_not_gate_before_activation(monkeypatch, tmp_path):
    # Same non-compliant numbers, but one block before activation: track only, no gate.
    from core.state import ValidatorState as VS
    from validator.service import run_once

    state = VS()
    state.winner_history = [WinnerEntry(hotkey="champ", repo="c/r", revision="rev1", ratio=0.4, commit_block=10)]
    state.conviction_ledger.earned["champ"] = 5000.0
    state.conviction_ledger.last_block = CONVICTION_ACTIVATION_BLOCK - TEMPO

    class _Chain:
        def commit_reveal_enabled(self):
            return True

        def tempo(self):
            return TEMPO

        def metagraph(self):
            return type("Metagraph", (), {"hotkeys": ["hk0", "champ"], "uids": [0, 1], "S": [0.0, 100.0]})()

        def get_all_commitments(self):
            return {}

        def emissions_by_hotkey(self, block):
            return {}

    chain = _Chain()
    monkeypatch.setattr("validator.service.load_state", lambda path: state)
    monkeypatch.setattr("validator.service.save_state", lambda path, s: None)
    monkeypatch.setattr("validator.service._make_chain", lambda args: chain)
    monkeypatch.setattr("validator.service._local_version_key", lambda: 1)
    monkeypatch.setattr("validator.service._assert_version_key_matches", lambda chain: 1)
    monkeypatch.setattr(
        "validator.service._evaluate_round",
        lambda args, state, chain, salt: (CONVICTION_ACTIVATION_BLOCK - 1, None, {}, chain.metagraph()),
    )

    captured = {}

    def fake_decide_weights(*a, **kwargs):
        captured["gated"] = kwargs.get("gated_hotkeys")
        return [1.0, 0.0], False

    monkeypatch.setattr("validator.service.decide_weights", fake_decide_weights)

    class _FakeWandb:
        def log(self, *a, **k):
            pass

        def finish(self):
            pass

    args = type(
        "Args", (),
        {"state_dir": str(tmp_path), "salt_file": None, "dry_run": True, "burn_uid": 0, "window_anchor": 0},
    )()
    run_once(args, wandb_logger=_FakeWandb())

    assert captured["gated"] == set()


def test_scoring_version_is_untouched_by_conviction():
    # Conviction gates weights only; it is not a scoring-rule change and must not have
    # bumped SCORING_VERSION (persisted scores/exclusions stay valid). 3 is issue #136's
    # commit-order-gauntlet bump, which landed independently of conviction.
    assert SCORING_VERSION == 3


# --- backfill progress logging (issue #154) ----------------------------------------------


def _progress_chain(*, key=None, emissions=None):
    class _Chain:
        config = type("Config", (), {"blockmachine_api_key": key})()

        def emissions_by_hotkey(self, block):
            return emissions or {}

        def archive_emissions_by_hotkey(self, block):
            return emissions or {}

    return _Chain()


def _run_catchup(state, chain, *, samples, caplog):
    from bittensor.utils.btlogging import logging as bt_logging
    from validator.service import _update_conviction_ledger

    bt_logging.set_info()
    state.conviction_ledger.last_block = START
    _update_conviction_ledger(state, chain, START + samples * TEMPO, TEMPO)
    return caplog.text


def test_backfill_announces_source_and_logs_each_20_percent(caplog):
    out = _run_catchup(ValidatorState(), _progress_chain(), samples=90, caplog=caplog)

    assert out.count("backfilling ledger") == 1  # exactly one start line
    assert f"from block {START:,} to {START + 90 * TEMPO:,} (90 tempo samples) via public archive node" in out
    for pct, done in ((20, 18), (40, 36), (60, 54), (80, 72), (100, 90)):
        assert f"backfill {pct}% ({done}/90 samples, at block {START + done * TEMPO:,}," in out
    assert out.count("backfill ") == 5  # five progress lines, no more


def test_backfill_start_line_names_blockmachine_and_never_the_key(caplog):
    out = _run_catchup(ValidatorState(), _progress_chain(key="sekrit-key"), samples=10, caplog=caplog)

    assert "via blockmachine RPC" in out
    assert "sekrit-key" not in out


def test_steady_state_single_sample_catchup_stays_quiet(caplog):
    out = _run_catchup(ValidatorState(), _progress_chain(), samples=1, caplog=caplog)

    assert "backfilling ledger" not in out
    assert "backfill " not in out.replace("backfilling", "")
    assert "ledger advanced 1 tempo(s)" in out  # the existing completion line is enough


# --- v1.1: gate on chain-locked alpha, not raw stake (issue #156) ------------------------


LOCK_START = CONVICTION_LOCK_CHECK_START_BLOCK


def test_lock_rule_pays_a_locked_winner_and_gates_staked_but_unlocked():
    ledger = ConvictionLedger(earned={"champ": 5000.0, "prev": 5000.0})
    staked = {"champ": 5000.0, "prev": 5000.0}  # both would satisfy v1's staked rule
    locked = {"champ": 4000.0, "prev": 0.0}  # but only champ actually locked
    report = conviction_report(
        ledger, ["champ", "prev"], staked, block=LOCK_START, conviction_by_hotkey=locked
    )
    assert report["champ"]["compliant"] is True
    # The v1 cliff-exit hole, pinned: fully staked but unlocked no longer satisfies the
    # gate -- stake can be dumped at any block, locked mass cannot.
    assert report["prev"]["compliant"] is False


def test_decaying_lock_gates_below_the_line_until_relocked():
    ledger = ConvictionLedger(earned={"champ": 5000.0})
    staked = {"champ": 5000.0}
    decayed = conviction_report(
        ledger, ["champ"], staked, block=LOCK_START, conviction_by_hotkey={"champ": 3999.99}
    )
    assert decayed["champ"]["compliant"] is False
    relocked = conviction_report(
        ledger, ["champ"], staked, block=LOCK_START, conviction_by_hotkey={"champ": 4000.0}
    )
    assert relocked["champ"]["compliant"] is True  # per-tempo re-check, reversible as ever


def test_staked_rule_applies_before_the_lock_check_start_block():
    # Post-activation but pre-switch (the announced grace window): v1's staked rule still
    # decides, while the lock is already reported so operators can watch winners lock up.
    ledger = ConvictionLedger(earned={"champ": 5000.0})
    report = conviction_report(
        ledger, ["champ"], {"champ": 4000.0}, block=LOCK_START - 1, conviction_by_hotkey={"champ": 0.0}
    )
    assert report["champ"]["compliant"] is True
    assert report["champ"]["conviction"] == 0.0


def test_lock_read_unavailable_falls_back_to_the_staked_rule():
    # locked <= staked always (locking requires the stake), so the fallback can never gate
    # a lock-compliant winner -- and it still gates a fully-unstaked dumper.
    ledger = ConvictionLedger(earned={"champ": 5000.0, "prev": 5000.0})
    staked = {"champ": 4000.0, "prev": 0.0}
    report = conviction_report(
        ledger, ["champ", "prev"], staked, block=LOCK_START, conviction_by_hotkey=None
    )
    assert report["champ"]["compliant"] is True
    assert report["prev"]["compliant"] is False
    assert report["champ"]["conviction"] is None  # honestly absent, not fabricated as 0


def test_service_report_reads_locks_from_the_chain_and_survives_a_failed_read():
    from validator.service import _conviction_report_for_winners

    state = ValidatorState()
    state.winner_history = [
        WinnerEntry(hotkey="champ", repo="c/r", revision="rev1", ratio=0.4, commit_block=10)
    ]
    state.conviction_ledger.earned["champ"] = 5000.0
    metagraph = type(
        "Metagraph", (), {"hotkeys": ["burn-sink", "champ"], "alpha_stake": [0.0, 5000.0]}
    )()

    class _Chain:
        def locked_alpha_by_hotkey(self, hotkeys):
            return {hotkey: 4000.0 for hotkey in hotkeys}

    report = _conviction_report_for_winners(state, metagraph, LOCK_START, _Chain())
    assert report["champ"]["conviction"] == 4000.0
    assert report["champ"]["compliant"] is True

    class _Broken:
        def locked_alpha_by_hotkey(self, hotkeys):
            raise ConnectionError("runtime api unavailable")

    fallback = _conviction_report_for_winners(state, metagraph, LOCK_START, _Broken())
    assert fallback["champ"]["conviction"] is None
    assert fallback["champ"]["compliant"] is True  # staked-rule fallback, staked 5000 >= 4000


def test_build_weights_metrics_logs_conviction_only_when_the_read_was_available():
    entry = {"earned": 5000.0, "staked": 5000.0, "required_conviction": 4000.0, "compliant": True}
    with_lock = build_weights_metrics(
        block=1, tempo=TEMPO, is_burn_tempo=False, uids=[0], weights=[1.0],
        conviction={"champ": dict(entry, conviction=4000.0)},
    )
    assert with_lock["conviction/champ/conviction"] == 4000.0
    without = build_weights_metrics(
        block=1, tempo=TEMPO, is_burn_tempo=False, uids=[0], weights=[1.0],
        conviction={"champ": dict(entry, conviction=None)},
    )
    assert "conviction/champ/conviction" not in without


def _deep_state(earned_by_hotkey=None, depth=None):
    """A retained history `depth` entries deep, with an optional per-hotkey earned ledger."""

    from core.constants import WINNER_HISTORY_DEPTH

    depth = depth or WINNER_HISTORY_DEPTH
    state = ValidatorState()
    state.winner_history = [
        WinnerEntry(hotkey=f"w{i}", repo=f"r{i}/c", revision=f"rev{i}", ratio=0.4, commit_block=10)
        for i in range(depth)
    ]
    state.conviction_ledger.earned.update(earned_by_hotkey or {})
    # The burn sink occupies BURN_UID (index 0) on a real metagraph, as compute_weights
    # assumes -- a fixture without it would let the report walk treat the top winner as
    # the burn hotkey (or miss that the burn hotkey must be skipped at all).
    hotkeys = ["burn-sink", *[w.hotkey for w in state.winner_history]]
    metagraph = type(
        "Metagraph", (), {"hotkeys": hotkeys, "alpha_stake": [0.0] * len(hotkeys)},
    )()
    return state, metagraph


def test_history_depth_is_the_owner_set_twenty():
    # Issue #175: owner deepened the fallback ladder 5 -> 20. Pinned so a silent revert
    # fails loudly rather than only shortening the ladder in production.
    from core.constants import WINNER_HISTORY_DEPTH

    assert WINNER_HISTORY_DEPTH == 20


def test_report_stops_reading_once_two_compliant_winners_are_found():
    # issue #175: selection only needs WINNER_LIMIT compliant entries, so the steady state
    # costs two chain reads no matter how deep retention goes.
    from validator.service import _conviction_report_for_winners

    state, metagraph = _deep_state()
    asked = []

    class _Chain:
        def locked_alpha_by_hotkey(self, hotkeys):
            asked.extend(hotkeys)
            return {hotkey: 0.0 for hotkey in hotkeys}

    report = _conviction_report_for_winners(state, metagraph, LOCK_START, _Chain())
    assert asked == ["w0", "w1"]  # nothing below the two compliant winners was queried
    assert list(report) == ["w0", "w1"]


def test_report_walks_deeper_only_while_winners_are_gated():
    # Three gated winners (earned above the free allowance, nothing locked) -> the walk
    # pays for depth exactly as far as it must, then stops at the next two compliant.
    from validator.service import _conviction_report_for_winners

    state, metagraph = _deep_state({f"w{i}": 5000.0 for i in range(3)})
    asked = []

    class _Chain:
        def locked_alpha_by_hotkey(self, hotkeys):
            asked.extend(hotkeys)
            return {hotkey: 0.0 for hotkey in hotkeys}

    report = _conviction_report_for_winners(state, metagraph, LOCK_START, _Chain())
    assert asked == ["w0", "w1", "w2", "w3", "w4"]
    assert [h for h, e in report.items() if not e["compliant"]] == ["w0", "w1", "w2"]
    assert [h for h, e in report.items() if e["compliant"]] == ["w3", "w4"]


def test_every_retained_entry_is_evaluated_when_all_are_gated():
    from core.constants import WINNER_HISTORY_DEPTH
    from core.weights import compute_weights
    from validator.service import _conviction_report_for_winners

    earned = {f"w{i}": 5000.0 for i in range(WINNER_HISTORY_DEPTH)}
    state, metagraph = _deep_state(earned)

    class _Chain:
        def locked_alpha_by_hotkey(self, hotkeys):
            return {hotkey: 0.0 for hotkey in hotkeys}

    report = _conviction_report_for_winners(state, metagraph, LOCK_START, _Chain())
    assert len(report) == WINNER_HISTORY_DEPTH  # nothing compliant -> the full walk
    gated = {hotkey for hotkey, entry in report.items() if not entry["compliant"]}
    weights = compute_weights(
        ["burn-sink", *metagraph.hotkeys], state.winner_history, is_burn_tempo=False,
        gated_hotkeys=gated,
    )
    assert weights[0] == 1.0  # 20 gated entries and no fallback left -> burn


def test_one_failed_lock_read_isolates_to_that_hotkey():
    # issue #175: a single flaky query must not drop every winner to the staked rule --
    # at depth 20 that would silently un-gate winners that genuinely have no lock.
    from validator.service import _conviction_report_for_winners

    # w0 has stake but no lock (so the staked rule would wrongly pass it), w1 is locked.
    state, metagraph = _deep_state({"w0": 5000.0, "w1": 5000.0})
    metagraph.alpha_stake = [9000.0] * len(metagraph.hotkeys)

    class _FlakyChain:
        def locked_alpha_by_hotkey(self, hotkeys):
            if hotkeys == ["w1"]:
                raise ConnectionError("runtime api hiccup")
            return {hotkey: 0.0 for hotkey in hotkeys}

    report = _conviction_report_for_winners(state, metagraph, LOCK_START, _FlakyChain())
    assert report["w0"]["conviction"] == 0.0 and report["w0"]["compliant"] is False
    # w1's own read failed -> staked rule for it alone (9000 staked >= 4000 required).
    assert report["w1"]["conviction"] is None and report["w1"]["compliant"] is True


def test_the_ladder_reaches_the_deepest_retained_entry():
    # issue #175: with everything above it gated, the last retained entry is still paid --
    # burning is the last resort, not the second one.
    from core.constants import WINNER_HISTORY_DEPTH
    from core.weights import compute_weights

    history = [
        WinnerEntry(hotkey=f"w{i}", repo=f"r{i}/c", revision=f"rev{i}", ratio=0.4, commit_block=10)
        for i in range(WINNER_HISTORY_DEPTH)
    ]
    hotkeys = ["burn-sink", *[w.hotkey for w in history]]
    all_but_last = {f"w{i}" for i in range(WINNER_HISTORY_DEPTH - 1)}

    weights = compute_weights(hotkeys, history, is_burn_tempo=False, gated_hotkeys=all_but_last)
    assert weights[hotkeys.index(f"w{WINNER_HISTORY_DEPTH - 1}")] == 1.0
    assert weights[0] == 0.0 and sum(weights) == 1.0

    last_two = {f"w{i}" for i in range(WINNER_HISTORY_DEPTH - 2)}
    pair = compute_weights(hotkeys, history, is_burn_tempo=False, gated_hotkeys=last_two)
    assert pair[hotkeys.index(f"w{WINNER_HISTORY_DEPTH - 2}")] == 0.7
    assert pair[hotkeys.index(f"w{WINNER_HISTORY_DEPTH - 1}")] == 0.3


def test_a_burn_hotkey_in_history_never_consumes_a_compliant_slot():
    """Regression, PR #176 review: the lazy walk's stopping point is only sound while it
    skips exactly what select_payees skips. compute_weights gates the burn hotkey, so if
    this walk counted it as a compliant winner it would stop one entry early -- leaving
    the next winner unevaluated, absent from the gated set, and paid with no conviction
    check at all.
    """

    from core.constants import BURN_UID
    from core.weights import compute_weights
    from validator.service import _conviction_report_for_winners

    # burn-sink and w0 are trivially compliant (nothing earned); w1..w4 each earned 5000
    # with nothing locked, so every one of them must be gated.
    state, metagraph = _deep_state({f"w{i}": 5000.0 for i in range(1, 5)}, depth=5)
    state.winner_history.insert(0, WinnerEntry(
        hotkey=metagraph.hotkeys[BURN_UID], repo="b/c", revision="revb", ratio=0.4, commit_block=1
    ))

    class _Chain:
        def locked_alpha_by_hotkey(self, hotkeys):
            return {hotkey: 0.0 for hotkey in hotkeys}

    report = _conviction_report_for_winners(state, metagraph, LOCK_START, _Chain())
    assert metagraph.hotkeys[BURN_UID] not in report  # never read, never counted

    gated = {hotkey for hotkey, entry in report.items() if not entry["compliant"]}
    weights = compute_weights(
        metagraph.hotkeys, state.winner_history, is_burn_tempo=False, gated_hotkeys=gated
    )
    paid = {hotkey: w for hotkey, w in zip(metagraph.hotkeys, weights) if w > 0}

    # Every hotkey that gets paid must be one the report actually cleared.
    for hotkey in paid:
        assert report[hotkey]["compliant"] is True
    assert paid == {"w0": 1.0}  # w1..w4 all gated -> w0 is the only qualifier
