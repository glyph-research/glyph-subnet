"""issue #43 (disable) / issue #88 (re-enable): the temporal burn feature is a network-wide,
source-committed on/off switch.

Enabled (the shipped default as of #88): the burn-tempo schedule applies -- decide_weights
restores 100% to BURN_UID on the computed burn tempo, sole-winner distribution otherwise.
Disabling (flipping the constant) must restore the exact prior no-burn behaviour.
"""

from core.constants import BURN_UID, BURN_WINDOW_TEMPOS
from core.weights import WinnerEntry
import weight_setter.service as weight_setter_service
from weight_setter.service import decide_weights

TEMPO = 360
HOTKEYS = ["uid0_burn", "hkA", "hkB"]
OUTPUTS = [("s0", 1000, "aa"), ("s1", 900, "bb")]


def history():
    return [WinnerEntry("hkA", "hkA/codec", "rev1", ratio=0.6, commit_block=0)]


def test_burn_enabled_by_default():
    assert weight_setter_service.BURN_ENABLED is True


def test_enabled_by_default_matches_burn_schedule_across_many_windows():
    from core.burn_schedule import derive_burn_seed, is_burn_tempo

    seed = derive_burn_seed(OUTPUTS)
    saw_burn_tempo = False
    for window in range(20):
        for position in range(BURN_WINDOW_TEMPOS):
            tempo_idx = window * BURN_WINDOW_TEMPOS + position
            block = tempo_idx * TEMPO
            weights, burn = decide_weights(
                HOTKEYS, history(), block=block, tempo=TEMPO, last_round_outputs=OUTPUTS
            )
            expected = is_burn_tempo(block, TEMPO, seed, 0)
            assert burn == expected
            if burn:
                saw_burn_tempo = True
                assert weights[BURN_UID] == 1.0
                assert sum(weights[1:]) == 0.0
            else:
                assert weights[BURN_UID] == 0.0
                assert weights[1] == 1.0  # pure rolling-winner distribution, sole winner
    assert saw_burn_tempo, "expected at least one burn tempo across 20 windows"


def test_disabled_via_monkeypatch_never_burns(monkeypatch):
    """The kill-switch direction: flipping BURN_ENABLED to False must restore the pure
    rolling-winner distribution with no tempo ever burning."""

    monkeypatch.setattr(weight_setter_service, "BURN_ENABLED", False)
    for window in range(50):
        for position in range(BURN_WINDOW_TEMPOS):
            tempo_idx = window * BURN_WINDOW_TEMPOS + position
            block = tempo_idx * TEMPO
            weights, burn = decide_weights(
                HOTKEYS, history(), block=block, tempo=TEMPO, last_round_outputs=OUTPUTS
            )
            assert burn is False
            assert weights[BURN_UID] == 0.0
            assert weights[1] == 1.0  # pure rolling-winner distribution, sole winner


def test_disabled_via_monkeypatch_matches_compute_weights_with_is_burn_tempo_false(monkeypatch):
    from core.weights import compute_weights

    monkeypatch.setattr(weight_setter_service, "BURN_ENABLED", False)
    weights, burn = decide_weights(
        HOTKEYS, history(), block=0, tempo=TEMPO, last_round_outputs=OUTPUTS
    )
    assert burn is False
    assert weights == compute_weights(HOTKEYS, history(), is_burn_tempo=False, burn_uid=BURN_UID)


# --- owner emergency burn override (issue #113) -----------------------------------


def test_force_burn_burns_even_on_a_non_burn_tempo():
    # block=0 with this seed/anchor is not a scheduled burn tempo (see the disabled-by-default
    # tests above using the same fixture) -- force_burn must override that unconditionally.
    weights, burn = decide_weights(
        HOTKEYS, history(), block=0, tempo=TEMPO, last_round_outputs=OUTPUTS, force_burn=True
    )
    assert burn is True
    assert weights[BURN_UID] == 1.0
    assert sum(weights[1:]) == 0.0


def test_force_burn_true_even_when_burn_enabled_is_false(monkeypatch):
    # The owner override must win regardless of the network-wide BURN_ENABLED switch --
    # it's a distinct, higher-priority emergency mechanism, not gated by it.
    monkeypatch.setattr(weight_setter_service, "BURN_ENABLED", False)
    weights, burn = decide_weights(
        HOTKEYS, history(), block=0, tempo=TEMPO, last_round_outputs=OUTPUTS, force_burn=True
    )
    assert burn is True
    assert weights[BURN_UID] == 1.0


def test_force_burn_false_falls_through_to_the_normal_schedule():
    # force_burn=False (the default) must reproduce the exact behavior of not passing it.
    for window in range(20):
        for position in range(BURN_WINDOW_TEMPOS):
            tempo_idx = window * BURN_WINDOW_TEMPOS + position
            block = tempo_idx * TEMPO
            with_default = decide_weights(
                HOTKEYS, history(), block=block, tempo=TEMPO, last_round_outputs=OUTPUTS
            )
            with_explicit_false = decide_weights(
                HOTKEYS, history(), block=block, tempo=TEMPO, last_round_outputs=OUTPUTS,
                force_burn=False,
            )
            assert with_default == with_explicit_false


def test_resolve_force_burn_true_only_for_a_true_owner_commitment():
    from core.commitments import BurnOverrideCommitment, serialize_burn_override
    from weight_setter.service import resolve_force_burn

    owner = "owner-hk"
    true_raw = serialize_burn_override(BurnOverrideCommitment(force_burn=True))
    false_raw = serialize_burn_override(BurnOverrideCommitment(force_burn=False))

    assert resolve_force_burn({owner: true_raw}, owner) is True
    assert resolve_force_burn({owner: false_raw}, owner) is False


def test_resolve_force_burn_false_when_missing_or_malformed_or_wrong_hotkey():
    from core.commitments import BurnOverrideCommitment, serialize_burn_override
    from weight_setter.service import resolve_force_burn

    owner = "owner-hk"
    true_raw = serialize_burn_override(BurnOverrideCommitment(force_burn=True))

    assert resolve_force_burn({}, owner) is False  # no commitment at all
    assert resolve_force_burn({owner: "garbage"}, owner) is False  # malformed
    assert resolve_force_burn({"someone-else": true_raw}, owner) is False  # wrong hotkey
    # A genuine codec commitment on the owner's own slot must not be mistaken for an override.
    assert resolve_force_burn({owner: "g1|a/b|" + "a" * 40}, owner) is False


def test_run_wires_the_owner_override_end_to_end(monkeypatch, tmp_path):
    """weight_setter.service.run() (the split-service PM2 process) resolves and threads
    force_burn through exactly like validator.service.run_once -- test the whole path once
    end to end rather than just the pieces."""

    from argparse import Namespace

    from core.commitments import BurnOverrideCommitment, serialize_burn_override
    from core.state import ValidatorState
    from weight_setter.service import run as run_weight_setter

    override_raw = serialize_burn_override(BurnOverrideCommitment(force_burn=True))

    class FakeChainForRun:
        def __init__(self, _config):
            pass

        def current_block(self):
            return 0

        def tempo(self):
            return TEMPO

        def metagraph(self):
            return type("Metagraph", (), {"hotkeys": ["owner-hk"], "uids": [0]})()

        def get_all_commitments(self):
            return {"owner-hk": override_raw}

        def blocks_until_weights_allowed(self):
            return None

        def set_weights(self, uids, weights, version_key):  # pragma: no cover - dry_run=True
            raise AssertionError("dry_run=True must not call set_weights")

    monkeypatch.setattr("weight_setter.service.load_state", lambda _path: ValidatorState())
    monkeypatch.setattr("weight_setter.service.assert_weights_version_matches", lambda chain: 1)
    monkeypatch.setattr("chain.chain.BittensorChain", FakeChainForRun)

    captured = {}

    def fake_decide_weights(*a, **kwargs):
        captured["force_burn"] = kwargs.get("force_burn")
        return [1.0], True

    monkeypatch.setattr("weight_setter.service.decide_weights", fake_decide_weights)

    args = Namespace(
        netuid=117, network="test", wallet_name="w", hotkey_name="h", wallet_path=None,
        state_dir=str(tmp_path), burn_uid=0, window_anchor=0, dry_run=True,
    )
    run_weight_setter(args)  # must not raise (dry_run short-circuits before set_weights)

    assert captured["force_burn"] is True
