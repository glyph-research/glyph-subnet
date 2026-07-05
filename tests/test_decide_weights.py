"""issue #43: the temporal burn feature is a network-wide, source-committed on/off switch.

Disabled (the shipped default): no tempo across many blocks/windows is ever a burn tempo,
and decide_weights returns the pure rolling-winner distribution. Re-enabling (flipping the
constant) must restore the exact previous burn behaviour.
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


def test_burn_disabled_by_default():
    assert weight_setter_service.BURN_ENABLED is False


def test_disabled_never_burns_across_many_windows():
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


def test_disabled_matches_compute_weights_with_is_burn_tempo_false():
    from core.weights import compute_weights

    weights, burn = decide_weights(
        HOTKEYS, history(), block=0, tempo=TEMPO, last_round_outputs=OUTPUTS
    )
    assert burn is False
    assert weights == compute_weights(HOTKEYS, history(), is_burn_tempo=False, burn_uid=BURN_UID)


def test_reenabled_restores_previous_burn_behaviour(monkeypatch):
    """Flipping BURN_ENABLED back to True must reproduce is_burn_tempo's own schedule exactly."""

    from core.burn_schedule import derive_burn_seed, is_burn_tempo

    monkeypatch.setattr(weight_setter_service, "BURN_ENABLED", True)
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
    assert saw_burn_tempo, "expected at least one burn tempo across 20 windows"
