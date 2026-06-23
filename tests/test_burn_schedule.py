from core.burn_schedule import (
    burn_position,
    derive_burn_seed,
    is_burn_tempo,
    tempo_index,
    window_index,
)
from core.constants import BURN_WINDOW_TEMPOS

TEMPO = 360
ANCHOR = 0


def window_blocks(window):
    """The four blocks landing one in each tempo of the given window."""
    start_tempo = window * BURN_WINDOW_TEMPOS
    return [(start_tempo + i) * TEMPO for i in range(BURN_WINDOW_TEMPOS)]


# --- seed derivation -------------------------------------------------------------

def test_bootstrap_seed_when_no_outputs():
    assert derive_burn_seed(None) == derive_burn_seed([])
    assert len(derive_burn_seed([])) == 32


def test_seed_is_deterministic_and_order_independent():
    a = [("s1", 100, "aa"), ("s2", 200, "bb")]
    b = [("s2", 200, "bb"), ("s1", 100, "aa")]
    assert derive_burn_seed(a) == derive_burn_seed(b)


def test_different_outputs_give_different_seed():
    a = derive_burn_seed([("s1", 100, "aa")])
    b = derive_burn_seed([("s1", 101, "aa")])
    assert a != b


# --- index math ------------------------------------------------------------------

def test_tempo_and_window_index():
    assert tempo_index(0, TEMPO, ANCHOR) == 0
    assert tempo_index(TEMPO, TEMPO, ANCHOR) == 1
    assert tempo_index(5 * TEMPO, TEMPO, ANCHOR) == 5
    assert window_index(0, TEMPO, ANCHOR) == 0
    assert window_index(4 * TEMPO, TEMPO, ANCHOR) == 1


def test_burn_position_in_range():
    seed = derive_burn_seed([("s1", 1, "x")])
    for window in range(50):
        assert 0 <= burn_position(seed, window) < BURN_WINDOW_TEMPOS


# --- exactly one burn tempo per window ------------------------------------------

def test_exactly_one_burn_tempo_per_window():
    seed = derive_burn_seed([("s1", 123, "deadbeef")])
    for window in range(64):
        flags = [is_burn_tempo(b, TEMPO, seed, ANCHOR) for b in window_blocks(window)]
        assert sum(flags) == 1, f"window {window} had {sum(flags)} burn tempos"


# --- honest agreement vs copy-cat divergence ------------------------------------

def test_same_seed_same_schedule():
    seed = derive_burn_seed([("s1", 50, "aa"), ("s2", 60, "bb")])
    blocks = [w * TEMPO for w in range(40)]
    sched_a = [is_burn_tempo(b, TEMPO, seed, ANCHOR) for b in blocks]
    sched_b = [is_burn_tempo(b, TEMPO, seed, ANCHOR) for b in blocks]
    assert sched_a == sched_b


def test_different_seed_diverges_somewhere():
    seed_true = derive_burn_seed([("s1", 50, "aa")])
    seed_guess = derive_burn_seed(None)  # copy-cat lacking outputs falls back to bootstrap
    blocks = [w * TEMPO for w in range(40)]
    sched_true = [is_burn_tempo(b, TEMPO, seed_true, ANCHOR) for b in blocks]
    sched_guess = [is_burn_tempo(b, TEMPO, seed_guess, ANCHOR) for b in blocks]
    assert sched_true != sched_guess
