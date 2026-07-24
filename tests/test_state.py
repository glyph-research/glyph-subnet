"""issue #180: recovering `improvement` for winner entries persisted before #177.

The real improvement of a pre-#177 winner is recoverable from state we already hold --
each entry dethroned the next one down the retained history -- so it is recomputed at load
time rather than defaulted. Only the oldest retained entry, whose opponent is no longer
retained, falls back to DEFAULT_WIN_MARGIN.
"""

import json

from core.constants import DEFAULT_WIN_MARGIN
from core.state import load_state, save_state
from core.weights import compute_weights, winner_share

# --- issue #180: backfill improvement for pre-#177 winners -------------------------------


def _write_state(path, entries, stamped=()):
    """Persist a winner_history whose entries omit `improvement` unless listed in
    `stamped` (index -> value), mimicking JSON written before #177."""

    history = []
    for index, (hotkey, ratio) in enumerate(entries):
        entry = {
            "hotkey": hotkey, "repo": f"{hotkey}/c", "revision": "rev123456",
            "ratio": ratio, "commit_block": 10,
        }
        if index in dict(stamped):
            entry["improvement"] = dict(stamped)[index]
        history.append(entry)
    path.write_text(json.dumps({"winner_history": history}))


def test_backfill_recovers_the_real_improvement_from_adjacent_ratios(tmp_path):
    # The live-state acceptance case from issue #180: uid 124 dethroned uid 122 by 5.56%,
    # which the 1%-default migration would have understated into a 40% share.
    path = tmp_path / "state.json"
    _write_state(path, [("uid124", 0.07199660936991374), ("uid122", 0.0762338638305664)])

    history = load_state(path).winner_history
    assert abs(history[0].improvement - 0.0555833) < 1e-6
    # 25 + 15*5.558 = 108% -> capped at the pot: the champion takes everything.
    assert winner_share(history[0], is_top_payee=True) > 1.0
    weights = compute_weights(
        ["burn-sink", "uid124", "uid122"], history, is_burn_tempo=False, burn_uid=0
    )
    assert weights == [0.0, 1.0, 0.0]


def test_the_oldest_retained_entry_keeps_the_margin_fallback(tmp_path):
    # It dethroned a winner we no longer retain, so there is nothing to compute from --
    # the one place DEFAULT_WIN_MARGIN legitimately applies.
    path = tmp_path / "state.json"
    _write_state(path, [("a", 0.40), ("b", 0.50)])
    assert load_state(path).winner_history[-1].improvement == DEFAULT_WIN_MARGIN


def test_a_stamped_improvement_is_never_recomputed(tmp_path):
    # A genuine 1% dethrone is legitimate and must survive the migration untouched, even
    # though the adjacent ratios would compute something else entirely.
    path = tmp_path / "state.json"
    _write_state(path, [("a", 0.40), ("b", 0.50)], stamped={0: 0.01})

    history = load_state(path).winner_history
    assert history[0].improvement == 0.01  # not the 20% the ratios would imply


def test_backfill_is_deterministic_and_survives_a_round_trip(tmp_path):
    # Every validator holds the same ratios, so the backfill lands identically; and once
    # written back the values are stamped, so a second load changes nothing.
    path = tmp_path / "state.json"
    _write_state(path, [("a", 0.40), ("b", 0.50), ("c", 0.80)])

    first = load_state(path)
    assert [round(e.improvement, 6) for e in first.winner_history] == [0.2, 0.375, DEFAULT_WIN_MARGIN]

    save_state(path, first)
    assert load_state(path).winner_history == first.winner_history


def test_a_worse_or_equal_ratio_backfills_to_zero_never_negative(tmp_path):
    # Defensive: history should never contain a winner worse than the one below it, but a
    # negative improvement would flip the sign of its share, so it clamps at 0.
    path = tmp_path / "state.json"
    _write_state(path, [("a", 0.60), ("b", 0.50)])
    assert load_state(path).winner_history[0].improvement == 0.0
