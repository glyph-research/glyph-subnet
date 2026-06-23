from core.weights import (
    WinnerEntry,
    compact_history,
    compute_weights,
    promote_winner,
    rank_key,
    rolling_weights_for_hotkeys,
    should_promote,
)


def entry(hotkey, ratio, block=0):
    return WinnerEntry(hotkey=hotkey, repo=f"{hotkey}/codec", revision="rev123456", ratio=ratio, commit_block=block)


# --- should_promote: lower ratio is better, strict epsilon to dethrone -----------

def test_vacant_crown_promotes():
    assert should_promote(0.90, None, 0.005) is True


def test_strict_epsilon_beat_promotes():
    # 0.6% better than incumbent 0.85 -> 0.845 <= 0.85*0.995 (=0.84575)
    assert should_promote(0.845, 0.85, 0.005) is True


def test_within_epsilon_does_not_promote():
    # 0.846 is better than 0.85 but not by the full 0.5% margin
    assert should_promote(0.846, 0.85, 0.005) is False


def test_exact_tie_does_not_dethrone():
    assert should_promote(0.85, 0.85, 0.005) is False


def test_worse_ratio_does_not_promote():
    assert should_promote(0.90, 0.85, 0.005) is False


# --- tie-break ordering ----------------------------------------------------------

def test_rank_key_prefers_lower_ratio_then_earlier_block():
    a = entry("a", 0.80, block=200)
    b = entry("b", 0.80, block=100)
    c = entry("c", 0.79, block=999)
    ranked = sorted([a, b, c], key=rank_key)
    assert [e.hotkey for e in ranked] == ["c", "b", "a"]


# --- history compaction ----------------------------------------------------------

def test_compact_history_dedup_and_limit_two():
    history = [entry("a", 0.8), entry("a", 0.81), entry("b", 0.82), entry("c", 0.83)]
    compacted = compact_history(history)
    assert [e.hotkey for e in compacted] == ["a", "b"]


def test_promote_winner_pushes_previous_down():
    history = [entry("a", 0.80)]
    history = promote_winner(history, entry("b", 0.78))
    assert [e.hotkey for e in history] == ["b", "a"]


# --- weight computation ----------------------------------------------------------

HOTKEYS = ["uid0_burn", "hkA", "hkB", "hkC"]


def test_normal_tempo_rolling_70_30():
    history = [entry("hkA", 0.78), entry("hkB", 0.80)]
    weights = compute_weights(HOTKEYS, history, is_burn_tempo=False, burn_uid=0)
    assert weights[1] == 0.70
    assert weights[2] == 0.30
    assert weights[0] == 0.0 and weights[3] == 0.0
    assert abs(sum(weights) - 1.0) < 1e-9


def test_single_winner_normalizes_to_full_weight():
    history = [entry("hkA", 0.78)]
    weights = compute_weights(HOTKEYS, history, is_burn_tempo=False, burn_uid=0)
    assert weights[1] == 1.0
    assert abs(sum(weights) - 1.0) < 1e-9


def test_burn_tempo_sends_all_to_burn_uid():
    history = [entry("hkA", 0.78), entry("hkB", 0.80)]
    weights = compute_weights(HOTKEYS, history, is_burn_tempo=True, burn_uid=0)
    assert weights[0] == 1.0
    assert sum(weights[1:]) == 0.0


def test_empty_history_burns():
    weights = compute_weights(HOTKEYS, [], is_burn_tempo=False, burn_uid=0)
    assert weights[0] == 1.0


def test_rolling_weights_ignores_unlisted_hotkeys():
    history = [entry("ghost", 0.5), entry("hkA", 0.78)]
    weights = rolling_weights_for_hotkeys(HOTKEYS, history)
    # ghost is not in HOTKEYS, so hkA becomes the sole eligible winner -> full weight
    assert weights[1] == 1.0
