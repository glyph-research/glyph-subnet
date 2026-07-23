from core.constants import DEFAULT_WIN_MARGIN
from core.weights import (
    WinnerEntry,
    allocate_pot,
    compact_history,
    compute_weights,
    promote_winner,
    rank_key,
    rolling_weights_for_hotkeys,
    should_promote,
    winner_share,
)


def entry(hotkey, ratio, block=0, improvement=DEFAULT_WIN_MARGIN):
    return WinnerEntry(
        hotkey=hotkey, repo=f"{hotkey}/codec", revision="rev123456", ratio=ratio,
        commit_block=block, improvement=improvement,
    )


# --- should_promote: lower ratio is better, 5% epsilon to dethrone ---------------

def test_vacant_crown_promotes():
    assert should_promote(0.90, None, 0.05) is True


def test_five_percent_beat_promotes():
    # ~5.9% better than incumbent 0.85 -> 0.80 <= 0.85*0.95 (=0.8075)
    assert should_promote(0.80, 0.85, 0.05) is True


def test_within_epsilon_does_not_promote():
    # 0.83 is better than 0.85 but not by the full 5% margin (0.83 > 0.8075)
    assert should_promote(0.83, 0.85, 0.05) is False


def test_five_percent_boundary_is_inclusive():
    # acceptance: a challenger dethrones iff challenger_ratio <= current_ratio * 0.95
    incumbent = 0.85
    threshold = incumbent * (1.0 - 0.05)  # exact dethrone line (== incumbent * 0.95)
    assert should_promote(threshold, incumbent, 0.05) is True  # exactly at the line dethrones
    assert should_promote(threshold * 1.0001, incumbent, 0.05) is False  # just above does not


def test_exact_tie_does_not_dethrone():
    assert should_promote(0.85, 0.85, 0.05) is False


def test_worse_ratio_does_not_promote():
    assert should_promote(0.90, 0.85, 0.05) is False


# --- tie-break ordering ----------------------------------------------------------

def test_rank_key_prefers_lower_ratio_then_earlier_block():
    a = entry("a", 0.80, block=200)
    b = entry("b", 0.80, block=100)
    c = entry("c", 0.79, block=999)
    ranked = sorted([a, b, c], key=rank_key)
    assert [e.hotkey for e in ranked] == ["c", "b", "a"]


# --- history compaction ----------------------------------------------------------

def test_compact_history_dedups_and_retains_to_the_history_depth():
    # Issue #170: retention depth is deliberately deeper than the two paid slots -- the
    # entries past them are the fallback ladder payment walks when a winner is gated.
    from core.constants import WINNER_HISTORY_DEPTH

    history = [entry("a", 0.8), entry("a", 0.81), entry("b", 0.82), entry("c", 0.83)]
    assert [e.hotkey for e in compact_history(history)] == ["a", "b", "c"]

    deeper_than_retention = [entry(f"hk{i}", 0.8) for i in range(WINNER_HISTORY_DEPTH + 3)]
    assert len(compact_history(deeper_than_retention)) == WINNER_HISTORY_DEPTH


def test_promote_winner_pushes_previous_down():
    history = [entry("a", 0.80)]
    history = promote_winner(history, entry("b", 0.78))
    assert [e.hotkey for e in history] == ["b", "a"]


# --- weight computation ----------------------------------------------------------

HOTKEYS = ["uid0_burn", "hkA", "hkB", "hkC"]


def test_normal_tempo_pays_by_improvement():
    # issue #177: shares follow the improvement each winner earned, not slot position.
    # hkA 2% -> 25 + 30 = 55%, hkB 1% -> 15%; together 70%, scaled up to fill the pot.
    history = [entry("hkA", 0.78, improvement=0.02), entry("hkB", 0.80, improvement=0.01)]
    weights = compute_weights(HOTKEYS, history, is_burn_tempo=False, burn_uid=0)
    assert abs(weights[1] - 0.55 / 0.70) < 1e-12
    assert abs(weights[2] - 0.15 / 0.70) < 1e-12
    assert weights[1] > weights[2]  # more improvement, more emission
    assert weights[0] == 0.0 and weights[3] == 0.0
    assert sum(weights) == 1.0


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


# --- issue #177: improvement-proportional pot allocation --------------------------


def test_owners_worked_example_exactly():
    # The acceptance table from issue #177:
    #   w1 +2% -> 25 + 15x2 = 55%   paid 55%
    #   w2 +1% ->      15x1 = 15%   paid 15%
    #   w3 +3% ->      15x3 = 45%   paid 30% (only 30% of the pot remained)
    #   w4     ->                   paid 0%
    payees = [
        entry("w1", 0.40, improvement=0.02),
        entry("w2", 0.41, improvement=0.01),
        entry("w3", 0.42, improvement=0.03),
        entry("w4", 0.43, improvement=0.01),
    ]
    shares = allocate_pot(payees)
    assert shares[0] == 0.55
    assert shares[1] == 0.15
    assert abs(shares[2] - 0.30) < 1e-12  # remainder cap: 45% computed, 30% left
    assert shares[3] == 0.0  # pot exhausted above it
    assert sum(shares) == 1.0


def test_base_share_is_paid_once_to_the_top_payee_only():
    top, second = allocate_pot([entry("a", 0.4, improvement=0.01), entry("b", 0.4, improvement=0.01)])
    # Both improved 1%; the top payee's extra 25% base is the whole difference between them.
    assert abs(top / second - (0.25 + 0.15) / 0.15) < 1e-12


def test_under_subscribed_shares_scale_up_to_fill_the_pot():
    # Owner decision: three winners at +1% compute to 40 + 15 + 15 = 70% -> scaled up so
    # the pot always reaches winners, with relative shares still tracking improvement.
    shares = allocate_pot([entry(h, 0.4, improvement=0.01) for h in "abc"])
    assert abs(shares[0] - 0.40 / 0.70) < 1e-12
    # Equal improvements earn equal shares, up to the 1-ulp float residue the exactness
    # correction parks on the last payee so the row sums to exactly 1.0.
    assert abs(shares[1] - shares[2]) < 1e-12
    assert abs(shares[1] - 0.15 / 0.70) < 1e-12
    assert sum(shares) == 1.0


def test_a_big_jump_takes_the_whole_pot():
    # Owner decision: no per-winner ceiling. 5% computes to 25 + 75 = 100%, and anything
    # larger is capped by the pot itself -- prior winners earn nothing that tempo.
    for improvement in (0.05, 0.10, 1.0):
        shares = allocate_pot([entry("big", 0.2, improvement=improvement), entry("prev", 0.4)])
        assert shares == [1.0, 0.0]


def test_a_vacant_crown_winner_earns_the_base_and_nothing_deeper():
    # No incumbent was dethroned -> improvement 0.0: it earns the base while it is the top
    # payee (scaled up when alone), and nothing at all once it is deeper in the ladder.
    alone = allocate_pot([entry("v", 0.4, improvement=0.0)])
    assert alone == [1.0]

    deeper = allocate_pot([entry("w", 0.3, improvement=0.02), entry("v", 0.4, improvement=0.0)])
    assert deeper[1] == 0.0


def test_pre_177_entries_migrate_to_the_minimum_margin():
    # Persisted entries have no improvement field; the default must be the smallest
    # nonzero margin so they earn a small share rather than nothing or an invented one.
    legacy = WinnerEntry(hotkey="old", repo="o/c", revision="rev123456", ratio=0.5)
    assert legacy.improvement == DEFAULT_WIN_MARGIN == 0.01
    assert winner_share(legacy, is_top_payee=False) == 0.15


def test_promotion_records_the_improvement_it_earned():
    history = [entry("incumbent", 0.50)]
    challenger = entry("challenger", 0.49)  # 2% better than 0.50
    history = promote_winner(history, challenger, dethroned_ratio=0.50)
    assert history[0].hotkey == "challenger"
    assert abs(history[0].improvement - 0.02) < 1e-12
    assert abs(winner_share(history[0], is_top_payee=True) - 0.55) < 1e-12

    # A vacant crown records no improvement at all.
    vacant = promote_winner([], entry("first", 0.60), dethroned_ratio=None)
    assert vacant[0].improvement == 0.0


def test_fractional_percents_are_linear():
    # 1.5% improvement is worth 22.5%, not floored to a whole percent.
    assert abs(winner_share(entry("a", 0.4, improvement=0.015), is_top_payee=False) - 0.225) < 1e-12


def test_one_percent_is_the_dethrone_boundary():
    # issue #177 lowered the margin 5% -> 1%.
    assert DEFAULT_WIN_MARGIN == 0.01
    assert should_promote(0.99, 1.00, DEFAULT_WIN_MARGIN) is True  # exactly 1% better
    assert should_promote(0.991, 1.00, DEFAULT_WIN_MARGIN) is False  # just under
