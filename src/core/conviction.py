"""Miner Conviction (issue #141): winners must keep earnings staked to keep earning.

41% of daily alpha flows to the two winner slots, and nothing else stops a long-reigning
champion from market-selling the whole position at once (king-dump-and-exit). The gate:
a winner hotkey whose total staked alpha falls below ``required_lock(earned)`` receives no
incentive that tempo -- its share goes to the burn sink (owner-confirmed: never reallocated
to the other winner, which would pay A for B's non-compliance). Reversible, not a verdict:
restaking above the line restores incentive at the next weight-setting, and the crown
itself is never affected.

Conviction v1.1 (issue #156): from ``CONVICTION_LOCK_CHECK_START_BLOCK`` the gated
quantity is the hotkey's chain-locked alpha (``btcli lock add``) instead of raw stake,
closing the cliff-exit v1 left open (stake could be fully unstaked at any block, so a
dethroned winner lost only future emission by dumping). The formula, both-slot gating,
burn-not-reallocate, and per-tempo reversibility are all unchanged -- only the measured
quantity is.

Everything here is pure and unit-tested; consensus safety comes from every validator
computing the identical ``earned`` ledger: one increment code path (``ledger_catchup``)
sampling the chain's per-tempo emission on a fixed block grid anchored at
``CONVICTION_TRACKING_START_BLOCK``, fed either live or from the archive node when
backfilling a gap -- same formula, two sources for the block data.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from core.constants import (
    CONVICTION_ACTIVATION_BLOCK,
    CONVICTION_FREE_ALPHA,
    CONVICTION_FREE_FRACTION,
    CONVICTION_LOCK_CHECK_START_BLOCK,
    CONVICTION_TRACKING_START_BLOCK,
)


def required_lock(earned: float) -> float:
    """Alpha that must remain staked to the winner's hotkey, given cumulative earnings.

    ``min(0.90 x earned, earned - 1000)`` clamped at >= 0. Equivalently, the free
    (unstaked-allowed) amount is ``max(10% x earned, 1000 alpha)``: young reigns
    (< 1000 earned) keep everything liquid, [1000, 10000] keeps exactly 1000 free, and
    above 10000 the 10% allowance takes over and grows with the reign.
    """

    return max(0.0, min((1.0 - CONVICTION_FREE_FRACTION) * earned, earned - CONVICTION_FREE_ALPHA))


def is_compliant(earned: float, staked: float) -> bool:
    """Alpha units on both sides -- price movement alone can never gate a compliant miner."""

    return staked >= required_lock(earned)


class ConvictionLedger(BaseModel):
    """Cumulative per-hotkey alpha earnings, persisted on the validator state.

    Totals are permanent per hotkey: dethrone-and-return does not reset the ledger.
    ``last_block`` is the last grid block already accumulated (0 = nothing yet; catchup
    then starts at ``CONVICTION_TRACKING_START_BLOCK``).
    """

    earned: dict[str, float] = Field(default_factory=dict)
    last_block: int = 0


def ledger_grid(last_block: int, current_block: int, tempo: int) -> list[int]:
    """The fixed sampling grid: ``CONVICTION_TRACKING_START_BLOCK + k*tempo`` for every
    grid point past ``last_block`` and at or before ``current_block``.

    Anchored at the protocol constant, never at "now", so every validator -- whenever it
    starts or however long it was down -- samples the identical blocks.
    """

    start = max(last_block, CONVICTION_TRACKING_START_BLOCK)
    steps_done = (start - CONVICTION_TRACKING_START_BLOCK) // tempo
    first = CONVICTION_TRACKING_START_BLOCK + (steps_done + 1) * tempo
    return list(range(first, current_block + 1, tempo))


def ledger_catchup(
    ledger: ConvictionLedger,
    *,
    current_block: int,
    tempo: int,
    emissions_at,
    on_applied=None,
) -> int:
    """Advance ``ledger`` to ``current_block``, one tempo-grid sample at a time.

    ``emissions_at(block) -> dict[hotkey, alpha]`` is the single increment code path: the
    caller feeds it from the live node for recent blocks and from the archive node when
    backfilling a gap or a fresh start. Returns the number of grid blocks accumulated.
    A raised exception leaves the ledger at the last fully-applied grid block, so the next
    catchup resumes exactly where this one stopped.

    ``on_applied(done, total, grid_block)`` (optional) fires after each fully-applied grid
    sample -- the caller's hook for progress logging (issue #154); this function itself
    stays log-agnostic.
    """

    blocks = ledger_grid(ledger.last_block, current_block, tempo)
    for index, block in enumerate(blocks, start=1):
        for hotkey, alpha in emissions_at(block).items():
            if alpha > 0:
                ledger.earned[hotkey] = ledger.earned.get(hotkey, 0.0) + float(alpha)
        ledger.last_block = block
        if on_applied is not None:
            on_applied(index, len(blocks), block)
    return len(blocks)


def conviction_report(
    ledger: ConvictionLedger,
    winner_hotkeys: list[str],
    staked_by_hotkey: dict[str, float],
    *,
    block: int,
    locked_by_hotkey: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Per-winner compliance snapshot for this weight-setting.

    Before ``CONVICTION_ACTIVATION_BLOCK`` every winner reports (and is) compliant --
    ledgers warm up, nothing gates. ``compliant=False`` means the caller must move that
    slot's weight to the burn sink this tempo.

    ``locked_by_hotkey`` is the hotkey's decay-adjusted chain-locked alpha (Conviction
    v1.1, issue #156): from ``CONVICTION_LOCK_CHECK_START_BLOCK`` it replaces raw stake
    as the gated quantity -- plain stake can be cliff-unstaked at any block, locked mass
    cannot. ``None`` (caller's lock read unavailable) falls back to the v1 staked rule
    for this tempo: since locked alpha is a subset of staked alpha, the fallback can
    never gate a lock-compliant winner, and still gates a fully-unstaked one. Either
    lock mode satisfies the gate; a decaying lock simply falls below the line as it
    decays and gates until re-locked.
    """

    report: dict[str, dict] = {}
    active = block >= CONVICTION_ACTIVATION_BLOCK
    lock_rule = block >= CONVICTION_LOCK_CHECK_START_BLOCK and locked_by_hotkey is not None
    for hotkey in winner_hotkeys:
        earned = ledger.earned.get(hotkey, 0.0)
        staked = staked_by_hotkey.get(hotkey, 0.0)
        locked = locked_by_hotkey.get(hotkey, 0.0) if locked_by_hotkey is not None else None
        measured = locked if lock_rule else staked
        report[hotkey] = {
            "earned": earned,
            "staked": staked,
            "locked": locked,
            "required_lock": required_lock(earned),
            "compliant": (not active) or is_compliant(earned, measured),
        }
    return report
