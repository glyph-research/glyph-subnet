"""Rolling winner history and weight generation.

Glyph scores codecs by compression ratio, where **lower is better**, so the promotion
comparator is inverted relative to a benchmark-accuracy subnet. Weights follow a
two-slot rolling winner policy (current 70% / previous 30%), overlaid with the temporal
burn schedule: on a burn tempo all weight goes to the burn UID.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from core.constants import (
    BURN_UID,
    DEFAULT_WIN_MARGIN,
    WINNER_BASE_SHARE,
    WINNER_HISTORY_DEPTH,
    WINNER_IMPROVEMENT_MULTIPLIER,
)


@dataclass(frozen=True)
class WinnerEntry:
    hotkey: str
    repo: str
    revision: str
    ratio: float  # compressed_bytes / raw_bytes; lower is better
    commit_block: int = 0  # earliest commit wins ties (makes copying worthless)
    # Fractional ratio improvement over the winner this entry dethroned (0.02 == 2%), the
    # basis of its emission share (issue #177). Recorded at promotion time by
    # ``promote_winner``, never recomputed later: the comparison is only meaningful
    # same-round (challenger vs the incumbent it beat, on identical streams), and the
    # opponent may not even exist by the time weights are set. 0.0 for a vacant crown
    # (nothing was dethroned) -- such a winner earns the base share and no more.
    # The default is the migration value for entries persisted before #177: the minimum
    # possible margin, so they earn the smallest nonzero improvement share rather than
    # nothing or an invented large number.
    improvement: float = DEFAULT_WIN_MARGIN

    @property
    def key(self) -> str:
        return f"{self.hotkey}:{self.repo}@{self.revision}"


def rank_key(entry: WinnerEntry) -> tuple[float, int]:
    """Sort key for picking the best among already-scored entries (e.g. the vacant-crown
    recovery): best (lowest) ratio, then earliest commit.

    NOT the challenge order -- issue #136 made the round a sequential gauntlet in commit
    order (``commit_order_key``); ranking challengers best-ratio-first is exactly the
    best-of-round semantics that replaced.
    """

    return (entry.ratio, entry.commit_block)


def commit_order_key(entry: WinnerEntry) -> tuple[int, str]:
    """Sequential-gauntlet challenge order (issue #136, owner-specified): earliest commit
    challenges first, hotkey as the deterministic tie-break for identical blocks --
    consistent with the duplicate-artifact ownership rule (issue #58). An earlier commit
    that legitimately dethrones the incumbent is then protected by the full margin against
    everything committed after it, so a copier's same-round marginal tweak wins nothing."""

    return (entry.commit_block, entry.hotkey)


def compact_history(
    history: list[WinnerEntry],
    eligible_hotkeys: set[str] | None = None,
    limit: int = WINNER_HISTORY_DEPTH,
) -> list[WinnerEntry]:
    """Deduplicated, eligibility-filtered winner history, newest first.

    ``limit`` is the RETENTION depth (issue #170), not the number of paid slots: payment
    walks this list and pays the first ``WINNER_LIMIT`` compliant entries, so the entries
    past the paid slots are the fallback ladder that keeps emission flowing when a winner
    is conviction-gated.
    """

    seen: set[str] = set()
    compacted: list[WinnerEntry] = []
    for entry in history:
        if eligible_hotkeys is not None and entry.hotkey not in eligible_hotkeys:
            continue
        if entry.hotkey in seen:
            continue
        seen.add(entry.hotkey)
        compacted.append(entry)
        if len(compacted) == limit:
            break
    return compacted


def promote_winner(
    history: list[WinnerEntry],
    winner: WinnerEntry,
    eligible_hotkeys: set[str] | None = None,
    dethroned_ratio: float | None = None,
) -> list[WinnerEntry]:
    """Crown ``winner``, stamping the improvement it just earned (issue #177).

    ``dethroned_ratio`` is the ratio of the winner it beat, from the same round on the
    same streams -- the only moment that comparison is meaningful. ``None`` means a vacant
    crown was taken (nothing to improve on), which records 0.0: that winner earns the base
    share while it is the top payee and nothing once it is deeper in the ladder.
    """

    if dethroned_ratio is not None and dethroned_ratio > 0:
        improvement = max(0.0, (dethroned_ratio - winner.ratio) / dethroned_ratio)
    else:
        improvement = 0.0
    next_history = [replace(winner, improvement=improvement), *history]
    if eligible_hotkeys is not None:
        eligible_hotkeys = set(eligible_hotkeys) | {winner.hotkey}
    return compact_history(next_history, eligible_hotkeys=eligible_hotkeys)


def should_promote(
    challenger_ratio: float,
    current_ratio: float | None,
    margin: float,
) -> bool:
    """Lower ratio is better. Dethroning the incumbent requires a strict epsilon beat.

    A vacant crown (``current_ratio is None``) is taken by any eligible challenger; the
    caller is responsible for the baseline (zstd -19) floor and earliest-commit
    tie-break among equally good challengers.
    """

    if current_ratio is None:
        return True
    return challenger_ratio <= current_ratio * (1.0 - margin)


def winner_share(entry: WinnerEntry, *, is_top_payee: bool) -> float:
    """The pot fraction ``entry`` has earned (issue #177), before the top-down cap.

    ``WINNER_BASE_SHARE`` for being the winner currently being paid, plus
    ``WINNER_IMPROVEMENT_MULTIPLIER`` x the improvement it recorded when it took the
    crown. Uncapped by design (owner decision): a >=5% jump computes to >=100% and takes
    the entire pot, which is the sharpest possible statement that big frontier moves are
    what the subnet pays for.
    """

    base = WINNER_BASE_SHARE if is_top_payee else 0.0
    return base + WINNER_IMPROVEMENT_MULTIPLIER * max(0.0, entry.improvement)


def allocate_pot(payees: list[WinnerEntry]) -> list[float]:
    """Pot fractions for ``payees`` (newest first), summing to exactly 1.0 -- or empty
    when there is nobody to pay and the caller should burn (issue #177).

    Top-down with a remainder cap: each winner receives ``min(its share, what is left)``,
    so a large improvement near the top can exhaust the pot and starve everyone below it.
    If the computed shares under-subscribe the pot instead, they are scaled up to fill it
    (owner decision) -- the pot always reaches winners, and relative shares still track
    relative improvement.
    """

    if not payees:
        return []
    shares: list[float] = []
    remaining = 1.0
    for index, entry in enumerate(payees):
        paid = min(winner_share(entry, is_top_payee=(index == 0)), remaining)
        shares.append(paid)
        remaining -= paid
        if remaining <= 0.0:
            break
    total = sum(shares)
    if total <= 0.0:
        return []
    shares = [share / total for share in shares]
    # Absorb float residue in the last paid entry so the row sums to EXACTLY 1.0: every
    # validator runs the identical ops in the identical order, so this stays deterministic
    # while keeping the weight vector exact rather than 1.0 +/- 1e-16.
    shares[-1] += 1.0 - sum(shares)
    shares.extend([0.0] * (len(payees) - len(shares)))
    return shares


def select_payees(
    history: list[WinnerEntry],
    eligible_hotkeys: set[str],
    gated_hotkeys: set[str] | None = None,
    limit: int = WINNER_HISTORY_DEPTH,
) -> list[WinnerEntry]:
    """The winners eligible for payment this tempo: entries of ``history`` (newest first)
    that are both eligible and conviction-compliant (issue #170).

    ``limit`` is the retained depth, not a paid-slot count: since #177 the number of paid
    winners is dynamic, decided by ``allocate_pot`` running out of pot.

    Walking past a gated entry is what makes burning rare: a non-compliant winner is
    skipped entirely and everyone below it moves up a rung, inheriting the larger shares
    (the base included). Compliance is re-read every tempo, so a winner that locks
    re-enters the ladder at its own position at the next weight-setting, and a dethroned
    winner that stays locked keeps earning whenever someone above it lapses.

    Selection never touches the crown: ``history[0]`` remains the incumbent for scoring,
    dethroning, and the commit-order gauntlet whatever its compliance.
    """

    gated = gated_hotkeys or set()
    payees: list[WinnerEntry] = []
    seen: set[str] = set()
    for entry in history:
        if entry.hotkey in seen or entry.hotkey not in eligible_hotkeys or entry.hotkey in gated:
            continue
        seen.add(entry.hotkey)
        payees.append(entry)
        if len(payees) == limit:
            break
    return payees


def rolling_weights_for_hotkeys(
    hotkeys: list[str],
    history: list[WinnerEntry],
    gated_hotkeys: set[str] | None = None,
) -> list[float]:
    payees = select_payees(history, set(hotkeys), gated_hotkeys)
    shares = allocate_pot(payees)
    if not shares:
        return [0.0 for _ in hotkeys]

    by_hotkey = {entry.hotkey: shares[index] for index, entry in enumerate(payees)}
    return [by_hotkey.get(hotkey, 0.0) for hotkey in hotkeys]


def compute_weights(
    hotkeys: list[str],
    history: list[WinnerEntry],
    *,
    is_burn_tempo: bool,
    burn_uid: int = BURN_UID,
    gated_hotkeys: set[str] | None = None,
) -> list[float]:
    """Final validator weights for a single tempo.

    On a burn tempo all weight goes to ``burn_uid``. On a normal tempo the pot is
    allocated by improvement (issue #177, see ``allocate_pot``): the winner ladder is
    walked newest-first and each winner earns a share of the pot proportional to how far
    it moved the frontier, until the pot runs out. If there is no eligible winner at all
    the emission is burned rather than spread arbitrarily.

    ``gated_hotkeys`` (Miner Conviction, issues #141/#166/#170): winners whose conviction
    is below the requirement this tempo. They are skipped when picking payees, so the pot
    flows to the compliant winners further down the retained history and burns only
    when none of them qualifies (owner decision, issue #170, superseding #166's
    redistribute-between-two-slots: no winner has any lever over another's conviction --
    compliance is purely that party's own lock -- so paying down the ladder is not an
    attack surface, and it keeps emission flowing to compliant, aligned winners instead of
    burning). Reversible by construction: history is untouched, so a winner that locks is
    paid again at the very next weight-setting.
    """

    if not 0 <= burn_uid < len(hotkeys):
        raise ValueError("burn_uid is outside the hotkey list")

    weights = [0.0 for _ in hotkeys]
    if is_burn_tempo:
        weights[burn_uid] = 1.0
        return weights

    burn_hotkey = hotkeys[burn_uid]
    miner_weights = rolling_weights_for_hotkeys(
        hotkeys, history, gated_hotkeys=(gated_hotkeys or set()) | {burn_hotkey}
    )
    if sum(miner_weights) == 0:
        weights[burn_uid] = 1.0
        return weights

    # Payee selection already excludes the burn UID (a sink, never a miner) and every gated
    # hotkey, and assigns WINNER_WEIGHTS normalized over however many payees qualified --
    # so the row already sums to exactly 1.0 with nothing left to redistribute here.
    return miner_weights
