"""Rolling winner history and weight generation.

Glyph scores codecs by compression ratio, where **lower is better**, so the promotion
comparator is inverted relative to a benchmark-accuracy subnet. Weights follow a
two-slot rolling winner policy (current 70% / previous 30%), overlaid with the temporal
burn schedule: on a burn tempo all weight goes to the burn UID.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.constants import BURN_UID, WINNER_HISTORY_DEPTH, WINNER_LIMIT, WINNER_WEIGHTS


@dataclass(frozen=True)
class WinnerEntry:
    hotkey: str
    repo: str
    revision: str
    ratio: float  # compressed_bytes / raw_bytes; lower is better
    commit_block: int = 0  # earliest commit wins ties (makes copying worthless)

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
) -> list[WinnerEntry]:
    next_history = [winner, *history]
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


def select_payees(
    history: list[WinnerEntry],
    eligible_hotkeys: set[str],
    gated_hotkeys: set[str] | None = None,
    limit: int = WINNER_LIMIT,
) -> list[WinnerEntry]:
    """The winners paid this tempo: the first ``limit`` entries of ``history`` (newest
    first) that are both eligible and conviction-compliant (issue #170).

    Walking past a gated entry is what makes burning rare: a non-compliant current winner
    doesn't hand its share sideways, it shifts the whole ladder up (previous winner ->
    0.7, previous-previous -> 0.3, ...). Compliance is re-read every tempo, so a winner
    that locks re-enters the ladder at its own position at the next weight-setting, and a
    dethroned winner that stays locked keeps earning whenever someone above it lapses.

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
    if not payees:
        return [0.0 for _ in hotkeys]

    slot_weights = list(WINNER_WEIGHTS[: len(payees)])
    total = sum(slot_weights)
    normalized = [weight / total for weight in slot_weights]

    by_hotkey = {entry.hotkey: normalized[index] for index, entry in enumerate(payees)}
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

    On a burn tempo all weight goes to ``burn_uid``. On a normal tempo
    weight follows the rolling 70/30 winners; if there is no eligible winner yet the
    emission is burned rather than spread arbitrarily.

    ``gated_hotkeys`` (Miner Conviction, issues #141/#166/#170): winners whose conviction
    is below the requirement this tempo. They are skipped when picking payees, so the pot
    goes to the two most recent compliant winners in the retained history and burns only
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
