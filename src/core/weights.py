"""Rolling winner history and weight generation.

Glyph scores codecs by compression ratio, where **lower is better**, so the promotion
comparator is inverted relative to a benchmark-accuracy subnet. Weights follow a
two-slot rolling winner policy (current 70% / previous 30%), overlaid with the temporal
burn schedule: on a burn tempo all weight goes to the burn UID.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.constants import BURN_UID, WINNER_LIMIT, WINNER_WEIGHTS


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
    limit: int = WINNER_LIMIT,
) -> list[WinnerEntry]:
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


def rolling_weights_for_hotkeys(
    hotkeys: list[str],
    history: list[WinnerEntry],
) -> list[float]:
    compacted = compact_history(history, eligible_hotkeys=set(hotkeys))
    if not compacted:
        return [0.0 for _ in hotkeys]

    slot_weights = list(WINNER_WEIGHTS[: len(compacted)])
    total = sum(slot_weights)
    normalized = [weight / total for weight in slot_weights]

    by_hotkey = {entry.hotkey: normalized[index] for index, entry in enumerate(compacted)}
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

    ``gated_hotkeys`` (Miner Conviction, issues #141/#166): winners whose conviction is
    below the requirement this tempo. A gated slot's share reallocates to the remaining
    compliant winner slot(s), renormalized over them; only when every occupied slot is
    gated does the whole winner pot burn (owner decision, issue #166: neither slot has
    any lever over the other's conviction -- compliance is purely the other party's own
    lock -- so reallocation is not an attack surface, and emission keeps flowing to
    compliant winners instead of burning). Reversible by construction: the slot itself is
    untouched, so a winner earns again at the next weight-setting after locking.
    """

    if not 0 <= burn_uid < len(hotkeys):
        raise ValueError("burn_uid is outside the hotkey list")

    weights = [0.0 for _ in hotkeys]
    if is_burn_tempo:
        weights[burn_uid] = 1.0
        return weights

    miner_weights = rolling_weights_for_hotkeys(hotkeys, history)
    if sum(miner_weights) == 0:
        weights[burn_uid] = 1.0
        return weights

    # The burn UID is a sink, never a miner: zero it even if it somehow entered history.
    for index in range(len(hotkeys)):
        weights[index] = 0.0 if index == burn_uid else miner_weights[index]
    total = sum(weights)
    if total <= 0:
        weights[burn_uid] = 1.0
        return weights
    weights = [weight / total for weight in weights]
    if gated_hotkeys:
        gated_mass = 0.0
        for index, hotkey in enumerate(hotkeys):
            if hotkey in gated_hotkeys and index != burn_uid and weights[index] > 0:
                gated_mass += weights[index]
                weights[index] = 0.0
        if gated_mass > 0:
            remaining = sum(w for i, w in enumerate(weights) if i != burn_uid)
            if remaining > 0:
                scale = (remaining + gated_mass) / remaining
                weights = [w * scale if i != burn_uid else w for i, w in enumerate(weights)]
            else:
                weights[burn_uid] += gated_mass
    return weights
