"""Persistent validator state."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from core.constants import DEFAULT_WIN_MARGIN, STATE_BACKUP_COUNT
from core.conviction import ConvictionLedger
from core.weights import WinnerEntry


class CommitmentState(BaseModel):
    hotkey: str
    repo: str
    revision: str
    block: int | None = None
    artifact_hash: str | None = None
    artifact_bytes: int | None = None
    valid: bool = False
    disqualification_reason: str | None = None
    # Local codec dir used for evaluation when HF download is bypassed (testnet/CI).
    local_path: str | None = None
    # Block at which the full security scan + artifact hash last actually ran (as opposed to
    # a cheap manifest-only re-check) -- issue #96's periodic re-verification cadence.
    last_full_check_block: int | None = None
    # Consecutive prechecks where the HF repo/revision was *genuinely gone* (404 -- deleted,
    # renamed, made private), as opposed to transiently unreachable. Reset to 0 by any
    # successful precheck or any differently-failing one; crossing
    # ``REPO_NOT_FOUND_EXCLUDE_STREAK`` triggers one-shot exclusion (issue #128).
    consecutive_repo_not_found: int = 0
    # True when this round's invalidity is only an HF fetch failure that is not (yet)
    # confirmed permanent -- a 5xx/timeout/DNS blip, or a 404 still under the #128 streak
    # threshold. Such a commitment must not cost its hotkey the crown (issue #135: a
    # 2-minute HF outage permanently dethroned the live champion).
    transiently_unreachable: bool = False

    @property
    def key(self) -> str:
        return f"{self.hotkey}:{self.repo}@{self.revision}"


class ScoreState(BaseModel):
    hotkey: str
    repo: str
    revision: str
    ratio: float  # compressed/raw; lower is better
    roundtrip_ok: bool
    throughput_bps: float
    valid: bool
    commit_block: int = 0
    evaluated_at_block: int | None = None
    # Stamped at evaluation time against core.constants.SCORING_VERSION (issue #104).
    # Defaults to 0 (never matches a real SCORING_VERSION, which starts at 1) so state
    # persisted before this field existed is treated as stale rather than trusted forever.
    scoring_version: int = 0

    def as_winner(self) -> WinnerEntry:
        return WinnerEntry(
            hotkey=self.hotkey,
            repo=self.repo,
            revision=self.revision,
            ratio=self.ratio,
            commit_block=self.commit_block,
        )


class ValidatorState(BaseModel):
    commitments: dict[str, CommitmentState] = Field(default_factory=dict)
    scores: dict[str, ScoreState] = Field(default_factory=dict)
    winner_history: list[WinnerEntry] = Field(default_factory=list)
    duplicate_hash_owner: dict[str, str] = Field(default_factory=dict)
    # Per-stream (stream_id, compressed_bytes, blob_sha256) of the most recent challenge
    # round's winner -- the material the temporal burn seed is derived from.
    last_round_outputs: list[tuple[str, int, str]] = Field(default_factory=list)
    # One-shot losers: a challenger that did not win is excluded from future rounds.
    excluded_hotkeys: set[str] = Field(default_factory=set)
    # Commit-phase digests observed on-chain and the block we first saw them at, per hotkey
    # ({hotkey: {digest: block}}). A two-phase reveal tie-breaks off the commit-phase block
    # recorded here, defeating commit front-running (exploit vector #9).
    commit_phase_seen: dict[str, dict[str, int]] = Field(default_factory=dict)
    # Block this validator anchored its burn-window origin to (persisted for stability).
    window_anchor_block: int | None = None
    # Miner Conviction earnings ledger (issue #141) -- cumulative per-hotkey alpha,
    # accumulated on the fixed CONVICTION_TRACKING_START_BLOCK grid. Permanent per hotkey:
    # dethrone-and-return never resets it.
    conviction_ledger: ConvictionLedger = Field(default_factory=ConvictionLedger)

    def eligible_hotkeys(self) -> set[str]:
        return {item.hotkey for item in self.commitments.values() if item.valid}

    def retained_hotkeys(self) -> set[str]:
        """Hotkeys whose ``winner_history`` entries must survive this round (issue #135).

        Everything currently valid, plus commitments that are invalid only because their
        repo is transiently unreachable -- a champion must lose the crown only to a genuine
        re-eval failure, a confirmed-permanent 404 (the #128 streak -> ``excluded_hotkeys``),
        or a content disqualification; never to an HF outage.
        """

        return {
            item.hotkey
            for item in self.commitments.values()
            if (item.valid or item.transiently_unreachable)
            and item.hotkey not in self.excluded_hotkeys
        }


def _invalidate_stale_scores(state: ValidatorState) -> None:
    """Drop any ``ScoreState`` computed under an older ``SCORING_VERSION`` and clear
    one-shot exclusions decided under that same stale regime (issue #104).

    ``SCORING_VERSION`` is a single network-wide constant, not per-hotkey, so a bump is an
    all-or-nothing event: every already-recorded score is equally stale, and every exclusion
    decided against those old numbers shouldn't outlive them either -- "everyone competes
    fresh," not "everyone except whoever already lost under the old math." Dropping a
    hotkey's score is enough on its own to re-admit it to the challenger filter
    (``c.key not in state.scores``); the reigning incumbent is unaffected either way since
    it's already unconditionally re-evaluated every round regardless of this cache.

    Exception (issue #143): a bump listed in ``SCORING_VERSION_START_BLOCKS`` is
    score-compatible -- ratio semantics unchanged, only round policy/ordering moved -- so
    older-version scores evaluated *before* that transition stay retained and directly
    comparable (see ``score_is_comparable``), and the exclusions decided alongside them
    survive too. Exclusions are cleared only on a genuine surface wipe, which is precisely
    what re-admitting the whole board is for.
    """

    stale_keys = []
    surface_wipe = False
    for key, score in state.scores.items():
        if score_is_comparable(score):
            continue
        stale_keys.append(key)
        if _compatible_transition_start(score.scoring_version) is None:
            surface_wipe = True  # a genuinely incompatible (surface-changing) transition
    if not stale_keys:
        return
    for key in stale_keys:
        del state.scores[key]
    if surface_wipe:
        state.excluded_hotkeys.clear()


def _compatible_transition_start(from_version: int) -> int | None:
    """Earliest start block of the ``from_version -> SCORING_VERSION`` transition, or
    ``None`` unless EVERY intermediate bump is listed as score-compatible.

    Chained on purpose: a v3 start-block entry must not let a v1 score (recorded before
    the surface-changing v1->v2 corpus overhaul) ride through -- each step has to declare
    compatibility for the whole path to be safe.
    """

    from core.constants import SCORING_VERSION, SCORING_VERSION_START_BLOCKS

    if not 0 <= from_version < SCORING_VERSION:
        return None
    blocks = [SCORING_VERSION_START_BLOCKS.get(v) for v in range(from_version + 1, SCORING_VERSION + 1)]
    if any(block is None for block in blocks):
        return None
    return min(blocks)


def score_is_comparable(score: ScoreState) -> bool:
    """True when this persisted score may be compared directly against current-version
    scores (issue #143): stamped with the current ``SCORING_VERSION``, or retained under a
    score-compatible transition chain and evaluated before it began."""

    from core.constants import SCORING_VERSION

    if score.scoring_version == SCORING_VERSION:
        return True
    start_block = _compatible_transition_start(score.scoring_version)
    return start_block is not None and (score.evaluated_at_block or 0) < start_block


def backfill_improvements(state: ValidatorState, unstamped: list[bool]) -> None:
    """Recover ``improvement`` for winner entries persisted before #177 (issue #180).

    Each entry dethroned the next one down the retained history, so its real improvement is
    ``(next.ratio - this.ratio) / next.ratio`` -- both ratios are already persisted, so this
    is a read of existing state, never a re-evaluation. Every validator holds the same
    ratios and runs the same arithmetic, so the result is identical fleet-wide.

    ``unstamped[i]`` marks entries whose JSON carried no ``improvement`` field at all. That
    distinction matters: once #177 is live every promotion stamps a value, and a genuine 1%
    dethrone must never be "corrected" into something else. The oldest retained entry is the
    one case with no successor to compare against -- it dethroned a winner we no longer
    retain -- so it keeps ``DEFAULT_WIN_MARGIN`` as the only real fallback in the system.
    """

    history = state.winner_history
    for index, entry in enumerate(history):
        if index >= len(unstamped) or not unstamped[index]:
            continue
        if index + 1 < len(history):
            dethroned = history[index + 1].ratio
            improvement = max(0.0, (dethroned - entry.ratio) / dethroned) if dethroned > 0 else 0.0
        else:
            improvement = DEFAULT_WIN_MARGIN
        history[index] = replace(entry, improvement=improvement)


STATE_BACKUP_SUFFIX = ".bak-"


def _log(message: str, *, level: str = "warning") -> None:
    """Log through bittensor's logger, imported lazily so ``core.state`` stays importable
    (and cheap) without the SDK. Indirected through one function so tests can capture what
    an operator would actually see on a recovery."""

    from bittensor.utils.btlogging import logging as bt_logging

    getattr(bt_logging, level)(message)


def _backup_paths(path: Path) -> list[Path]:
    """Existing rotated backups of ``path``, newest first.

    Ordered by filename: the stamp is fixed-width UTC, so lexicographic order is
    chronological order, and unlike mtime it survives an operator copying files around.
    """

    return sorted(path.parent.glob(f"{path.name}{STATE_BACKUP_SUFFIX}*"), reverse=True)


def _rotate_backups(path: Path, payload: str) -> None:
    """Keep the newest ``STATE_BACKUP_COUNT`` snapshots of successful saves (issue #187).

    Written from the payload just committed, so every backup is a known-good save rather
    than a copy of whatever happened to be on disk. A backup is a convenience, never a
    correctness requirement: failing to write or prune one must not fail the save that
    already succeeded, so this only logs.
    """

    if STATE_BACKUP_COUNT <= 0:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    backup = path.with_name(f"{path.name}{STATE_BACKUP_SUFFIX}{stamp}")
    try:
        backup.write_text(payload)
        for stale in _backup_paths(path)[STATE_BACKUP_COUNT:]:
            stale.unlink()
    except OSError as exc:
        _log(f"could not write/prune state backup {backup.name}: {exc}")


def _load_from(path: Path) -> ValidatorState | None:
    """Parse one state file, or ``None`` if it is unreadable, empty, or corrupt.

    Returning ``None`` rather than raising is what lets ``load_state`` walk to the next
    candidate: an empty file is exactly the corruption #187 hit, and it must be survivable.
    """

    try:
        raw = path.read_text()
    except OSError:
        return None
    if not raw.strip():
        return None
    try:
        # pydantic's ValidationError subclasses ValueError, so this covers both a malformed
        # document and a well-formed one that isn't a ValidatorState.
        state = ValidatorState.model_validate_json(raw)
    except ValueError:
        return None
    # Which winner entries predate #177's improvement field, as written on disk -- pydantic
    # has already filled the default in by the time we see the model, so absence can only
    # be observed in the raw JSON (issue #180).
    try:
        persisted = json.loads(raw).get("winner_history") or []
    except (ValueError, AttributeError):
        persisted = []
    backfill_improvements(
        state, [isinstance(e, dict) and "improvement" not in e for e in persisted]
    )
    _invalidate_stale_scores(state)
    return state


def load_state(path: Path) -> ValidatorState:
    """Load validator state, degrading rather than crashing on a damaged file (issue #187).

    A validator that starts with a loud warning beats one that cannot start at all: the
    production incident was a 0-byte state file after an auto-update restart, which raised
    out of here and crash-looped every round and every commit-phase poll. Note the asymmetry
    that fix removes -- a *missing* file has always started fine, so the more likely
    corruption must not be the less survivable one.

    Order of preference: the state file, then the newest usable rotated backup, then a fresh
    ``ValidatorState``. Every fallback is logged at ERROR, because each one means real lost
    work (scores, exclusions, commitments) that an operator needs to know about -- the
    conviction ledger is the one part that always rebuilds deterministically from the
    archive.
    """

    if not path.exists():
        return ValidatorState()
    state = _load_from(path)
    if state is not None:
        return state

    _log(f"validator state at {path} is empty or corrupt -- trying rotated backups", level="error")
    for backup in _backup_paths(path):
        recovered = _load_from(backup)
        if recovered is not None:
            _log(
                f"recovered validator state from backup {backup.name} -- anything scored "
                f"since that backup was written is lost and will be re-earned",
                level="error",
            )
            return recovered
    _log(
        "no usable validator state backup found -- starting from empty state; scores, "
        "exclusions and commitments will be rebuilt from scratch (the conviction ledger "
        "rebuilds deterministically from the archive)",
        level="error",
    )
    return ValidatorState()


def save_state(path: Path, state: ValidatorState) -> None:
    """Persist ``state`` atomically, then refresh the rolling backups (issue #187).

    This used to be a bare ``Path.write_text``, which truncates the file to zero and *then*
    writes: any process death inside that window left a 0-byte state file behind. The
    auto-updater restarting the validator on a version bump is exactly such an event, so
    every operator was exposed on every release -- a fleet-wide hazard on a timer rather
    than bad luck.

    Writing a sibling temp file and ``os.replace``-ing it into place is atomic on POSIX
    within a filesystem: a restart sees either the complete old file or the complete new
    one, never an empty one. The temp file is a sibling deliberately -- ``os.replace``
    across filesystems is not atomic.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w") as handle:
        handle.write(payload)
        handle.flush()
        # replace() orders the rename, it does not guarantee the bytes reached disk first;
        # fsync buys durability against machine-level power loss too, not just the process
        # death that actually bit us.
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _rotate_backups(path, payload)
