"""Persistent validator state."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

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
    """

    from core.constants import SCORING_VERSION

    stale_keys = [key for key, score in state.scores.items() if score.scoring_version != SCORING_VERSION]
    if not stale_keys:
        return
    for key in stale_keys:
        del state.scores[key]
    state.excluded_hotkeys.clear()


def load_state(path: Path) -> ValidatorState:
    if not path.exists():
        return ValidatorState()
    state = ValidatorState.model_validate_json(path.read_text())
    _invalidate_stale_scores(state)
    return state


def save_state(path: Path, state: ValidatorState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True))
