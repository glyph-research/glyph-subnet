"""Persistent validator state."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

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

    def eligible_hotkeys(self) -> set[str]:
        return {item.hotkey for item in self.commitments.values() if item.valid}


def load_state(path: Path) -> ValidatorState:
    if not path.exists():
        return ValidatorState()
    return ValidatorState.model_validate_json(path.read_text())


def save_state(path: Path, state: ValidatorState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True))
