"""Miner codec commitment parsing and serialization.

A Glyph commitment is a permanent, one-per-hotkey pointer to a codec artifact hosted
on HuggingFace, pinned to an immutable revision. The compact on-chain form is
``g1|repo|revision`` to stay under Bittensor metadata raw-length limits. The artifact
itself (compressor + decompressor) is fetched and hashed at precheck time; the sha256
is not stored on-chain in this version (rev-pin only).

To stop commit front-running (exploit vector #9) the codec is published in two phases on
the same hotkey, one block apart:

* **commit phase** -- ``g1c|<sha256(repo|rev|salt)>``. This reveals nothing about the repo,
  so a mempool watcher cannot copy it.
* **reveal phase** -- ``g1r|repo|revision|salt``. Validators recompute the digest and match
  it against the commit-phase value they observed, and key the earliest-commit tie-break
  (DESIGN §3.5) off the *commit-phase* block. A copier who only learns ``repo|rev`` at reveal
  time therefore cannot land an earlier commit.

Legacy single-phase ``g1|repo|rev`` / ``glyph:{json}`` commitments are still parsed (they
simply carry no front-running protection and tie-break on first observation).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator

from core.constants import (
    COMMIT_PHASE_PREFIX,
    COMMITMENT_PREFIX,
    COMMITMENT_VERSION,
    COMPACT_COMMITMENT_PREFIX,
    REVEAL_PHASE_PREFIX,
)


class CodecCommitment(BaseModel):
    """Compact on-chain commitment payload."""

    v: int = Field(default=COMMITMENT_VERSION)
    repo: str = Field(min_length=3, max_length=160)
    rev: str = Field(min_length=6, max_length=80)

    @field_validator("repo")
    @classmethod
    def validate_repo(cls, value: str) -> str:
        value = value.strip()
        parts = value.split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError("repo must be a HuggingFace repo id like namespace/name")
        if any(part in {".", ".."} for part in parts):
            raise ValueError("repo contains invalid path segments")
        if "|" in value:
            raise ValueError("repo must not contain '|'")
        return value

    @field_validator("rev")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("revision is required")
        if "|" in value:
            raise ValueError("revision must not contain '|'")
        return value

    @property
    def key(self) -> str:
        return f"{self.repo}@{self.rev}"


@dataclass(frozen=True)
class ParsedCommitment:
    hotkey: str
    commitment: CodecCommitment
    raw: str
    # Populated for two-phase (reveal) commitments so the validator can match the reveal
    # against the commit-phase digest it observed and key the tie-break off the commit block.
    salt: str | None = None
    digest: str | None = None


def commitment_digest(repo: str, rev: str, salt: str) -> str:
    """The commit-phase digest binding ``repo``, ``rev`` and a per-commit ``salt``."""

    return hashlib.sha256(f"{repo}|{rev}|{salt}".encode()).hexdigest()


def serialize_commitment(commitment: CodecCommitment) -> str:
    """Serialize a legacy single-phase commitment for Bittensor metadata storage."""

    return f"{COMPACT_COMMITMENT_PREFIX}{commitment.repo}|{commitment.rev}"


def serialize_commit_phase(digest: str) -> str:
    """Phase 1: the hiding commitment (publishes only the digest)."""

    return f"{COMMIT_PHASE_PREFIX}{digest}"


def serialize_reveal_phase(repo: str, rev: str, salt: str) -> str:
    """Phase 2: the opening commitment (publishes repo|rev|salt)."""

    return f"{REVEAL_PHASE_PREFIX}{repo}|{rev}|{salt}"


def parse_commit_phase_digest(raw: str) -> str | None:
    """Return the digest if ``raw`` is a commit-phase value, else None."""

    data = raw.strip()
    if data.startswith(COMMIT_PHASE_PREFIX):
        return data[len(COMMIT_PHASE_PREFIX) :]
    return None


def parse_commitment(raw: str) -> tuple[CodecCommitment, str | None]:
    """Parse a *revealed* commitment string into ``(commitment, salt)``.

    Handles the two-phase reveal form ``g1r|repo|rev|salt`` (salt returned), the legacy
    compact ``g1|repo|revision`` and ``glyph:{json}`` forms (salt None). Commit-phase
    ``g1c|...`` values are not revealed commitments and raise.
    """

    data = raw.strip()
    if data.startswith(COMMIT_PHASE_PREFIX):
        raise ValueError("commit-phase digest is not a revealed commitment")
    if data.startswith(REVEAL_PHASE_PREFIX):
        _, repo, rev, salt = data.split("|", 3)
        return CodecCommitment(repo=repo, rev=rev), salt
    if data.startswith(COMPACT_COMMITMENT_PREFIX):
        _, repo, revision = data.split("|", 2)
        return CodecCommitment(repo=repo, rev=revision), None
    if data.startswith(COMMITMENT_PREFIX):
        data = data[len(COMMITMENT_PREFIX) :]
    import json

    payload = json.loads(data)
    commitment = CodecCommitment.model_validate(payload)
    if commitment.v != COMMITMENT_VERSION:
        raise ValueError(f"unsupported commitment version {commitment.v}")
    return commitment, None


def prune_commit_phase_seen(
    seen: dict[str, dict[str, int]], current_block: int, max_age_blocks: int
) -> int:
    """Drop commit-phase digests older than ``max_age_blocks`` and empty hotkey entries.

    Bounds the persisted ``commit_phase_seen`` map: a digest whose reveal never arrives
    (abandoned commit) is aged out, and resolved digests are removed by the validator when
    it matches a reveal. Returns the number of digests removed.
    """

    removed = 0
    for hotkey in list(seen):
        digests = seen[hotkey]
        for digest in list(digests):
            if current_block - digests[digest] > max_age_blocks:
                del digests[digest]
                removed += 1
        if not digests:
            del seen[hotkey]
    return removed


def parse_commit_phase_by_hotkey(raw_commitments: dict[str, str]) -> dict[str, str]:
    """Map hotkey -> commit-phase digest for hotkeys currently in the commit phase."""

    out: dict[str, str] = {}
    for hotkey, raw in raw_commitments.items():
        if not raw:
            continue
        digest = parse_commit_phase_digest(raw)
        if digest:
            out[hotkey] = digest
    return out


def parse_commitments_by_hotkey(raw_commitments: dict[str, str]) -> list[ParsedCommitment]:
    """Parse all *revealed* commitments. Commit-phase entries are skipped here."""

    parsed: list[ParsedCommitment] = []
    for hotkey, raw in raw_commitments.items():
        if not raw:
            continue
        try:
            commitment, salt = parse_commitment(raw)
        except Exception:
            continue
        digest = (
            commitment_digest(commitment.repo, commitment.rev, salt) if salt is not None else None
        )
        parsed.append(
            ParsedCommitment(
                hotkey=hotkey, commitment=commitment, raw=raw, salt=salt, digest=digest
            )
        )
    return parsed
