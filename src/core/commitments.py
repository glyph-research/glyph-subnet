"""Miner codec commitment parsing and serialization.

A Glyph commitment is a permanent, one-per-hotkey pointer to a codec artifact hosted
on HuggingFace, pinned to an immutable revision. The compact on-chain form is
``g1|repo|revision`` to stay under Bittensor metadata raw-length limits. The artifact
itself (compressor + decompressor) is fetched and hashed at precheck time; the sha256
is not stored on-chain in this version (rev-pin only).
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator

from core.constants import (
    COMMITMENT_PREFIX,
    COMMITMENT_VERSION,
    COMPACT_COMMITMENT_PREFIX,
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
        return value

    @field_validator("rev")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("revision is required")
        return value

    @property
    def key(self) -> str:
        return f"{self.repo}@{self.rev}"


@dataclass(frozen=True)
class ParsedCommitment:
    hotkey: str
    commitment: CodecCommitment
    raw: str


def serialize_commitment(commitment: CodecCommitment) -> str:
    """Serialize a commitment for Bittensor metadata storage."""

    return f"{COMPACT_COMMITMENT_PREFIX}{commitment.repo}|{commitment.rev}"


def parse_commitment(raw: str) -> CodecCommitment:
    """Parse a commitment string.

    The canonical format is ``g1|repo|revision``. Legacy ``glyph:{json}`` forms are
    accepted for migration flexibility.
    """

    data = raw.strip()
    if data.startswith(COMPACT_COMMITMENT_PREFIX):
        _, repo, revision = data.split("|", 2)
        return CodecCommitment(repo=repo, rev=revision)
    if data.startswith(COMMITMENT_PREFIX):
        data = data[len(COMMITMENT_PREFIX) :]
    import json

    payload = json.loads(data)
    commitment = CodecCommitment.model_validate(payload)
    if commitment.v != COMMITMENT_VERSION:
        raise ValueError(f"unsupported commitment version {commitment.v}")
    return commitment


def parse_commitments_by_hotkey(raw_commitments: dict[str, str]) -> list[ParsedCommitment]:
    parsed: list[ParsedCommitment] = []
    for hotkey, raw in raw_commitments.items():
        if not raw:
            continue
        try:
            parsed.append(
                ParsedCommitment(hotkey=hotkey, commitment=parse_commitment(raw), raw=raw)
            )
        except Exception:
            continue
    return parsed
