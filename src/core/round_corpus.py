"""Beacon-derived per-round corpus selection (issue #22).

Validators must evaluate on fresh data each round, not a fixed corpus. Per-round data is
selected from large, immutably-pinned public datasets using ONLY the public round beacon, so
every validator reconstructs byte-identical data for a given beacon -- the same cross-validator
guarantee ``eval/streams.py`` is meant to give for windows.

This module is the deterministic, beacon-only *selection* core. Turning a selection into bytes
(HF Parquet row-group / byte-range addressing of the pinned snapshots) is layered on top in a
follow-up; keeping selection pure and beacon-only makes the consensus-critical part unit-testable
without network or GPU.

Consensus note: selection takes **no validator-private input** (no salt). Any private input
would make validators derive different corpora and split consensus. (Removing the private salt
from the existing window-sampling path is gated on owner sign-off; this module never adds one.)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# Per-chunk size for the per-round corpus (matches the launch 8x4 MiB shape).
CHUNK_BYTES = 4 * 2**20


@dataclass(frozen=True)
class SourceSpec:
    """A pinned dataset source and how many chunks it contributes to each round's corpus."""

    name: str
    dataset: str
    revision: str  # immutable HF commit sha -- pinned once, identical for every validator
    config: str | None
    chunks: int


# The launch mix, pinned to immutable revisions (3x FineWeb / 3x Pile / 2x enwik9). Changing a
# revision is a coordinated source-spec update, committed once on-chain -- never per round.
SOURCE_SPEC: tuple[SourceSpec, ...] = (
    SourceSpec("fineweb", "HuggingFaceFW/fineweb", "9bb295ddab0e05d785b879661af7260fed5140fc", "sample-10BT", 3),
    SourceSpec("pile", "monology/pile-uncopyrighted", "3be90335b66f24456a5d6659d9c8d208c0357119", None, 3),
    SourceSpec("enwik9", "haukur/enwik9", "15d4ab0113acefe259d989ead474922f08c37f9e", None, 2),
)


@dataclass(frozen=True)
class ChunkLocator:
    """A single per-round chunk's source + the beacon-derived seed the provider resolves to bytes."""

    source: str
    dataset: str
    revision: str
    config: str | None
    chunk_index: int  # 0..chunks-1 within this source
    seed_hex: str  # beacon-only seed; provider maps it to (file, row-group, byte offset)


def round_seed(beacon: str, source: str, chunk_index: int) -> bytes:
    """Deterministic per-(beacon, source, chunk) seed. Beacon-only -- no private input."""

    return hashlib.sha256(f"{beacon}|{source}|{chunk_index}".encode()).digest()


def select_index(seed: bytes, count: int) -> int:
    """A deterministic index in ``[0, count)`` derived from ``seed`` (uniform via wide reduction)."""

    if count <= 0:
        raise ValueError("count must be positive")
    return int.from_bytes(seed, "big") % count


def position_keys(seed: bytes) -> tuple[int, int, int]:
    """Raw 64-bit (file_key, row_group_key, offset_key) from a chunk seed.

    The provider applies ``% count`` as each count becomes known (the row-group count depends on
    which file was picked), so picking the file and the row-group can happen in stages while
    staying deterministic in ``seed``.
    """

    h = hashlib.sha256(seed).digest()
    return (
        int.from_bytes(h[0:8], "big"),
        int.from_bytes(h[8:16], "big"),
        int.from_bytes(h[16:24], "big"),
    )


def resolve_position(seed: bytes, num_files: int, num_row_groups: int) -> tuple[int, int, int]:
    """Map a chunk seed to a concrete (file_index, row_group_index, offset_key).

    ``num_files`` / ``num_row_groups`` come from the pinned snapshot's Parquet metadata (fixed at
    the pinned revision, so identical across validators). ``offset_key`` seeds the byte offset the
    provider uses within the decoded row-group. All three are deterministic in ``seed``.
    """

    if num_files <= 0 or num_row_groups <= 0:
        raise ValueError("num_files and num_row_groups must be positive")
    file_key, row_group_key, offset_key = position_keys(seed)
    return file_key % num_files, row_group_key % num_row_groups, offset_key


def derive_round_corpus(beacon: str, spec: tuple[SourceSpec, ...] = SOURCE_SPEC) -> list[ChunkLocator]:
    """The per-round chunk locators for ``beacon``: beacon-only, deterministic, order-stable.

    The provider resolves each locator to bytes via HF range reads against the pinned revision;
    concatenated in this order they form the round's corpus (same shape as the offline launch
    corpus, but freshly selected every beacon).
    """

    locators: list[ChunkLocator] = []
    for source in spec:
        for chunk_index in range(source.chunks):
            seed = round_seed(beacon, source.name, chunk_index)
            locators.append(
                ChunkLocator(
                    source=source.name,
                    dataset=source.dataset,
                    revision=source.revision,
                    config=source.config,
                    chunk_index=chunk_index,
                    seed_hex=seed.hex(),
                )
            )
    return locators
