"""Temporal burn schedule (DESIGN §6.1).

25% daily burn, applied temporally rather than as a static weight carve-out: tempos are
grouped into windows of ``BURN_WINDOW_TEMPOS`` (4). Exactly one tempo per window is a
burn tempo at an unpredictable position derived from a seed ``S`` that comes from the
most recent challenge round's evaluation outputs -- so only validators that actually ran
(or replayed) the jobs can compute the schedule. Combined with native commit-reveal
weights, a copy-cat validator cannot reproduce the schedule and diverges maximally on
the tempos it gets wrong, which Yuma bonding punishes hard.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from core.constants import BURN_WINDOW_TEMPOS, WINDOW_ANCHOR_BLOCK

BurnOutput = tuple[object, int, str]  # (stream_id, compressed_bytes, blob_sha256_hex)
_BOOTSTRAP = b"glyph-burn-bootstrap-v1"


def derive_burn_seed(
    last_round_outputs: Iterable[BurnOutput] | None,
    *,
    bootstrap: bytes = _BOOTSTRAP,
) -> bytes:
    """Derive seed ``S`` from the most recent challenge round's per-stream outputs.

    Honest validators ran the same beacon-seeded jobs and therefore compute the same
    ``S``. Before the first challenge round there are no outputs, so a fixed bootstrap
    seed keeps the schedule deterministic (DESIGN §6.1 edge case).
    """

    outputs = list(last_round_outputs or [])
    if not outputs:
        return hashlib.sha256(bootstrap).digest()

    digest = hashlib.sha256()
    for stream_id, size, blob_hash in sorted(outputs, key=lambda item: str(item[0])):
        digest.update(f"{stream_id}|{int(size)}|{blob_hash};".encode())
    return digest.digest()


def tempo_index(block: int, tempo: int, anchor: int = WINDOW_ANCHOR_BLOCK) -> int:
    if tempo <= 0:
        raise ValueError("tempo must be positive")
    return max(0, block - anchor) // tempo


def window_index(block: int, tempo: int, anchor: int = WINDOW_ANCHOR_BLOCK) -> int:
    return tempo_index(block, tempo, anchor) // BURN_WINDOW_TEMPOS


def burn_position(seed: bytes, window: int) -> int:
    """The 0..BURN_WINDOW_TEMPOS-1 position that is the burn tempo for ``window``."""

    payload = seed + int(window).to_bytes(8, "big")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big") % BURN_WINDOW_TEMPOS


def is_burn_tempo(
    block: int,
    tempo: int,
    seed: bytes,
    anchor: int = WINDOW_ANCHOR_BLOCK,
) -> bool:
    idx = tempo_index(block, tempo, anchor)
    window = idx // BURN_WINDOW_TEMPOS
    position = idx % BURN_WINDOW_TEMPOS
    return position == burn_position(seed, window)
