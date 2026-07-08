"""Beacon-seeded stream sampling.

``derive_seed`` mixes a post-corpus block hash with a per-validator **private salt** and the
round index; ``sample_source_streams`` turns the seed into a deterministic set of long
contiguous byte windows confined to one corpus source.

NOTE: because the salt is per-validator-private, validators currently sample *different*
windows -- the selection is identical run-to-run for a given validator, but NOT identical
across validators. Byte-identical-across-validators selection requires deriving from the
public beacon only; that is the direction of issue #22 (beacon-only per-round corpus +
window selection), pending owner sign-off on dropping the private salt from the data path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

@dataclass(frozen=True)
class StreamSpec:
    stream_id: str
    offset: int
    length: int
    source: str | None = None
    scored: bool = True


@dataclass(frozen=True)
class RangeSource:
    """Where a stream's bytes live remotely, so a worker can range-fetch them itself.

    The production Chutes path passes this instead of inlining the bytes: the deployed
    runner does an HTTP ``Range`` fetch of ``url`` for ``[offset, offset+length)``, mirroring
    the chute's ``StreamSource`` (url/offset/length). The validator never pulls (and re-uploads)
    the 256 MiB sample. ``offset``/``length`` are global byte coordinates into the corpus, so
    the published corpus must be one contiguous blob in the same order the manifest hashes.
    """

    url: str
    offset: int
    length: int


def derive_seed(block_hash: str, salt: str, round_index: int) -> int:
    payload = f"{block_hash}:{salt}:{round_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sample_source_streams(
    seed: int,
    source_start: int,
    source_bytes: int,
    *,
    stream_bytes: int,
    streams: int,
    source: str | None = None,
    scored: bool = True,
) -> list[StreamSpec]:
    """Pick ``streams`` windows of ``stream_bytes`` confined to a single source's byte range.

    Each window's start offset is derived from ``seed`` and the stream index (so selection is
    fresh per round and per validator via ``derive_seed``), confined to the source span
    ``[source_start, source_start + source_bytes)``. Returned offsets are global corpus
    coordinates. Used by the per-source eval (issue #10): two random 4 MiB FineWeb windows.
    """

    if source_bytes <= 0 or streams <= 0:
        return []

    length = min(stream_bytes, source_bytes)
    max_offset = source_bytes - length
    seed_bytes = int(seed).to_bytes(8, "big")

    specs: list[StreamSpec] = []
    prefix = source or "source"
    for index in range(streams):
        if max_offset == 0:
            local = 0
        else:
            digest = hashlib.sha256(seed_bytes + index.to_bytes(4, "big")).digest()
            local = int.from_bytes(digest, "big") % (max_offset + 1)
        specs.append(
            StreamSpec(
                stream_id=f"{prefix}-{index}",
                offset=source_start + local,
                length=length,
                source=source,
                scored=scored,
            )
        )
    return specs
