"""Beacon-seeded stream sampling (DESIGN §3.2, §5).

Every validator derives the identical paired sample from the same chain-beacon seed, so
no validator chooses the data. ``derive_seed`` mixes a post-corpus block hash with a
private validator salt and the round index; ``sample_streams`` turns the seed into a
deterministic set of long contiguous byte windows over the corpus.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from core.constants import STREAM_BYTES, STREAMS_PER_ROUND


@dataclass(frozen=True)
class StreamSpec:
    stream_id: str
    offset: int
    length: int


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


def sample_streams(
    seed: int,
    total_bytes: int,
    *,
    stream_bytes: int = STREAM_BYTES,
    streams: int = STREAMS_PER_ROUND,
) -> list[StreamSpec]:
    """Deterministically pick ``streams`` contiguous windows of ``stream_bytes``.

    Each window's start offset is derived from ``seed`` and the stream index, so the
    selection is identical across validators and stable across runs. Windows are long and
    contiguous (DESIGN §4) so online-learning codecs warm up within a stream.
    """

    if total_bytes <= 0 or streams <= 0:
        return []

    length = min(stream_bytes, total_bytes)
    max_offset = total_bytes - length
    seed_bytes = int(seed).to_bytes(8, "big")

    specs: list[StreamSpec] = []
    for index in range(streams):
        if max_offset == 0:
            offset = 0
        else:
            digest = hashlib.sha256(seed_bytes + index.to_bytes(4, "big")).digest()
            offset = int.from_bytes(digest, "big") % (max_offset + 1)
        specs.append(StreamSpec(stream_id=f"s{index}", offset=offset, length=length))
    return specs
