"""Compression scoring: ratio aggregation and the validity gates (DESIGN §3.3, §7).

Score = total compressed bytes / total raw bytes over the round's streams (lower is
better). A codec is *valid* only if it passes every gate; an invalid codec is never
promoted regardless of its nominal ratio.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from core.constants import BASELINE_LEVEL


@dataclass(frozen=True)
class StreamResult:
    stream_id: str
    raw_bytes: int
    compressed_bytes: int
    roundtrip_ok: bool
    compress_secs: float
    decompress_secs: float
    blob_hash: str

    @property
    def decompress_throughput_bps(self) -> float:
        if self.decompress_secs <= 0:
            return float("inf")
        return self.raw_bytes / self.decompress_secs


@dataclass
class CodecScore:
    valid: bool
    ratio: float
    throughput_bps_min: float
    reasons: list[str] = field(default_factory=list)


def aggregate_ratio(results: Sequence[StreamResult]) -> float:
    raw = sum(r.raw_bytes for r in results)
    compressed = sum(r.compressed_bytes for r in results)
    if raw <= 0:
        return float("inf")
    return compressed / raw


def score_codec(
    results: Sequence[StreamResult],
    *,
    floor_bps: float,
    budget_secs: float,
) -> CodecScore:
    """Apply the validity gates and compute the aggregate ratio.

    Gates (all must pass): bit-exact round-trip on *every* stream; minimum decompress
    throughput >= ``floor_bps``; per-stream compress wall-clock <= ``budget_secs``.
    """

    if not results:
        return CodecScore(False, float("inf"), 0.0, ["no streams evaluated"])

    reasons: list[str] = []

    failed = [r.stream_id for r in results if not r.roundtrip_ok]
    if failed:
        reasons.append(f"round-trip failed on streams: {failed}")

    throughput_min = min(r.decompress_throughput_bps for r in results)
    if throughput_min < floor_bps:
        reasons.append(
            f"decompress throughput {throughput_min:.0f} B/s below floor {floor_bps:.0f} B/s"
        )

    slow = [r.stream_id for r in results if r.compress_secs > budget_secs]
    if slow:
        reasons.append(f"compress over budget ({budget_secs:.0f}s) on streams: {slow}")

    ratio = aggregate_ratio(results)
    return CodecScore(
        valid=not reasons,
        ratio=ratio,
        throughput_bps_min=throughput_min,
        reasons=reasons,
    )


def zstd_baseline_ratio(streams: Sequence[bytes], level: int = BASELINE_LEVEL) -> float:
    """The zstd -19 vacant-crown floor, computed live on the round's streams.

    A codec must beat this ratio to take an empty crown (DESIGN §3.3).
    """

    import zstandard as zstd

    compressor = zstd.ZstdCompressor(level=level)
    raw = sum(len(s) for s in streams)
    compressed = sum(len(compressor.compress(s)) for s in streams)
    if raw <= 0:
        return float("inf")
    return compressed / raw
