"""Compression scoring: ratio aggregation and the validity gates.

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
    source: str | None = None
    scored: bool = True

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


def stream_ratio(result: StreamResult) -> float:
    if result.raw_bytes <= 0:
        return float("inf")
    return result.compressed_bytes / result.raw_bytes


def source_ratio_breakdown(results: Sequence[StreamResult]) -> dict[str, float]:
    """Average per-stream ratios within each labeled scored source."""

    groups: dict[str, list[StreamResult]] = {}
    for result in results:
        if not result.scored:
            continue
        if result.source is None:
            continue
        groups.setdefault(result.source, []).append(result)
    return {
        source: sum(stream_ratio(result) for result in source_results) / len(source_results)
        for source, source_results in groups.items()
        if source_results
    }


def scored_ratio(results: Sequence[StreamResult]) -> float:
    """Final scored ratio: the flat, equally-weighted mean of each scored stream's ratio.

    Every scored stream counts the same regardless of which source it came from (issue
    #112). The previous mean-of-per-source-means rule weighted by *source* -- with the
    asymmetric 2x fineweb-edu / 1x pile mix, fineweb-edu's two streams would collectively
    count the same as pile's one. ``source_ratio_breakdown`` still exists for wandb
    per-source visibility; it is just no longer the aggregation step for the final score.
    Unlabeled streams keep the legacy pooled compressed/raw ratio.
    """

    scored = [result for result in results if result.scored]
    if not scored:
        return float("inf")
    if any(result.source is not None for result in scored):
        return sum(stream_ratio(result) for result in scored) / len(scored)
    return aggregate_ratio(scored)


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

    scored = [r for r in results if r.scored]
    if not scored:
        return CodecScore(False, float("inf"), 0.0, ["no scored streams evaluated"])

    failed = [r.stream_id for r in scored if not r.roundtrip_ok]
    if failed:
        reasons.append(f"round-trip failed on streams: {failed}")

    throughput_min = min(r.decompress_throughput_bps for r in scored)
    if throughput_min < floor_bps:
        reasons.append(
            f"decompress throughput {throughput_min:.0f} B/s below floor {floor_bps:.0f} B/s"
        )

    slow = [r.stream_id for r in scored if r.compress_secs > budget_secs]
    if slow:
        reasons.append(f"compress over budget ({budget_secs:.0f}s) on streams: {slow}")

    ratio = scored_ratio(results)
    return CodecScore(
        valid=not reasons,
        ratio=ratio,
        throughput_bps_min=throughput_min,
        reasons=reasons,
    )


def zstd_baseline_ratio(
    streams: Sequence[bytes],
    level: int = BASELINE_LEVEL,
    *,
    sources: Sequence[str | None] | None = None,
) -> float:
    """The zstd -19 vacant-crown floor, computed live on the round's streams.

    A codec must beat this ratio to take an empty crown.
    """

    import zstandard as zstd

    compressor = zstd.ZstdCompressor(level=level)
    results = [
        StreamResult(
            stream_id=f"zstd-{index}",
            raw_bytes=len(stream),
            compressed_bytes=len(compressor.compress(stream)),
            roundtrip_ok=True,
            compress_secs=0.0,
            decompress_secs=0.0,
            blob_hash="",
            source=sources[index] if sources is not None else None,
        )
        for index, stream in enumerate(streams)
    ]
    return scored_ratio(results)
