"""Paired evaluation of codecs over identical beacon-seeded streams (DESIGN §3.2).

The incumbent and every challenger are run on the *same* streams in a round, so shared
data difficulty cancels and one-shot scoring is fair. Each stream's compress+decompress
runs as one same-worker job via a ``CodecRunner``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from eval.corpus import CorpusProvider
from eval.runner import ArtifactRef, CodecRunner, ResourceCaps, RunnerError, StreamInput
from eval.scoring import CodecScore, StreamResult, score_codec
from eval.streams import StreamSpec


@dataclass
class EvalOutcome:
    hotkey: str
    score: CodecScore
    results: list[StreamResult] = field(default_factory=list)
    error: str | None = None

    def burn_outputs(self) -> list[tuple[str, int, str]]:
        """Per-stream (id, compressed_bytes, blob_hash) -- the burn-seed material."""

        return [(r.stream_id, r.compressed_bytes, r.blob_hash) for r in self.results]


def evaluate_artifact(
    runner: CodecRunner,
    hotkey: str,
    artifact: ArtifactRef,
    provider: CorpusProvider,
    stream_specs: Sequence[StreamSpec],
    *,
    caps: ResourceCaps,
    floor_bps: float,
    budget_secs: float,
) -> EvalOutcome:
    results: list[StreamResult] = []
    for spec in stream_specs:
        data = provider.materialize(spec)
        try:
            results.append(runner.run_stream(artifact, StreamInput(spec.stream_id, data), caps=caps))
        except RunnerError as exc:
            # A crashing entrypoint is a failed (non-bit-exact) stream; the codec is invalid.
            results.append(
                StreamResult(
                    stream_id=spec.stream_id,
                    raw_bytes=len(data),
                    compressed_bytes=0,
                    roundtrip_ok=False,
                    compress_secs=0.0,
                    decompress_secs=0.0,
                    blob_hash="",
                )
            )
            score = score_codec(results, floor_bps=floor_bps, budget_secs=budget_secs)
            return EvalOutcome(hotkey, score, results, error=str(exc))

    score = score_codec(results, floor_bps=floor_bps, budget_secs=budget_secs)
    return EvalOutcome(hotkey, score, results)


def paired_eval(
    runner: CodecRunner,
    artifacts: Sequence[tuple[str, ArtifactRef]],
    provider: CorpusProvider,
    stream_specs: Sequence[StreamSpec],
    *,
    caps: ResourceCaps,
    floor_bps: float,
    budget_secs: float,
) -> dict[str, EvalOutcome]:
    return {
        hotkey: evaluate_artifact(
            runner,
            hotkey,
            artifact,
            provider,
            stream_specs,
            caps=caps,
            floor_bps=floor_bps,
            budget_secs=budget_secs,
        )
        for hotkey, artifact in artifacts
    }
