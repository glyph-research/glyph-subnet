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


def _prepare_stream(runner: CodecRunner, provider: CorpusProvider, spec: StreamSpec) -> StreamInput:
    """Build the stream input for ``spec``: a remote range source or inline bytes.

    A runner that fetches bytes itself (``prefers_remote_source``) gets a ``RangeSource`` when
    the corpus exposes one, so the validator never materializes the 256 MiB sample. Otherwise
    (local runner, or no published corpus URL) the bytes are read and inlined as before.
    """

    if getattr(runner, "prefers_remote_source", False):
        source = provider.stream_source(spec)
        if source is not None:
            return StreamInput(spec.stream_id, source=source)
    return StreamInput(spec.stream_id, data=provider.materialize(spec))


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
        stream_input = _prepare_stream(runner, provider, spec)
        try:
            results.append(runner.run_stream(artifact, stream_input, caps=caps))
        except RunnerError as exc:
            # A crashing entrypoint is a failed (non-bit-exact) stream; the codec is invalid.
            results.append(
                StreamResult(
                    stream_id=spec.stream_id,
                    raw_bytes=stream_input.raw_len,
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
