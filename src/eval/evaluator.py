"""Paired evaluation of codecs over identical beacon-seeded streams.

The incumbent and every challenger are run on the *same* streams in a round, so shared
data difficulty cancels and one-shot scoring is fair. Each stream's compress+decompress
runs as one same-worker job via a ``CodecRunner``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

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

        return [(r.stream_id, r.compressed_bytes, r.blob_hash) for r in self.results if r.scored]


def _prepare_stream(runner: CodecRunner, provider: CorpusProvider, spec: StreamSpec) -> StreamInput:
    """Build the stream input for ``spec``: a remote range source or inline bytes.

    A runner that fetches bytes itself (``prefers_remote_source``) gets a ``RangeSource`` when
    the corpus exposes one, so the heavy stream bytes are not re-uploaded. Either way the
    validator computes ``expected_sha256`` from the trusted corpus locally and pins it on the
    input -- that anchors the round-trip target so the split compress/decompress workers can't
    fake bit-exactness (#14). Hashing is a local read, not an upload.
    """

    data = provider.materialize(spec)
    expected_sha256 = hashlib.sha256(data).hexdigest()
    if getattr(runner, "prefers_remote_source", False):
        source = provider.stream_source(spec)
        if source is not None:
            return StreamInput(spec.stream_id, source=source, expected_sha256=expected_sha256)
    return StreamInput(spec.stream_id, data=data, expected_sha256=expected_sha256)


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
            result = runner.run_stream(artifact, stream_input, caps=caps)
            results.append(replace(result, source=spec.source, scored=spec.scored))
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
                    source=spec.source,
                    scored=spec.scored,
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
