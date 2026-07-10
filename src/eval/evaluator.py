"""Paired evaluation of codecs over identical beacon-seeded streams.

The incumbent and every challenger are run on the *same* streams in a round, so shared
data difficulty cancels and one-shot scoring is fair. Each stream's compress+decompress
runs as one same-worker job via a ``CodecRunner``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from bittensor.utils.btlogging import logging as bt_logging

from eval.corpus import CorpusProvider
from eval.runner import ArtifactRef, CodecRunner, HostUnavailableError, ResourceCaps, RunnerError, StreamInput
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
    total = len(stream_specs)
    for index, spec in enumerate(stream_specs, start=1):
        bt_logging.info(f"evaluating {hotkey}: stream {spec.stream_id} ({index}/{total})...")
        stream_input = _prepare_stream(runner, provider, spec)
        try:
            result = replace(runner.run_stream(artifact, stream_input, caps=caps), source=spec.source, scored=spec.scored)
            results.append(result)
            ratio = result.compressed_bytes / result.raw_bytes if result.raw_bytes else 0.0
            bt_logging.info(
                f"evaluating {hotkey}: stream {spec.stream_id} done -- ratio={ratio:.4f} "
                f"roundtrip_ok={result.roundtrip_ok} compress_secs={result.compress_secs:.1f} "
                f"decompress_secs={result.decompress_secs:.1f}"
            )
        except HostUnavailableError:
            # The validator host, not the codec, could not run this phase (e.g. GPU
            # occupied). Propagate so the whole round aborts and NO codec is scored or
            # excluded -- the main loop logs it and retries when the host is healthy.
            # Never record this as a failed stream (that would be a one-shot exclusion for
            # a host fault the codec had no control over).
            raise
        except RunnerError as exc:
            bt_logging.warning(f"evaluating {hotkey}: stream {spec.stream_id} failed: {exc}")
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
