"""Host-unavailable preflight: a validator GPU that is already occupied must abort the
round (HostUnavailableError), not mark the codec invalid (one-shot exclusion)."""

from __future__ import annotations

import pytest

from eval import runner_docker
from eval.evaluator import evaluate_artifact
from eval.runner import ArtifactRef, HostUnavailableError, ResourceCaps, RunnerError, StreamInput
from eval.scoring import StreamResult


def test_host_unavailable_is_runner_error_subclass():
    # Subclassing matters: callers that only catch RunnerError still catch this, but code
    # that wants to distinguish host faults can catch HostUnavailableError first.
    assert issubclass(HostUnavailableError, RunnerError)


def test_free_vram_parsing(monkeypatch):
    class _Proc:
        returncode = 0
        stdout = "2048\n40000\n"  # MiB, two GPUs; min should win

    monkeypatch.setattr(runner_docker.subprocess, "run", lambda *a, **k: _Proc())
    assert runner_docker._free_vram_bytes(None) == 2048 * 2**20


def test_free_vram_unknown_on_error(monkeypatch):
    def _boom(*a, **k):
        raise OSError("nvidia-smi missing")

    monkeypatch.setattr(runner_docker.subprocess, "run", _boom)
    assert runner_docker._free_vram_bytes(None) == -1  # -> caller skips the guard


class _OccupiedGpuRunner:
    """Stand-in runner that raises HostUnavailableError like DockerRunner's preflight does
    when the host GPU is occupied."""

    prefers_remote_source = False

    def run_stream(self, artifact, stream, *, caps):
        raise HostUnavailableError("validator GPU has only 0.3 GiB free")


class _StubProvider:
    def materialize(self, spec):
        return b"x" * 1024

    def stream_source(self, spec):
        return None


def test_evaluator_propagates_host_unavailable(monkeypatch):
    from eval.streams import StreamSpec

    specs = [StreamSpec(stream_id="fineweb-0", offset=0, length=1024, source="fineweb")]
    # Must RAISE (abort round), not return an EvalOutcome with an invalid stream.
    with pytest.raises(HostUnavailableError):
        evaluate_artifact(
            _OccupiedGpuRunner(),
            "hotkeyA",
            ArtifactRef(repo="r", rev="v"),
            _StubProvider(),
            specs,
            caps=ResourceCaps(),
            floor_bps=10240,
            budget_secs=450.0,
        )


class _BrokenCodecRunner:
    prefers_remote_source = False

    def run_stream(self, artifact, stream, *, caps):
        raise RunnerError("entrypoint exited 1")  # a genuine codec fault


def test_evaluator_still_fails_broken_codec(monkeypatch):
    # Contrast: an ordinary RunnerError (codec's fault) must still produce an invalid
    # outcome (one-shot exclusion), not abort the round.
    from eval.streams import StreamSpec

    specs = [StreamSpec(stream_id="fineweb-0", offset=0, length=1024, source="fineweb")]
    outcome = evaluate_artifact(
        _BrokenCodecRunner(),
        "hotkeyB",
        ArtifactRef(repo="r", rev="v"),
        _StubProvider(),
        specs,
        caps=ResourceCaps(),
        floor_bps=10240,
        budget_secs=450.0,
    )
    assert not outcome.score.valid
    assert outcome.error is not None
