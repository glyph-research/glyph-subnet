"""Codec execution backends.

A ``CodecRunner`` runs one stream's full round-trip -- compress then decompress -- on a
**single worker**, which is what lets Glyph require only same-system determinism (DESIGN
§4). Two implementations share the contract:

- ``LocalSubprocessRunner``: executes the artifact's entrypoints in a subprocess on the
  host. DEV/CI/own-codec only -- a key-holding validator host must not execute arbitrary
  untrusted artifacts. Used for M0, ``glyph-check-model``, and tests.
- ``ChutesRunner`` (see chute_app.py / runner_chutes wiring): dispatches the same job to
  the deployed Chutes (SN64) endpoint for production. Added in a later step.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from core.artifact import load_manifest, resolve_argv
from core.constants import (
    COMPRESS_BUDGET_SECS,
    DEFAULT_MAX_ARTIFACT_BYTES,
    RAM_CAP_BYTES,
    VRAM_CAP_BYTES,
)
from eval.scoring import StreamResult
from eval.streams import RangeSource


class RunnerError(Exception):
    """Raised when a codec entrypoint fails, times out, or produces no output."""


@dataclass
class ArtifactRef:
    repo: str
    rev: str
    sha256: str | None = None
    local_path: str | None = None  # directory containing manifest.json (local runner)


@dataclass
class StreamInput:
    stream_id: str
    data: bytes | None = None  # inline bytes: local runner, tests, small/smoke streams
    source: RangeSource | None = None  # production: bytes range-fetched by the remote worker

    @property
    def raw_len(self) -> int:
        """Raw stream length, whether materialized inline or known from the remote range."""

        if self.data is not None:
            return len(self.data)
        if self.source is not None:
            return self.source.length
        return 0


@dataclass
class ResourceCaps:
    wall_clock_secs: float = COMPRESS_BUDGET_SECS
    ram_bytes: int = RAM_CAP_BYTES
    vram_bytes: int = VRAM_CAP_BYTES
    artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES
    network: bool = False


class CodecRunner(Protocol):
    # True when the runner fetches stream bytes itself from a ``StreamInput.source``; the
    # evaluator then skips materializing/inlining them. False runners get inline ``data``.
    prefers_remote_source: bool

    def run_stream(
        self, artifact: ArtifactRef, stream: StreamInput, *, caps: ResourceCaps
    ) -> StreamResult: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LocalSubprocessRunner:
    """Run a codec's entrypoints in a subprocess on the host (dev/CI/own-codec only)."""

    prefers_remote_source = False  # runs on the host; always needs the bytes materialized

    def __init__(self, *, strict_sandbox: bool = False):
        # strict_sandbox best-effort drops network via `unshare --net` when available.
        # Real isolation of untrusted artifacts is the Chutes container's job (DESIGN §6).
        self.strict_sandbox = strict_sandbox

    def run_stream(
        self, artifact: ArtifactRef, stream: StreamInput, *, caps: ResourceCaps | None = None
    ) -> StreamResult:
        caps = caps or ResourceCaps()
        if stream.data is None:
            raise RunnerError("LocalSubprocessRunner needs inline stream data, not a remote source")
        if not artifact.local_path:
            raise RunnerError("LocalSubprocessRunner requires artifact.local_path")
        artifact_dir = Path(artifact.local_path)
        manifest = load_manifest(artifact_dir)

        with tempfile.TemporaryDirectory(prefix="glyph-run-") as tmp:
            tmp_dir = Path(tmp)
            stream_file = tmp_dir / "stream.bin"
            blob_file = tmp_dir / "blob.bin"
            roundtrip_file = tmp_dir / "roundtrip.bin"

            stream_file.write_bytes(stream.data)
            source_hash = _sha256_file(stream_file)

            compress_argv = resolve_argv(manifest.entrypoints.compress, stream_file, blob_file)
            compress_secs = self._exec(compress_argv, artifact_dir, caps)
            if not blob_file.exists():
                raise RunnerError("compress produced no output blob")
            compressed_bytes = blob_file.stat().st_size
            blob_hash = _sha256_file(blob_file)

            decompress_argv = resolve_argv(manifest.entrypoints.decompress, blob_file, roundtrip_file)
            decompress_secs = self._exec(decompress_argv, artifact_dir, caps)
            roundtrip_ok = roundtrip_file.exists() and _sha256_file(roundtrip_file) == source_hash

            return StreamResult(
                stream_id=stream.stream_id,
                raw_bytes=len(stream.data),
                compressed_bytes=compressed_bytes,
                roundtrip_ok=roundtrip_ok,
                compress_secs=compress_secs,
                decompress_secs=decompress_secs,
                blob_hash=blob_hash,
            )

    def _exec(self, argv: list[str], cwd: Path, caps: ResourceCaps) -> float:
        wrapped = argv
        if self.strict_sandbox and not caps.network and shutil.which("unshare"):
            wrapped = ["unshare", "--net", "--", *argv]
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                wrapped,
                cwd=str(cwd),
                env=os.environ.copy(),
                timeout=caps.wall_clock_secs,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.TimeoutExpired as exc:
            raise RunnerError(f"entrypoint timed out after {caps.wall_clock_secs:.0f}s") from exc
        elapsed = time.perf_counter() - start
        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", "replace")[-500:]
            raise RunnerError(f"entrypoint exited {proc.returncode}: {tail}")
        return elapsed
