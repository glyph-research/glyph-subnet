"""Codec execution backends.

A ``CodecRunner`` runs one stream's full round-trip -- compress then decompress -- on a
**single worker**, which is what lets Glyph require only same-system determinism. Two
implementations share the contract:

- ``LocalSubprocessRunner``: executes the artifact's entrypoints in a subprocess on the
  host. DEV/CI/own-codec only -- a key-holding validator host must not execute arbitrary
  untrusted artifacts. Used for M0, ``glyph-check-model``, and tests.
- ``ChutesRunner`` (see chute_app.py / runner_chutes wiring): dispatches the same job to
  the deployed Chutes (SN64) endpoint for production. Added in a later step.
"""

from __future__ import annotations

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
from core.hashing import sha256_file
from eval.scoring import StreamResult
from eval.streams import RangeSource


class RunnerError(Exception):
    """Raised when a codec entrypoint fails, times out, or produces no output."""


class HostUnavailableError(RunnerError):
    """Raised when the *validator host* (not the codec) cannot run a scored phase -- e.g.
    the GPU is already occupied by another process so any codec would OOM at model load.

    Distinct from RunnerError so the evaluator can tell "the codec is broken" (-> invalid,
    one-shot exclusion) apart from "our machine is broken" (-> abort the round, penalize
    nobody, retry when healthy). Mis-attributing the latter as the former permanently
    excludes innocent codecs, which is exactly what happened when a leaked process held the
    validator GPU during a round."""


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
    # sha256 of the raw stream, computed by the validator from the trusted corpus. The
    # round-trip target: anchoring it here (not on the untrusted worker's report) stops a
    # codec from faking bit-exactness across the split compress/decompress workers.
    expected_sha256: str | None = None

    @property
    def raw_len(self) -> int:
        """Raw stream length, whether materialized inline or known from the remote range."""

        if self.data is not None:
            return len(self.data)
        if self.source is not None:
            return self.source.length
        return 0


@dataclass
class CompressOutcome:
    blob: bytes  # the compressed artifact, handed to a *separate* decompress worker
    compressed_bytes: int
    compress_secs: float
    blob_hash: str
    source_hash: str  # sha256 of the raw input as seen by the compress worker
    raw_bytes: int


@dataclass
class DecompressOutcome:
    output_hash: str  # sha256 of the reconstructed bytes
    decompress_secs: float
    raw_bytes: int


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


_SUBPROCESS_ENV_ALLOWLIST = {
    "CUDA_VISIBLE_DEVICES",
    "GLYPH_TS_ZIP_DEVICE",
    "GLYPH_TS_ZIP_THREADS",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
    "PATH",
    "PYTHONPATH",
}

_PRIVDROP_UID_ENV = "GLYPH_CODEC_PRIVDROP_UID"
_PRIVDROP_GID_ENV = "GLYPH_CODEC_PRIVDROP_GID"
_PRIVDROP_REQUIRE_ENV = "GLYPH_CODEC_REQUIRE_PRIVDROP"
_NO_NEW_PRIVS_ENV = "GLYPH_CODEC_NO_NEW_PRIVS"


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _setpriv_prefix() -> list[str]:
    uid = os.environ.get(_PRIVDROP_UID_ENV)
    gid = os.environ.get(_PRIVDROP_GID_ENV)
    requested = bool(uid or gid or _env_truthy(_NO_NEW_PRIVS_ENV))
    if not requested:
        return []
    if bool(uid) != bool(gid):
        raise RunnerError(f"{_PRIVDROP_UID_ENV} and {_PRIVDROP_GID_ENV} must be set together")
    setpriv = shutil.which("setpriv")
    if setpriv is None:
        if _env_truthy(_PRIVDROP_REQUIRE_ENV):
            raise RunnerError("setpriv unavailable for required codec privilege drop")
        return []
    prefix = [setpriv, "--no-new-privs", "--bounding-set=-all"]
    if uid and gid:
        prefix += ["--reuid", uid, "--regid", gid, "--clear-groups"]
    return [*prefix, "--"]


def _subprocess_env(home: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key in _SUBPROCESS_ENV_ALLOWLIST}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["HOME"] = str(home)
    env["XDG_CACHE_HOME"] = str(home / ".cache")
    env["TMPDIR"] = str(home)
    env["NO_PROXY"] = "*"
    # Force the HF/transformers stack fully offline so a neural codec can only load weights
    # bundled in its already-snapshot_download'd + prechecked artifact dir -- never a runtime
    # download. Defense-in-depth with the `unshare --net` isolation: every model byte is forced
    # through the trusted prep boundary (where it is hashed/prechecked), and an attempted
    # download fails immediately instead of hanging until the wall-clock timeout.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    return env


class LocalSubprocessRunner:
    """Run a codec's entrypoints in a subprocess on the host (dev/CI/own-codec only)."""

    prefers_remote_source = False  # runs on the host; always needs the bytes materialized

    def __init__(self, *, strict_sandbox: bool = False, require_network_isolation: bool = False):
        # strict_sandbox drops network via `unshare --net` when available. Production
        # Chutes execution sets require_network_isolation so this fails closed.
        self.strict_sandbox = strict_sandbox
        self.require_network_isolation = require_network_isolation

    def _artifact_dir(self, artifact: ArtifactRef) -> Path:
        if not artifact.local_path:
            raise RunnerError("LocalSubprocessRunner requires artifact.local_path")
        return Path(artifact.local_path)

    def compress(
        self, artifact: ArtifactRef, data: bytes, *, caps: ResourceCaps | None = None
    ) -> CompressOutcome:
        """Run only the compress entrypoint in its own sandbox; return the blob to transfer."""

        caps = caps or ResourceCaps()
        artifact_dir = self._artifact_dir(artifact)
        manifest = load_manifest(artifact_dir)
        with tempfile.TemporaryDirectory(prefix="glyph-compress-") as tmp:
            tmp_dir = Path(tmp)
            stream_file = tmp_dir / "stream.bin"
            blob_file = tmp_dir / "blob.bin"
            stream_file.write_bytes(data)
            source_hash = sha256_file(stream_file)
            argv = resolve_argv(manifest.entrypoints.compress, stream_file, blob_file)
            secs = self._exec(argv, artifact_dir, caps, home=tmp_dir)
            if not blob_file.exists():
                raise RunnerError("compress produced no output blob")
            blob = blob_file.read_bytes()
            return CompressOutcome(
                blob=blob,
                compressed_bytes=len(blob),
                compress_secs=secs,
                blob_hash=sha256_file(blob_file),
                source_hash=source_hash,
                raw_bytes=len(data),
            )

    def decompress(
        self, artifact: ArtifactRef, blob: bytes, *, caps: ResourceCaps | None = None
    ) -> DecompressOutcome:
        """Run only the decompress entrypoint in a FRESH sandbox seeded with only the blob.

        Because this sandbox (and, in production, this *separate chute/container*) shares no
        filesystem, process table, or shared memory with the compress worker, a codec cannot
        stash the raw input during compress and read it back here -- it must genuinely encode
        everything in ``blob``, so the measured compression ratio is honest.
        """

        caps = caps or ResourceCaps()
        artifact_dir = self._artifact_dir(artifact)
        manifest = load_manifest(artifact_dir)
        with tempfile.TemporaryDirectory(prefix="glyph-decompress-") as tmp:
            tmp_dir = Path(tmp)
            blob_file = tmp_dir / "blob.bin"
            roundtrip_file = tmp_dir / "roundtrip.bin"
            blob_file.write_bytes(blob)
            argv = resolve_argv(manifest.entrypoints.decompress, blob_file, roundtrip_file)
            secs = self._exec(argv, artifact_dir, caps, home=tmp_dir)
            if not roundtrip_file.exists():
                raise RunnerError("decompress produced no output")
            return DecompressOutcome(
                output_hash=sha256_file(roundtrip_file),
                decompress_secs=secs,
                raw_bytes=roundtrip_file.stat().st_size,
            )

    def run_stream(
        self, artifact: ArtifactRef, stream: StreamInput, *, caps: ResourceCaps | None = None
    ) -> StreamResult:
        caps = caps or ResourceCaps()
        if stream.data is None:
            raise RunnerError("LocalSubprocessRunner needs inline stream data, not a remote source")
        comp = self.compress(artifact, stream.data, caps=caps)
        decomp = self.decompress(artifact, comp.blob, caps=caps)  # separate sandbox
        # Prefer the validator-anchored hash; fall back to the compress worker's own view only
        # when no trusted hash was supplied (local smoke/tests).
        expected = stream.expected_sha256 or comp.source_hash
        return StreamResult(
            stream_id=stream.stream_id,
            raw_bytes=comp.raw_bytes,
            compressed_bytes=comp.compressed_bytes,
            roundtrip_ok=decomp.output_hash == expected,
            compress_secs=comp.compress_secs,
            decompress_secs=decomp.decompress_secs,
            blob_hash=comp.blob_hash,
        )

    def _exec(self, argv: list[str], cwd: Path, caps: ResourceCaps, *, home: Path) -> float:
        # Security invariant: untrusted codec code only ever runs HERE, and it runs with the
        # network dropped (`unshare --net`) while the corpus stream is present. All trusted prep
        # that needs the network (snapshot_download of the artifact + weights, corpus fetch) runs
        # BEFORE this and executes no untrusted code. So even arbitrary code / malicious weights /
        # malicious deps cannot exfiltrate: the data is only present when there is no network, and
        # `_subprocess_env` also pins the HF stack offline as a second layer.
        wrapped = argv
        setpriv_prefix = _setpriv_prefix()
        if setpriv_prefix:
            home.chmod(0o777)
            wrapped = [*setpriv_prefix, *wrapped]
        if self.strict_sandbox and not caps.network:
            if shutil.which("unshare"):
                wrapped = ["unshare", "--net", "--", *wrapped]
            elif self.require_network_isolation:
                raise RunnerError("network isolation unavailable for untrusted artifact execution")
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                wrapped,
                cwd=str(cwd),
                env=_subprocess_env(home),
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
