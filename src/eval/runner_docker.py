"""DockerRunner: run codec entrypoints as ephemeral Docker containers on the validator host,
instead of dispatching to the remote Chutes eval chutes.

Motivation: Chutes now mandates a specific, frequently-unavailable GPU SKU (``pro_6000``) for
TEE chutes tied to an integrated subnet, and a platform-side code-verification issue (aegis
``cllmv``) makes the deployed eval chutes intermittently unroutable regardless of code
correctness (see ``chute_app.py``'s module docstring for the full diagnosis). Running
compress/decompress locally in Docker sidesteps both: the operator controls their own
GPU/image, and there is no remote platform to fight with.

Isolation model mirrors ``LocalSubprocessRunner``/``ChutesRunner`` (DESIGN §6,
exploit-prevention #14): compress and decompress each run in a FRESH, ephemeral container
(``docker run --rm``) with no shared filesystem, network, or process table between them, so a
codec cannot stash the raw input during compress and read it back during decompress. Untrusted
code execution gets ``--network none`` (the container-native equivalent of
``LocalSubprocessRunner``'s ``unshare --net``) whenever ``ResourceCaps.network`` is False (the
default). Docker does not inherit the host's environment by default, so no explicit secret
scrubbing is needed the way ``_SUBPROCESS_ENV_ALLOWLIST`` handles it for subprocess exec --
only the vars this module explicitly passes via ``-e`` reach the container.

The artifact must already be fetched to local disk (``ArtifactRef.local_path``) before this
runner is invoked, exactly like ``LocalSubprocessRunner`` -- see ``needs_local_artifact``.

``gpu=True`` requires the host's GPU to match ``core.constants.DOCKER_REFERENCE_GPU`` (checked
once at construction via ``nvidia-smi``, fail-closed) -- see ``_verify_gpu_model``. Every
validator using ``--docker-gpu`` must run on identical hardware, or compress_secs/decompress_secs
are not comparable across validators (DESIGN §4 same-system determinism), same rationale as
Chutes' ``REFERENCE_SKU`` pin.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from core.artifact import load_manifest, resolve_argv
from core.constants import DOCKER_REFERENCE_GPU
from core.hashing import sha256_file
from eval.runner import (
    ArtifactRef,
    CompressOutcome,
    DecompressOutcome,
    ResourceCaps,
    RunnerError,
    StreamInput,
)
from eval.scoring import StreamResult

# A general-purpose default; codecs needing extra packages (e.g. torch for a neural codec)
# should ship their own image via --docker-image, or the manifest's own entrypoint can
# `pip install` at the cost of wall-clock budget. Pre-pull whatever image you configure --
# a cold pull inside the timed run eats into caps.wall_clock_secs.
DEFAULT_DOCKER_IMAGE = "python:3.12-slim"

_CONTAINER_ARTIFACT_DIR = "/artifact"
_CONTAINER_SCRATCH_DIR = "/scratch"
# Host env vars forwarded into the container only when the host itself has them set --
# never blanket-inherited (see module docstring).
_FORWARDED_ENV_VARS = ("CUDA_VISIBLE_DEVICES", "GLYPH_TS_ZIP_DEVICE", "GLYPH_TS_ZIP_THREADS")


def _verify_gpu_model(gpu_device: str | None, reference_gpu: str) -> None:
    """Fail closed unless every GPU this runner would use matches ``reference_gpu`` (see
    ``core.constants.DOCKER_REFERENCE_GPU``): compress_secs/decompress_secs are only comparable
    across validators (DESIGN §4 same-system determinism) if the hardware is identical. Checked
    once per ``DockerRunner`` construction via the HOST's ``nvidia-smi`` -- GPU hardware doesn't
    change mid-process, so this doesn't need to run per-invocation.
    """

    if shutil.which("nvidia-smi") is None:
        raise RunnerError(
            f"--docker-gpu requires nvidia-smi on PATH to verify the GPU model is {reference_gpu!r}; "
            "none found"
        )
    cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    if gpu_device:
        cmd += ["-i", gpu_device]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except Exception as exc:  # noqa: BLE001
        raise RunnerError(f"failed to query GPU model via nvidia-smi: {exc}") from exc
    if proc.returncode != 0:
        raise RunnerError(f"nvidia-smi failed: {proc.stderr.strip()[:300]}")
    names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not names:
        raise RunnerError("nvidia-smi reported no GPUs")
    mismatched = [n for n in names if reference_gpu not in n]
    if mismatched:
        raise RunnerError(
            f"DockerRunner requires GPU model containing {reference_gpu!r} for cross-validator "
            f"throughput comparability (DESIGN §4); found: {', '.join(mismatched)}"
        )


class DockerRunner:
    """Run a codec's entrypoints as ephemeral Docker containers (local production path)."""

    prefers_remote_source = False  # runs on the host; always needs the bytes materialized
    needs_local_artifact = True  # reign_worker.artifact_ref() must snapshot_download first

    def __init__(
        self,
        *,
        image: str = DEFAULT_DOCKER_IMAGE,
        gpu: bool = False,
        gpu_device: str | None = None,
        docker_bin: str = "docker",
    ):
        if shutil.which(docker_bin) is None:
            raise RunnerError(
                f"docker binary not found on PATH ({docker_bin!r}); install Docker to use DockerRunner"
            )
        if gpu:
            # Network-wide requirement, not a per-operator override (same rationale as
            # WINDOW_ANCHOR_BLOCK / REFERENCE_SKU) -- deliberately no CLI flag to bypass this.
            _verify_gpu_model(gpu_device, DOCKER_REFERENCE_GPU)
        self.image = image
        self.gpu = gpu
        self.gpu_device = gpu_device
        self.docker_bin = docker_bin

    def _artifact_dir(self, artifact: ArtifactRef) -> Path:
        if not artifact.local_path:
            raise RunnerError("DockerRunner requires artifact.local_path (fetch the artifact locally first)")
        # `docker run -v` requires an absolute host path; a relative local_path would otherwise
        # resolve against the Docker daemon's cwd, not the caller's.
        return Path(artifact.local_path).resolve()

    def compress(
        self, artifact: ArtifactRef, data: bytes, *, caps: ResourceCaps | None = None
    ) -> CompressOutcome:
        caps = caps or ResourceCaps()
        artifact_dir = self._artifact_dir(artifact)
        manifest = load_manifest(artifact_dir)
        with tempfile.TemporaryDirectory(prefix="glyph-docker-compress-") as tmp:
            tmp_dir = Path(tmp)
            stream_file = tmp_dir / "in.bin"
            blob_file = tmp_dir / "out.bin"
            stream_file.write_bytes(data)
            source_hash = sha256_file(stream_file)
            argv = resolve_argv(
                manifest.entrypoints.compress,
                f"{_CONTAINER_SCRATCH_DIR}/in.bin",
                f"{_CONTAINER_SCRATCH_DIR}/out.bin",
            )
            secs = self._run_container(argv, artifact_dir, tmp_dir, caps)
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
        """Run decompress in a FRESH container seeded with only the blob (see module docstring)."""

        caps = caps or ResourceCaps()
        artifact_dir = self._artifact_dir(artifact)
        manifest = load_manifest(artifact_dir)
        with tempfile.TemporaryDirectory(prefix="glyph-docker-decompress-") as tmp:
            tmp_dir = Path(tmp)
            blob_file = tmp_dir / "in.bin"
            roundtrip_file = tmp_dir / "out.bin"
            blob_file.write_bytes(blob)
            argv = resolve_argv(
                manifest.entrypoints.decompress,
                f"{_CONTAINER_SCRATCH_DIR}/in.bin",
                f"{_CONTAINER_SCRATCH_DIR}/out.bin",
            )
            secs = self._run_container(argv, artifact_dir, tmp_dir, caps)
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
            raise RunnerError("DockerRunner needs inline stream data, not a remote source")
        comp = self.compress(artifact, stream.data, caps=caps)
        decomp = self.decompress(artifact, comp.blob, caps=caps)  # separate container
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

    def _run_container(
        self, argv: list[str], artifact_dir: Path, scratch_dir: Path, caps: ResourceCaps
    ) -> float:
        name = f"glyph-runner-{uuid.uuid4().hex[:12]}"
        cmd = [
            self.docker_bin, "run", "--rm", "--name", name,
            "-w", _CONTAINER_ARTIFACT_DIR,
            "-v", f"{artifact_dir}:{_CONTAINER_ARTIFACT_DIR}:ro",
            "-v", f"{scratch_dir}:{_CONTAINER_SCRATCH_DIR}",
            "--memory", str(caps.ram_bytes),
            "--pids-limit", "512",
        ]
        if not caps.network:
            # Untrusted codec code only ever runs HERE, with the network dropped while the
            # corpus stream/blob is present -- same invariant as LocalSubprocessRunner's
            # `unshare --net`. The artifact fetch (network-requiring, trusted) already
            # happened before this runner was invoked.
            cmd += ["--network", "none"]
        if self.gpu:
            # NVIDIA_VISIBLE_DEVICES scoped to the actual --gpus selection, not always "all" --
            # when gpu_device pins a specific card, the container shouldn't see (or be able to
            # request) any other.
            cmd += ["--gpus", f"device={self.gpu_device}" if self.gpu_device else "all"]
            cmd += ["-e", f"NVIDIA_VISIBLE_DEVICES={self.gpu_device or 'all'}",
                    "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"]
        for env_var in _FORWARDED_ENV_VARS:
            value = os.environ.get(env_var)
            if value is not None:
                cmd += ["-e", f"{env_var}={value}"]
        cmd += [
            "-e", f"HOME={_CONTAINER_SCRATCH_DIR}",
            "-e", f"XDG_CACHE_HOME={_CONTAINER_SCRATCH_DIR}/.cache",
            "-e", f"TMPDIR={_CONTAINER_SCRATCH_DIR}",
            "-e", "NO_PROXY=*",
            # Force the HF/transformers stack fully offline, mirroring LocalSubprocessRunner:
            # a neural codec can only load weights bundled in its already-prechecked artifact
            # dir, never a runtime download (defense-in-depth alongside --network none).
            "-e", "HF_HUB_OFFLINE=1",
            "-e", "TRANSFORMERS_OFFLINE=1",
            "-e", "HF_DATASETS_OFFLINE=1",
            self.image,
            *argv,
        ]
        start = time.perf_counter()
        try:
            proc = subprocess.run(cmd, timeout=caps.wall_clock_secs, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.TimeoutExpired as exc:
            # `docker run`'s own process dying on timeout does NOT stop the daemon-side
            # container; kill it explicitly (best-effort) so it doesn't keep running/billing.
            subprocess.run(
                [self.docker_bin, "kill", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            raise RunnerError(f"entrypoint timed out after {caps.wall_clock_secs:.0f}s") from exc
        elapsed = time.perf_counter() - start
        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", "replace")[-500:]
            raise RunnerError(f"entrypoint exited {proc.returncode}: {tail}")
        return elapsed
