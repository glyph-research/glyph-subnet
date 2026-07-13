"""DockerRunner: run codec entrypoints as ephemeral Docker containers on the validator host,
instead of dispatching to the remote Chutes eval chutes.

Motivation: Chutes now mandates a specific, frequently-unavailable GPU SKU (``pro_6000``) for
TEE chutes tied to an integrated subnet, and a platform-side code-verification issue (aegis
``cllmv``) makes the deployed eval chutes intermittently unroutable regardless of code
correctness (see ``chute_app.py``'s module docstring for the full diagnosis). Running
compress/decompress locally in Docker sidesteps both: the operator controls their own
GPU/image, and there is no remote platform to fight with.

Isolation model mirrors ``LocalSubprocessRunner``/``ChutesRunner`` (exploit-prevention #14):
compress and decompress each run in a FRESH, ephemeral container
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
are not comparable across validators (same-system determinism), same rationale as Chutes'
``REFERENCE_SKU`` pin.

A manifest may also declare its own digest-pinned ``image`` (issue #48): a miner-published
image with whatever deps/weights it needs, rather than the operator-supplied generic
``self.image``. That path runs a different lifecycle (``_run_networked_lifecycle``): the
container starts detached with network attached and no eval data present, warms up (installs
deps / downloads weights / loads the model) until it signals ready, has its network severed,
and only THEN is the eval input written into its (already-mounted, so-far-empty) scratch
directory before the scored compress/decompress entrypoint runs via ``docker exec``. A
manifest with no ``image`` keeps the original single-shot ``docker run --rm --network none``
path unchanged (the zstd reference codec and other dev/CI codecs that need no warmup).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from core.artifact import Warmup, is_image_digest_pinned, load_manifest, resolve_argv
from core.constants import DOCKER_REFERENCE_GPU, SCRATCH_CAP_BYTES
from core.hashing import sha256_file
from eval.runner import (
    ArtifactRef,
    CompressOutcome,
    DecompressOutcome,
    InsufficientGpuMemoryError,
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
_DEFAULT_CONTAINER_USER = "65534:65534"
# Host env vars forwarded into the container only when the host itself has them set --
# never blanket-inherited (see module docstring).
_FORWARDED_ENV_VARS = ("CUDA_VISIBLE_DEVICES", "GLYPH_TS_ZIP_DEVICE", "GLYPH_TS_ZIP_THREADS")
_WARMUP_READY_POLL_SECS = 1.0
_MIN_CONTAINER_CREATE_TIMEOUT_SECS = 60.0
# Headroom required above a manifest's declared resources["vram_gb"] before launching that
# codec's container (issue #105): declared values are miner-supplied and not verified
# elsewhere, and a prior container's leaked/still-tearing-down memory shouldn't be counted as
# available. Flat rather than proportional so it means the same thing at any declared size.
_GPU_MEMORY_HEADROOM_GB = 2.0


def _allow_sandbox_read_tree(root: Path) -> None:
    for dirpath, _dirnames, filenames in os.walk(root):
        directory = Path(dirpath)
        directory.chmod(directory.stat().st_mode | 0o755)
        for filename in filenames:
            path = directory / filename
            path.chmod(path.stat().st_mode | 0o444)


def _dir_size_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            pass  # file removed/replaced mid-walk by the still-running codec; skip it
    return total


class _DiskWatchdog:
    """Kills ``container_name`` if ``scratch_dir``'s total size exceeds ``cap_bytes`` while a
    blocking docker run/exec call is in progress (issue #54).

    ``subprocess.run(..., timeout=...)`` alone only bounds wall-clock time, not how much disk
    a codec can fill before that timeout elapses (the compress/decompress wall-clock budget
    can be tens of minutes) -- this polls in the background and kills the container the
    moment the cap is exceeded, real-time rather than only detected after the fact.
    ``--ulimit fsize=`` (set on the container itself) is the complementary per-file kernel
    cap; this catches the many-small-files case that alone wouldn't.
    """

    def __init__(self, docker_bin: str, container_name: str, scratch_dir: Path, cap_bytes: int, poll_secs: float = 2.0):
        self._docker_bin = docker_bin
        self._container_name = container_name
        self._scratch_dir = scratch_dir
        self._cap_bytes = cap_bytes
        self._poll_secs = poll_secs
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self.triggered = False

    def _poll(self) -> None:
        while not self._stop.wait(self._poll_secs):
            if _dir_size_bytes(self._scratch_dir) > self._cap_bytes:
                self.triggered = True
                subprocess.run(
                    [self._docker_bin, "kill", self._container_name],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                return

    def __enter__(self) -> "_DiskWatchdog":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def _verify_gpu_model(gpu_device: str | None, reference_gpu: str) -> None:
    """Fail closed unless every GPU this runner would use matches ``reference_gpu`` (see
    ``core.constants.DOCKER_REFERENCE_GPU``): compress_secs/decompress_secs are only comparable
    across validators (same-system determinism) if the hardware is identical. Checked once per
    ``DockerRunner`` construction via the HOST's ``nvidia-smi`` -- GPU hardware doesn't change
    mid-process, so this doesn't need to run per-invocation.
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
            f"throughput comparability; found: {', '.join(mismatched)}"
        )


def _free_gpu_memory_gb(gpu_device: str | None) -> float | None:
    """Minimum free VRAM (GiB) across the GPU(s) this runner would use, via ``nvidia-smi``.

    Returns None if it cannot be determined (missing binary, query failure, unparseable
    output) so the caller skips the preflight check rather than false-failing a codec on a
    host-tooling hiccup unrelated to actual GPU capacity.
    """

    if shutil.which("nvidia-smi") is None:
        return None
    cmd = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    if gpu_device:
        cmd += ["-i", gpu_device]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if proc.returncode != 0:
            return None
        values_mib = [int(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
    except Exception:  # noqa: BLE001
        return None
    if not values_mib:
        return None
    return min(values_mib) / 1024.0  # nvidia-smi reports MiB; report the tightest GPU


def _check_gpu_memory(manifest, gpu_device: str | None) -> None:
    """Fail closed before launching a codec's container if its declared
    ``resources["vram_gb"]`` plus a safety margin doesn't fit in currently-free VRAM
    (issue #105).

    Distinguishes a host-capacity problem (this GPU doesn't have room right now, e.g. a
    leaked/still-tearing-down prior container in the same round) from an entrypoint crash, so
    it surfaces as InsufficientGpuMemoryError rather than an indistinguishable "entrypoint
    exited ..." RunnerError. A manifest with no declared ``vram_gb``, or a host where free
    memory can't be determined, has nothing to check against and is skipped -- this is a
    defense-in-depth capacity guard, not a replacement for the manifest schema requiring the
    field.
    """

    declared = manifest.resources.get("vram_gb")
    if declared is None:
        return
    try:
        needed_gb = float(declared) + _GPU_MEMORY_HEADROOM_GB
    except (TypeError, ValueError):
        return
    free_gb = _free_gpu_memory_gb(gpu_device)
    if free_gb is None:
        return
    if free_gb < needed_gb:
        raise InsufficientGpuMemoryError(
            f"insufficient free GPU memory: need ~{needed_gb:.1f} GiB "
            f"(declared {float(declared):.1f} GiB + {_GPU_MEMORY_HEADROOM_GB:.1f} GiB headroom), "
            f"{free_gb:.1f} GiB free"
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
        seccomp_profile: str | None = None,
        container_user: str = _DEFAULT_CONTAINER_USER,
        scratch_cap_bytes: int = SCRATCH_CAP_BYTES,
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
        self.seccomp_profile = seccomp_profile
        self.container_user = container_user
        # Unlike DOCKER_REFERENCE_GPU/REFERENCE_SKU, this is a host-safety limit, not something
        # that needs to be bit-identical across validators for scoring -- overridable (mainly
        # for tests exercising the cap without writing tens of GiB), defaults to the
        # network-wide constant.
        self.scratch_cap_bytes = scratch_cap_bytes

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
            blob_file = tmp_dir / "out.bin"
            source_hash = hashlib.sha256(data).hexdigest()
            argv = resolve_argv(
                manifest.entrypoints.compress,
                f"{_CONTAINER_SCRATCH_DIR}/in.bin",
                f"{_CONTAINER_SCRATCH_DIR}/out.bin",
            )
            secs = self._execute(argv, artifact_dir, tmp_dir, caps, manifest, data)
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
            roundtrip_file = tmp_dir / "out.bin"
            argv = resolve_argv(
                manifest.entrypoints.decompress,
                f"{_CONTAINER_SCRATCH_DIR}/in.bin",
                f"{_CONTAINER_SCRATCH_DIR}/out.bin",
            )
            secs = self._execute(argv, artifact_dir, tmp_dir, caps, manifest, blob)
            if not roundtrip_file.exists():
                raise RunnerError("decompress produced no output")
            return DecompressOutcome(
                output_hash=sha256_file(roundtrip_file),
                decompress_secs=secs,
                raw_bytes=roundtrip_file.stat().st_size,
            )

    def _execute(
        self, argv: list[str], artifact_dir: Path, tmp_dir: Path, caps: ResourceCaps, manifest, input_data: bytes
    ) -> float:
        """Run one phase's entrypoint, dispatching on whether the manifest declares its own
        image (issue #48's networked warmup/seal/benchmark lifecycle) or not (the original
        single-shot ``--network none``-from-start path, unchanged)."""

        if self.gpu:
            _check_gpu_memory(manifest, self.gpu_device)
        if manifest.image is not None:
            return self._run_networked_lifecycle(argv, artifact_dir, tmp_dir, caps, manifest, input_data)
        (tmp_dir / "in.bin").write_bytes(input_data)
        return self._run_container(argv, artifact_dir, tmp_dir, caps)

    def _docker(self, args: list[str], *, timeout: float, check: bool = True) -> subprocess.CompletedProcess:
        try:
            proc = subprocess.run([self.docker_bin, *args], capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            if not check:
                raise
            raise RunnerError(f"docker {' '.join(args)} timed out after {timeout:.0f}s") from exc
        if check and proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", "replace")[-300:]
            raise RunnerError(f"docker {' '.join(args)} failed: {tail}")
        return proc

    def _await_warmup(self, name: str, warmup: Warmup) -> None:
        """Block until the container signals ready, or raise on timeout/failure -- fail-closed,
        never runs unbounded (issue #48's "bounded warmup timeout that kills+fails the codec").

        Two mutually exclusive readiness protocols: a bounded warmup ``command`` executed via
        ``docker exec`` (ready = exits 0), or, if none is given, polling for the container to
        create its own ``ready_file`` up to ``timeout_secs``.
        """

        if warmup.command:
            try:
                proc = subprocess.run(
                    [self.docker_bin, "exec", name, *warmup.command],
                    timeout=warmup.timeout_secs, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
            except subprocess.TimeoutExpired as exc:
                raise RunnerError(f"warmup command timed out after {warmup.timeout_secs:.0f}s") from exc
            if proc.returncode != 0:
                tail = proc.stderr.decode("utf-8", "replace")[-500:]
                raise RunnerError(f"warmup command exited {proc.returncode}: {tail}")
            return
        deadline = time.time() + warmup.timeout_secs
        while time.time() < deadline:
            probe = subprocess.run(
                [self.docker_bin, "exec", name, "test", "-e", warmup.ready_file],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if probe.returncode == 0:
                return
            time.sleep(_WARMUP_READY_POLL_SECS)
        raise RunnerError(f"warmup did not create {warmup.ready_file!r} within {warmup.timeout_secs:.0f}s")

    def _run_networked_lifecycle(
        self,
        argv: list[str],
        artifact_dir: Path,
        scratch_dir: Path,
        caps: ResourceCaps,
        manifest,
        input_data: bytes,
    ) -> float:
        """Warmup (network ON, no eval data present) -> seal (network severed) -> write the
        eval input -> benchmark (``docker exec``, no network) -- see module docstring.

        The container starts with an EMPTY scratch mount and its own dedicated bridge network
        so it can install deps / download weights, signals readiness, then has that network
        severed BEFORE any eval byte is written into the (already-mounted) scratch dir: a codec
        with network access during warmup has nothing to exfiltrate because nothing is there
        yet, and by the time real data exists the network is already gone.

        Deliberately does not consult ``caps.network`` -- the warmup-then-seal shape is a
        structural guarantee of this lifecycle, not a caller-configurable toggle (same
        rationale as ``_verify_gpu_model`` not being bypassable by a CLI flag).
        """

        image = manifest.image
        if not is_image_digest_pinned(image):
            raise RunnerError(
                f"manifest image {image!r} must be pinned by digest (repo@sha256:<hex>); refusing to run"
            )
        warmup = manifest.warmup or Warmup()
        scratch_dir.chmod(0o777)
        _allow_sandbox_read_tree(artifact_dir)
        name = f"glyph-runner-{uuid.uuid4().hex[:12]}"
        network = f"glyph-warmup-{uuid.uuid4().hex[:12]}"
        self._docker(["network", "create", network], timeout=15)
        try:
            run_cmd = [
                "run", "-d", "--name", name,
                "--network", network,
                "-w", _CONTAINER_ARTIFACT_DIR,
                "-v", f"{artifact_dir}:{_CONTAINER_ARTIFACT_DIR}:ro",
                "-v", f"{scratch_dir}:{_CONTAINER_SCRATCH_DIR}",
                "--memory", str(caps.ram_bytes),
                "--pids-limit", "512",
                "--user", self.container_user,
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges:true",
                # Per-file kernel cap; covers both the networked warmup's downloads and the
                # scored benchmark. The background _DiskWatchdog below covers the
                # many-small-files case this alone wouldn't (issue #54).
                "--ulimit", f"fsize={self.scratch_cap_bytes}",
            ]
            if self.seccomp_profile:
                run_cmd += ["--security-opt", f"seccomp={self.seccomp_profile}"]
            if self.gpu:
                run_cmd += ["--gpus", f"device={self.gpu_device}" if self.gpu_device else "all"]
                run_cmd += ["-e", f"NVIDIA_VISIBLE_DEVICES={self.gpu_device or 'all'}",
                            "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"]
            for env_var in _FORWARDED_ENV_VARS:
                value = os.environ.get(env_var)
                if value is not None:
                    run_cmd += ["-e", f"{env_var}={value}"]
            run_cmd += [
                "-e", f"HOME={_CONTAINER_SCRATCH_DIR}",
                "-e", f"XDG_CACHE_HOME={_CONTAINER_SCRATCH_DIR}/.cache",
                "-e", f"TMPDIR={_CONTAINER_SCRATCH_DIR}",
                # Deliberately NOT forcing HF_HUB_OFFLINE here -- warmup is the one phase
                # allowed network access, precisely so deps/weights can be fetched. The sealed
                # exec phase below has no network route at all regardless, offline or not.
                image,
            ]
            if warmup.command:
                # A `warmup.command` is exec'd into the container separately (see
                # _await_warmup), so the container's own process just needs to stay alive to
                # be exec'd into -- override the image's CMD/ENTRYPOINT with a lightweight
                # keep-alive. Without a `warmup.command`, the image's OWN CMD/ENTRYPOINT is
                # the long-running warmup+serving process (it must not exit and must create
                # `ready_file` itself) -- do not override it in that case.
                run_cmd += ["sleep", "infinity"]
            # A cold pull of an uncached image happens here too (same convention as the
            # default path: pre-pull to avoid eating into a timed budget). Container creation
            # (this call) is a distinct concern from the readiness wait below -- floor it at a
            # sane minimum so a caller's intentionally-short warmup.timeout_secs (e.g. a test
            # asserting the *readiness* deadline fires quickly) can't also starve `docker run
            # -d` itself under host load, leaking a container that never finished starting.
            self._docker(run_cmd, timeout=max(warmup.timeout_secs, _MIN_CONTAINER_CREATE_TIMEOUT_SECS))
            # One watchdog spans both warmup (downloads/deps) and the scored benchmark --
            # the same scratch mount and disk budget apply across both (issue #54).
            with _DiskWatchdog(self.docker_bin, name, scratch_dir, self.scratch_cap_bytes) as watchdog:
                try:
                    self._await_warmup(name, warmup)
                    # SEAL: sever network before any eval byte exists anywhere the container
                    # can see it.
                    self._docker(["network", "disconnect", network, name], timeout=15)
                except Exception as exc:
                    self._docker(["kill", name], timeout=15, check=False)
                    if watchdog.triggered:
                        raise RunnerError(
                            f"codec exceeded the {self.scratch_cap_bytes:,}-byte scratch disk cap "
                            "during warmup and was killed"
                        ) from None
                    raise exc
                # Only now does the eval input exist in the (already-mounted) scratch dir.
                (scratch_dir / "in.bin").write_bytes(input_data)
                exec_cmd = ["exec", "--user", self.container_user, name, *argv]
                start = time.perf_counter()
                try:
                    proc = subprocess.run(
                        [self.docker_bin, *exec_cmd],
                        timeout=caps.wall_clock_secs, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise RunnerError(f"entrypoint timed out after {caps.wall_clock_secs:.0f}s") from exc
                elapsed = time.perf_counter() - start
                if proc.returncode != 0:
                    if watchdog.triggered:
                        raise RunnerError(
                            f"codec exceeded the {self.scratch_cap_bytes:,}-byte scratch disk cap and was killed"
                        )
                    tail = proc.stderr.decode("utf-8", "replace")[-500:]
                    raise RunnerError(f"entrypoint exited {proc.returncode}: {tail}")
            return elapsed
        finally:
            self._docker(["rm", "-f", name], timeout=15, check=False)
            self._docker(["network", "rm", network], timeout=15, check=False)

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
        # The codec process runs as a non-root uid. The host temp dir is otherwise owned by the
        # validator user, so make only this ephemeral scratch mount writable to the sandbox uid.
        scratch_dir.chmod(0o777)
        _allow_sandbox_read_tree(artifact_dir)
        cmd = [
            self.docker_bin, "run", "--rm", "--name", name,
            "-w", _CONTAINER_ARTIFACT_DIR,
            "-v", f"{artifact_dir}:{_CONTAINER_ARTIFACT_DIR}:ro",
            "-v", f"{scratch_dir}:{_CONTAINER_SCRATCH_DIR}",
            "--memory", str(caps.ram_bytes),
            "--pids-limit", "512",
            "--user", self.container_user,
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
            # Per-file kernel cap (RLIMIT_FSIZE); the background _DiskWatchdog below covers
            # the many-small-files case this alone wouldn't (issue #54).
            "--ulimit", f"fsize={self.scratch_cap_bytes}",
        ]
        if self.seccomp_profile:
            cmd += ["--security-opt", f"seccomp={self.seccomp_profile}"]
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
        with _DiskWatchdog(self.docker_bin, name, scratch_dir, self.scratch_cap_bytes) as watchdog:
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
                if watchdog.triggered:
                    raise RunnerError(
                        f"codec exceeded the {self.scratch_cap_bytes:,}-byte scratch disk cap and was killed"
                    )
                tail = proc.stderr.decode("utf-8", "replace")[-500:]
                raise RunnerError(f"entrypoint exited {proc.returncode}: {tail}")
            return elapsed
