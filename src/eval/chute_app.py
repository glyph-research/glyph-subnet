"""The glyph eval chutes (DESIGN §6): TWO deployed Chutes (SN64) endpoints.

Compress and decompress run on **separate** chutes -- ``/compress`` (CHUTE_COMPRESSOR_NAME)
and ``/decompress`` (CHUTE_DECOMPRESSOR_NAME) -- so they execute in separate containers. The
decompress worker only ever receives the compressed blob, never the raw input, so a codec
cannot stash the raw bytes during compress and read them back during decompress to fake the
ratio (exploit-prevention #14). Both pin the reference GPU SKU via ``NodeSelector.include``
so every validator measures identical compressed bytes.

Deploy both (separately):
  chutes deploy eval.chute_app:compressor_chute   --accept-fee
  chutes deploy eval.chute_app:decompressor_chute --accept-fee

Invocation contract (validated against the live Chutes API):
- ``POST {base}/compress``   with ``Authorization: Basic <cpk_...>`` -> ``CompressRequest`` /
  ``CompressResultModel`` (carries ``blob_b64``).
- ``POST {base}/decompress`` with the same auth -> ``DecompressRequest`` (the blob) /
  ``DecompressResultModel``. The validator gates bit-exactness on ``output_sha256`` against the
  hash it computed from the trusted corpus -- never the worker's self-report.
- A cord MUST return a JSON-serializable dict, not a pydantic model -- returning a model raises
  ``TypeError: Type is not JSON serializable``, surfaced as a misleading 500 "No infrastructure".
``tests/test_chute_contract.py`` pins the binding offline; ``scripts/smoke_chute.py`` runs a
live split round-trip. The boundaries stay isolated here and in ``runner_chutes.ChutesRunner``.
"""

from __future__ import annotations

import base64
from typing import Any

from pydantic import BaseModel

from core.constants import (
    CHUTE_COMPRESSOR_NAME,
    CHUTE_DECOMPRESSOR_NAME,
    CHUTE_NAME,
    CHUTE_USERNAME,
    REFERENCE_MIN_VRAM_GB,
    REFERENCE_SKU,
)

try:  # chutes SDK is only needed to build/deploy/run the chute
    from chutes.chute import Chute, NodeSelector
    from chutes.image import Image
except Exception:  # pragma: no cover - environment without chutes
    Chute = NodeSelector = Image = None


class ArtifactSpec(BaseModel):
    repo: str
    rev: str
    sha256: str | None = None


class StreamSource(BaseModel):
    stream_id: str
    inline_b64: str | None = None  # small/testing path
    url: str | None = None  # production: HTTP range fetch from the corpus
    offset: int = 0
    length: int = 0


class CompressRequest(BaseModel):
    artifact: ArtifactSpec
    stream: StreamSource
    wall_clock_secs: float = 3600.0


class CompressResultModel(BaseModel):
    stream_id: str
    raw_bytes: int
    compressed_bytes: int
    compress_secs: float
    blob_b64: str  # the compressed artifact, handed to a *separate* decompressor chute
    blob_hash: str
    source_sha256: str  # the compress worker's view of the raw hash (validator cross-checks)
    artifact_hash: str | None = None
    error: str | None = None


class DecompressRequest(BaseModel):
    artifact: ArtifactSpec
    stream_id: str
    blob_b64: str
    wall_clock_secs: float = 3600.0


class DecompressResultModel(BaseModel):
    stream_id: str
    raw_bytes: int
    decompress_secs: float
    output_sha256: str  # sha256 of the reconstructed bytes; the validator gates on this
    artifact_hash: str | None = None
    error: str | None = None


def _materialize(src: StreamSource) -> bytes:
    if src.inline_b64 is not None:
        return base64.b64decode(src.inline_b64)
    if src.url:
        import requests

        headers = {}
        if src.length:
            headers["Range"] = f"bytes={src.offset}-{src.offset + src.length - 1}"
        response = requests.get(src.url, headers=headers, timeout=120)
        response.raise_for_status()
        return response.content
    raise ValueError("stream source needs inline_b64 or url")


def _prepare_artifact(spec: ArtifactSpec) -> tuple[str, str]:
    """snapshot_download + re-precheck + hash check on the worker. Returns (local_path, digest).

    Raises ``ValueError`` (with the precheck/hash reason) so the cord can report it as a failed
    stream rather than a dispatch error.
    """

    from huggingface_hub import snapshot_download

    from validation.precheck import precheck_artifact_dir

    local = snapshot_download(repo_id=spec.repo, revision=spec.rev)
    precheck = precheck_artifact_dir(local, spec.repo, spec.rev)
    digest = precheck.artifact_hash
    if not precheck.ok:
        raise ValueError("artifact precheck failed: " + "; ".join(precheck.errors))
    if spec.sha256 and digest != spec.sha256:
        raise ValueError(f"artifact hash mismatch: got {digest}, expected {spec.sha256}")
    return local, digest


def _compress(req: CompressRequest) -> CompressResultModel:
    """Run only the compress entrypoint on this worker; return the blob for the decompressor."""

    from eval.runner import ArtifactRef, LocalSubprocessRunner, ResourceCaps, RunnerError

    data = _materialize(req.stream)
    try:
        local, digest = _prepare_artifact(req.artifact)
    except ValueError as exc:
        return CompressResultModel(
            stream_id=req.stream.stream_id, raw_bytes=len(data), compressed_bytes=0,
            compress_secs=0.0, blob_b64="", blob_hash="", source_sha256="", error=str(exc),
        )
    runner = LocalSubprocessRunner(strict_sandbox=True, require_network_isolation=True)
    artifact = ArtifactRef(repo=req.artifact.repo, rev=req.artifact.rev, sha256=digest, local_path=local)
    try:
        out = runner.compress(artifact, data, caps=ResourceCaps(wall_clock_secs=req.wall_clock_secs))
    except RunnerError as exc:
        return CompressResultModel(
            stream_id=req.stream.stream_id, raw_bytes=len(data), compressed_bytes=0,
            compress_secs=0.0, blob_b64="", blob_hash="", source_sha256="",
            artifact_hash=digest, error=str(exc),
        )
    return CompressResultModel(
        stream_id=req.stream.stream_id,
        raw_bytes=out.raw_bytes,
        compressed_bytes=out.compressed_bytes,
        compress_secs=out.compress_secs,
        blob_b64=base64.b64encode(out.blob).decode("ascii"),
        blob_hash=out.blob_hash,
        source_sha256=out.source_hash,
        artifact_hash=digest,
    )


def _decompress(req: DecompressRequest) -> DecompressResultModel:
    """Run only the decompress entrypoint on this worker, seeded with only the blob."""

    from eval.runner import ArtifactRef, LocalSubprocessRunner, ResourceCaps, RunnerError

    try:
        local, digest = _prepare_artifact(req.artifact)
    except ValueError as exc:
        return DecompressResultModel(
            stream_id=req.stream_id, raw_bytes=0, decompress_secs=0.0, output_sha256="", error=str(exc),
        )
    runner = LocalSubprocessRunner(strict_sandbox=True, require_network_isolation=True)
    artifact = ArtifactRef(repo=req.artifact.repo, rev=req.artifact.rev, sha256=digest, local_path=local)
    try:
        blob = base64.b64decode(req.blob_b64)
        out = runner.decompress(artifact, blob, caps=ResourceCaps(wall_clock_secs=req.wall_clock_secs))
    except RunnerError as exc:
        return DecompressResultModel(
            stream_id=req.stream_id, raw_bytes=0, decompress_secs=0.0, output_sha256="",
            artifact_hash=digest, error=str(exc),
        )
    return DecompressResultModel(
        stream_id=req.stream_id,
        raw_bytes=out.raw_bytes,
        decompress_secs=out.decompress_secs,
        output_sha256=out.output_hash,
        artifact_hash=digest,
    )


def build_image() -> "Image":
    if Image is None:
        raise RuntimeError("chutes SDK is not installed; `pip install chutes`")
    return (
        Image(username=CHUTE_USERNAME, name=CHUTE_NAME, tag="0.1")
        .from_base("parachutes/python:3.12")
        .apt_install("zstd")
        .run_command("pip install zstandard huggingface_hub requests 'pydantic>=2.8'")
        .add("src", "/app/src")  # copies all service packages; run `chutes build` from repo root
        .set_workdir("/app")
        .with_env("PYTHONPATH", "/app/src")
    )


def _build_chute(name: str):
    if Chute is None:
        raise RuntimeError("chutes SDK is not installed; `pip install chutes`")
    return Chute(
        username=CHUTE_USERNAME,
        name=name,
        image=build_image(),
        node_selector=NodeSelector(
            gpu_count=1,
            min_vram_gb_per_gpu=REFERENCE_MIN_VRAM_GB,
            include=[REFERENCE_SKU],  # reference-SKU pin: identical bytes across validators
        ),
        # Chutes mandates confidential (TEE) execution as of 2026-05-12 -- non-TEE deploys are
        # rejected ("Only TEE chutes are supported"). TEE also gives the attestation that the
        # reference SKU / image actually ran, reinforcing same-system determinism.
        tee=True,
        # Keep a warmed instance alive 1h past its last request. Without this the chute inherits
        # the platform default (~5 min idle) and goes cold between sporadic eval invocations, so
        # the next validator round hits a cold start and the gateway 500s/429s ("no infrastructure
        # available") until an instance re-warms. 3600s comfortably bridges normal round gaps
        # without holding the GPU indefinitely when validation is idle.
        shutdown_after_seconds=3600,
        # Invocation-gateway capacity is concurrency * max_instances; too low and concurrent
        # validator dispatches 429. 16 gives headroom for bursty multi-stream rounds. (The codec
        # subprocess still enforces the per-run VRAM/RAM caps, so concurrency is a dispatch limit,
        # not a resource override.)
        concurrency=16,
    )


def build_compressor_chute():
    """The compress-only chute. Deploy: chutes deploy eval.chute_app:compressor_chute --accept-fee"""

    chute = _build_chute(CHUTE_COMPRESSOR_NAME)

    @chute.cord(public_api_path="/compress", method="POST")
    async def compress(self, req: CompressRequest) -> dict[str, Any]:  # noqa: ANN001
        # Return a plain JSON dict: a pydantic model trips the serializer and surfaces as a
        # misleading 500 "No infrastructure available".
        import asyncio

        result = await asyncio.to_thread(_compress, req)
        return result.model_dump()

    return chute


def build_decompressor_chute():
    """The decompress-only chute. Deploy: chutes deploy eval.chute_app:decompressor_chute --accept-fee"""

    chute = _build_chute(CHUTE_DECOMPRESSOR_NAME)

    @chute.cord(public_api_path="/decompress", method="POST")
    async def decompress(self, req: DecompressRequest) -> dict[str, Any]:  # noqa: ANN001
        import asyncio

        result = await asyncio.to_thread(_decompress, req)
        return result.model_dump()

    return chute


# Module-level handles for `chutes deploy eval.chute_app:compressor_chute` /
# `:decompressor_chute`. Built lazily-safe: importing this module must never fail (e.g. when
# the build context / cwd is not the repo root). Deploy both as SEPARATE chutes so compress
# and decompress run in separate containers (exploit-prevention #14).
try:
    compressor_chute = build_compressor_chute() if Chute is not None else None
    decompressor_chute = build_decompressor_chute() if Chute is not None else None
except Exception:
    compressor_chute = decompressor_chute = None
