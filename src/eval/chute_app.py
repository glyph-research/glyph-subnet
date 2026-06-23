"""The glyph-runner Chute (DESIGN §6): a deployed Chutes (SN64) endpoint.

One cord, ``run_stream``, performs an entire compress->decompress round-trip for a single
stream on a single worker -- which is what lets Glyph require only same-system
determinism. The endpoint is pinned to the reference GPU SKU via ``NodeSelector.include``
so every validator measures identical compressed bytes.

Deploy with:  chutes deploy eval.chute_app:chute --accept-fee

Invocation contract (validated against the live Chutes API):
- ``POST {base}/run_stream`` with ``Authorization: Basic <cpk_...>``; the request body is
  ``RunStreamRequest`` and the reply is a JSON dump of ``StreamResultModel``.
- A cord MUST return a JSON-serializable dict, not a pydantic model -- returning a model
  raises ``TypeError: Type is not JSON serializable`` in the response serializer, which the
  gateway surfaces as a misleading 500 "No infrastructure available".
``tests/test_chute_contract.py`` pins this binding offline (no GPU); ``scripts/smoke_chute.py``
runs a live round-trip (inline + url/range) against a deployed instance. The boundaries stay
isolated in ``_evaluate`` (here) and ``runner_chutes.ChutesRunner`` (dispatch).
"""

from __future__ import annotations

import base64

from pydantic import BaseModel

from core.constants import (
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


class RunStreamRequest(BaseModel):
    artifact: ArtifactSpec
    stream: StreamSource
    wall_clock_secs: float = 3600.0


class StreamResultModel(BaseModel):
    stream_id: str
    raw_bytes: int
    compressed_bytes: int
    roundtrip_ok: bool
    compress_secs: float
    decompress_secs: float
    blob_hash: str
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


def _evaluate(req: RunStreamRequest) -> StreamResultModel:
    """Run one stream's round-trip on this worker. Reuses the local runner inside the chute."""

    from huggingface_hub import snapshot_download

    from core.artifact import hash_artifact
    from eval.runner import (
        ArtifactRef,
        LocalSubprocessRunner,
        ResourceCaps,
        RunnerError,
        StreamInput,
    )

    data = _materialize(req.stream)
    local = snapshot_download(repo_id=req.artifact.repo, revision=req.artifact.rev)
    digest, _total = hash_artifact(local)

    def failure(error: str) -> StreamResultModel:
        return StreamResultModel(
            stream_id=req.stream.stream_id,
            raw_bytes=len(data),
            compressed_bytes=0,
            roundtrip_ok=False,
            compress_secs=0.0,
            decompress_secs=0.0,
            blob_hash="",
            artifact_hash=digest,
            error=error,
        )

    if req.artifact.sha256 and digest != req.artifact.sha256:
        return failure(f"artifact hash mismatch: got {digest}, expected {req.artifact.sha256}")

    runner = LocalSubprocessRunner()
    artifact = ArtifactRef(repo=req.artifact.repo, rev=req.artifact.rev, sha256=digest, local_path=local)
    try:
        result = runner.run_stream(
            artifact, StreamInput(req.stream.stream_id, data),
            caps=ResourceCaps(wall_clock_secs=req.wall_clock_secs),
        )
    except RunnerError as exc:
        return failure(str(exc))

    return StreamResultModel(
        stream_id=result.stream_id,
        raw_bytes=result.raw_bytes,
        compressed_bytes=result.compressed_bytes,
        roundtrip_ok=result.roundtrip_ok,
        compress_secs=result.compress_secs,
        decompress_secs=result.decompress_secs,
        blob_hash=result.blob_hash,
        artifact_hash=digest,
    )


def build_image():
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


def build_chute():
    if Chute is None:
        raise RuntimeError("chutes SDK is not installed; `pip install chutes`")
    image = build_image()
    chute = Chute(
        username=CHUTE_USERNAME,
        name=CHUTE_NAME,
        image=image,
        node_selector=NodeSelector(
            gpu_count=1,
            min_vram_gb_per_gpu=REFERENCE_MIN_VRAM_GB,
            include=[REFERENCE_SKU],  # reference-SKU pin: identical bytes across validators
        ),
        # Capacity gating on the invocation gateway is concurrency * max_instances; with the
        # default 1*1 a single in-flight (or stale-tracked) request trips a 429 "at maximum
        # capacity". A little headroom avoids that. (NB: REFERENCE_SKU must equal the SKU the
        # subnet mandates -- e.g. an integrated SN64 subnet forces include=['pro_6000'].)
        concurrency=4,
    )

    @chute.cord(public_api_path="/run_stream", method="POST")
    async def run_stream(self, req: RunStreamRequest) -> dict:  # noqa: ANN001
        # Two things the Chutes runtime requires that are easy to get wrong:
        # 1) the round-trip is blocking (snapshot_download + subprocess codec run) -- run it
        #    OFF the event loop, else health-check pings stall and the instance is reaped;
        # 2) return a plain JSON-serializable dict -- the response serializer cannot encode a
        #    raw pydantic model (TypeError: Type is not JSON serializable: StreamResultModel),
        #    which surfaces to the caller as a misleading 500 "No infrastructure available".
        import asyncio

        result = await asyncio.to_thread(_evaluate, req)
        return result.model_dump()

    return chute


# Module-level handle for `chutes deploy eval.chute_app:chute`. Built lazily-safe:
# importing this module must never fail (e.g. when the build context / cwd is not the repo
# root). `chutes build/deploy`, run from the repo root, constructs it successfully.
try:
    chute = build_chute() if Chute is not None else None
except Exception:
    chute = None
