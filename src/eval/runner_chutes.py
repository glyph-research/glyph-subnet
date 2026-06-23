"""ChutesRunner: dispatch one stream across the SEPARATE compress and decompress chutes.

Implements the ``CodecRunner`` contract but runs the codec on Chutes (SN64) serverless GPU
instead of the host, so a key-holding validator never executes untrusted artifacts locally.
Compress and decompress hit two different deployed chutes (separate containers): the blob
from ``/compress`` is the *only* thing passed to ``/decompress``, so a codec cannot stash the
raw input and read it back to fake the ratio (exploit-prevention #14). Auth uses a ``cpk_``
API key. Bit-exactness is gated on the decompress worker's output hash matching the hash the
validator computed from the trusted corpus (``StreamInput.expected_sha256``), never a
worker's self-report.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import requests

from core.constants import (
    CHUTE_COMPRESSOR_NAME,
    CHUTE_DECOMPRESSOR_NAME,
    CHUTE_USERNAME,
    REFERENCE_SKU,
)
from eval.runner import ArtifactRef, ResourceCaps, RunnerError, StreamInput
from eval.scoring import StreamResult

# Confirmed live invocation contract (auth Authorization: Basic <cpk_...>; a cord returns a
# JSON dict). Override per deployment via --compress-chute-url / --decompress-chute-url or
# GLYPH_COMPRESS_CHUTE_URL / GLYPH_DECOMPRESS_CHUTE_URL.
DEFAULT_COMPRESS_URL = f"https://{CHUTE_USERNAME}-{CHUTE_COMPRESSOR_NAME}.chutes.ai"
DEFAULT_DECOMPRESS_URL = f"https://{CHUTE_USERNAME}-{CHUTE_DECOMPRESSOR_NAME}.chutes.ai"


def _load_api_key(key_file: str | None) -> str:
    """Load the Chutes ``cpk_`` invocation key from the environment (or an explicit file).

    The key comes from ``CHUTES_API_KEY`` (set it in a ``.env`` file -- see
    ``.env.example``). ``--chutes-key-file`` may point at a file containing the raw key.
    """

    if key_file:
        path = Path(key_file)
        if not path.is_file():
            raise RunnerError(f"--chutes-key-file not found: {key_file}")
        value = path.read_text().strip()
        if value:
            return value

    env = os.environ.get("CHUTES_API_KEY")
    if env and env.strip():
        return env.strip()

    raise RunnerError(
        "no Chutes API key found. Set CHUTES_API_KEY in a .env file (see .env.example) or "
        "pass --chutes-key-file. Create a key with `chutes keys create --name "
        "glyph-validator --admin`."
    )


class ChutesRunner:
    prefers_remote_source = True  # the compress chute range-fetches the corpus; don't inline bytes

    def __init__(
        self,
        *,
        reference_sku: str = REFERENCE_SKU,
        key_file: str | None = None,
        compress_base_url: str | None = None,
        decompress_base_url: str | None = None,
        timeout: float = 3600.0,
    ):
        self.reference_sku = reference_sku
        self.compress_base_url = (
            compress_base_url or os.environ.get("GLYPH_COMPRESS_CHUTE_URL") or DEFAULT_COMPRESS_URL
        ).rstrip("/")
        self.decompress_base_url = (
            decompress_base_url or os.environ.get("GLYPH_DECOMPRESS_CHUTE_URL") or DEFAULT_DECOMPRESS_URL
        ).rstrip("/")
        self.timeout = timeout
        self.api_key = _load_api_key(key_file)

    @staticmethod
    def _stream_payload(stream: StreamInput) -> dict:
        """The compress request's stream shape: range-fetched URL (production) or inline bytes."""

        if stream.source is not None:
            return {
                "stream_id": stream.stream_id,
                "url": stream.source.url,
                "offset": stream.source.offset,
                "length": stream.source.length,
            }
        if stream.data is not None:
            return {
                "stream_id": stream.stream_id,
                "inline_b64": base64.b64encode(stream.data).decode("ascii"),
            }
        raise RunnerError("stream has neither inline data nor a remote source")

    @staticmethod
    def _artifact_payload(artifact: ArtifactRef) -> dict:
        return {"repo": artifact.repo, "rev": artifact.rev, "sha256": artifact.sha256}

    @classmethod
    def _compress_request(cls, artifact: ArtifactRef, stream: StreamInput, caps: ResourceCaps) -> dict:
        return {
            "artifact": cls._artifact_payload(artifact),
            "stream": cls._stream_payload(stream),
            "wall_clock_secs": caps.wall_clock_secs,
        }

    @classmethod
    def _decompress_request(
        cls, artifact: ArtifactRef, stream_id: str, blob_b64: str, caps: ResourceCaps
    ) -> dict:
        return {
            "artifact": cls._artifact_payload(artifact),
            "stream_id": stream_id,
            "blob_b64": blob_b64,
            "wall_clock_secs": caps.wall_clock_secs,
        }

    def _post(self, url: str, payload: dict) -> dict:
        # Chutes invocation auth is `Authorization: Basic <cpk_...>`; the bare key 401s.
        auth = self.api_key if self.api_key.startswith("Basic ") else f"Basic {self.api_key}"
        try:
            response = requests.post(
                url, json=payload, headers={"Authorization": auth}, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # network/HTTP errors are dispatch failures
            raise RunnerError(f"chute dispatch failed ({url}): {exc}") from exc

    def run_stream(
        self, artifact: ArtifactRef, stream: StreamInput, *, caps: ResourceCaps | None = None
    ) -> StreamResult:
        caps = caps or ResourceCaps()
        comp = self._post(
            f"{self.compress_base_url}/compress", self._compress_request(artifact, stream, caps)
        )
        if comp.get("error"):
            return self._failed(stream, int(comp.get("raw_bytes", stream.raw_len)))

        decomp = self._post(
            f"{self.decompress_base_url}/decompress",
            self._decompress_request(artifact, stream.stream_id, comp["blob_b64"], caps),
        )
        if decomp.get("error"):
            return self._failed(stream, int(comp.get("raw_bytes", stream.raw_len)))

        # Gate bit-exactness on the validator-anchored hash; only fall back to the compress
        # worker's self-reported source hash when no trusted hash was supplied (local smoke).
        expected = stream.expected_sha256 or comp.get("source_sha256", "")
        roundtrip_ok = bool(expected) and decomp.get("output_sha256") == expected
        return StreamResult(
            stream_id=stream.stream_id,
            raw_bytes=int(comp["raw_bytes"]),
            compressed_bytes=int(comp["compressed_bytes"]),
            roundtrip_ok=roundtrip_ok,
            compress_secs=float(comp["compress_secs"]),
            decompress_secs=float(decomp["decompress_secs"]),
            blob_hash=comp.get("blob_hash", ""),
        )

    @staticmethod
    def _failed(stream: StreamInput, raw_bytes: int) -> StreamResult:
        """A worker-side codec failure is an invalid stream, not a dispatch error."""

        return StreamResult(
            stream_id=stream.stream_id,
            raw_bytes=raw_bytes,
            compressed_bytes=0,
            roundtrip_ok=False,
            compress_secs=0.0,
            decompress_secs=0.0,
            blob_hash="",
        )
