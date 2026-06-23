"""ChutesRunner: dispatch a same-worker round-trip to the deployed glyph-runner chute.

Implements the same ``CodecRunner`` contract as ``LocalSubprocessRunner`` but executes
the codec on Chutes (SN64) serverless GPU instead of the host, so a key-holding validator
never runs untrusted artifacts locally. Auth uses a ``cpk_`` API key.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import requests

from core.constants import CHUTE_NAME, CHUTE_USERNAME, REFERENCE_SKU
from eval.runner import ArtifactRef, ResourceCaps, RunnerError, StreamInput
from eval.scoring import StreamResult

# The public invocation URL pattern should be confirmed on first deploy; override via
# --chute-url / GLYPH_CHUTE_URL.
DEFAULT_BASE_URL = f"https://{CHUTE_USERNAME}-{CHUTE_NAME}.chutes.ai"


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
    prefers_remote_source = True  # the chute range-fetches the corpus; don't inline bytes

    def __init__(
        self,
        *,
        reference_sku: str = REFERENCE_SKU,
        key_file: str | None = None,
        base_url: str | None = None,
        timeout: float = 3600.0,
    ):
        self.reference_sku = reference_sku
        self.base_url = (base_url or os.environ.get("GLYPH_CHUTE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.api_key = _load_api_key(key_file)

    @staticmethod
    def _build_payload(artifact: ArtifactRef, stream: StreamInput, caps: ResourceCaps) -> dict:
        """Select the stream dispatch shape: range-fetched URL (production) or inline bytes.

        When the stream carries a ``source`` the chute range-fetches the corpus itself
        (DESIGN: fetch before isolation) so the validator never inlines (and re-uploads) the
        bytes. Otherwise the bytes go inline -- the path kept for tests / local smoke / small
        streams. The chute's ``StreamSource`` accepts either shape.
        """

        if stream.source is not None:
            stream_payload = {
                "stream_id": stream.stream_id,
                "url": stream.source.url,
                "offset": stream.source.offset,
                "length": stream.source.length,
            }
        elif stream.data is not None:
            stream_payload = {
                "stream_id": stream.stream_id,
                "inline_b64": base64.b64encode(stream.data).decode("ascii"),
            }
        else:
            raise RunnerError("stream has neither inline data nor a remote source")
        return {
            "artifact": {"repo": artifact.repo, "rev": artifact.rev, "sha256": artifact.sha256},
            "stream": stream_payload,
            "wall_clock_secs": caps.wall_clock_secs,
        }

    def run_stream(
        self, artifact: ArtifactRef, stream: StreamInput, *, caps: ResourceCaps | None = None
    ) -> StreamResult:
        caps = caps or ResourceCaps()
        payload = self._build_payload(artifact, stream, caps)
        url = f"{self.base_url}/run_stream"
        # Chutes invocation auth is `Authorization: Basic <cpk_...>` (per `chutes keys create`);
        # sending the bare key 401s.
        auth = self.api_key if self.api_key.startswith("Basic ") else f"Basic {self.api_key}"
        try:
            response = requests.post(
                url, json=payload, headers={"Authorization": auth}, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # network/HTTP errors are dispatch failures
            raise RunnerError(f"chute dispatch failed: {exc}") from exc

        if data.get("error"):
            # A worker-side codec failure is an invalid stream, not a dispatch error.
            return StreamResult(
                stream_id=stream.stream_id,
                raw_bytes=int(data.get("raw_bytes", stream.raw_len)),
                compressed_bytes=0,
                roundtrip_ok=False,
                compress_secs=0.0,
                decompress_secs=0.0,
                blob_hash="",
            )
        return StreamResult(
            stream_id=data["stream_id"],
            raw_bytes=data["raw_bytes"],
            compressed_bytes=data["compressed_bytes"],
            roundtrip_ok=data["roundtrip_ok"],
            compress_secs=data["compress_secs"],
            decompress_secs=data["decompress_secs"],
            blob_hash=data.get("blob_hash", ""),
        )
