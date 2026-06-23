"""The glyph-runner invocation contract, pinned offline (issue #7).

These bind the validator side (``eval.runner_chutes.ChutesRunner``) to the deployed-chute
side (``eval.chute_app``) without a live GPU: the request the runner builds must validate as
the chute's ``RunStreamRequest``, and the reply the chute returns (a ``StreamResultModel``
dump) must parse back into a ``StreamResult``. A live end-to-end round-trip lives in
``scripts/smoke_chute.py``.
"""

import base64

from eval.chute_app import ArtifactSpec, RunStreamRequest, StreamResultModel, StreamSource, _materialize
from eval.runner import ArtifactRef, ResourceCaps, StreamInput
from eval.runner_chutes import ChutesRunner
from eval.streams import RangeSource

ART = ArtifactRef(repo="org/codec", rev="r1", sha256="abc")
CAPS = ResourceCaps(wall_clock_secs=12.0)


def _request_for(stream_input: StreamInput) -> RunStreamRequest:
    # raises if the request the runner builds no longer matches the chute's request model
    return RunStreamRequest.model_validate(ChutesRunner._build_payload(ART, stream_input, CAPS))


# --- request side: runner payload validates as the chute's RunStreamRequest -------

def test_inline_payload_matches_request_model():
    req = _request_for(StreamInput("s0", data=b"hello"))
    assert req.artifact == ArtifactSpec(repo="org/codec", rev="r1", sha256="abc")
    assert req.stream.inline_b64 == base64.b64encode(b"hello").decode("ascii")
    assert req.stream.url is None
    assert req.wall_clock_secs == 12.0


def test_url_range_payload_matches_request_model():
    req = _request_for(StreamInput("s1", source=RangeSource("https://c/corpus.bin", 1024, 4096)))
    assert (req.stream.url, req.stream.offset, req.stream.length) == ("https://c/corpus.bin", 1024, 4096)
    assert req.stream.inline_b64 is None


def test_chute_materializes_the_inline_shape_the_runner_sends():
    payload = ChutesRunner._build_payload(ART, StreamInput("s0", data=b"abc"), CAPS)
    src = StreamSource.model_validate(payload["stream"])
    assert _materialize(src) == b"abc"


# --- response side: chute reply parses back into a StreamResult -------------------

def test_success_reply_round_trips_through_runner():
    dumped = StreamResultModel(
        stream_id="s0", raw_bytes=100, compressed_bytes=40, roundtrip_ok=True,
        compress_secs=0.2, decompress_secs=0.1, blob_hash="deadbeef",
    ).model_dump()
    result = ChutesRunner._parse_response(dumped, StreamInput("s0", data=b"x" * 100))
    assert (result.stream_id, result.raw_bytes, result.compressed_bytes) == ("s0", 100, 40)
    assert result.roundtrip_ok is True
    assert result.blob_hash == "deadbeef"


def test_error_reply_scored_as_failed_roundtrip_with_known_length():
    dumped = StreamResultModel(
        stream_id="s2", raw_bytes=4096, compressed_bytes=0, roundtrip_ok=False,
        compress_secs=0.0, decompress_secs=0.0, blob_hash="", error="codec exited 1",
    ).model_dump()
    # remote stream (no inline bytes) -> raw length comes from the reply / known range length
    result = ChutesRunner._parse_response(dumped, StreamInput("s2", source=RangeSource("https://c", 0, 4096)))
    assert result.roundtrip_ok is False
    assert result.compressed_bytes == 0
    assert result.raw_bytes == 4096
