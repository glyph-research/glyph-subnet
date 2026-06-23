"""The split compress/decompress chute contract, pinned offline (#14).

Binds the validator side (``eval.runner_chutes.ChutesRunner``) to the deployed chutes
(``eval.chute_app``) without a live GPU: the requests the runner builds must validate as the
chutes' request models, and the chutes' reply models must carry the fields the runner reads.
A live split round-trip lives in ``scripts/smoke_chute.py``.
"""

import base64

from eval.chute_app import (
    ArtifactSpec,
    CompressRequest,
    CompressResultModel,
    DecompressRequest,
    DecompressResultModel,
    StreamSource,
    _materialize,
)
from eval.runner import ArtifactRef, ResourceCaps, StreamInput
from eval.runner_chutes import ChutesRunner
from eval.streams import RangeSource

ART = ArtifactRef(repo="org/codec", rev="r1", sha256="abc")
CAPS = ResourceCaps(wall_clock_secs=12.0)


# --- request side: runner payloads validate as the chutes' request models --------

def test_compress_request_inline_validates():
    payload = ChutesRunner._compress_request(ART, StreamInput("s0", data=b"hi"), CAPS)
    req = CompressRequest.model_validate(payload)
    assert req.artifact == ArtifactSpec(repo="org/codec", rev="r1", sha256="abc")
    assert req.stream.inline_b64 == base64.b64encode(b"hi").decode("ascii")
    assert req.stream.url is None
    assert req.wall_clock_secs == 12.0


def test_compress_request_url_validates():
    payload = ChutesRunner._compress_request(
        ART, StreamInput("s1", source=RangeSource("https://c/corpus.bin", 1024, 4096)), CAPS
    )
    req = CompressRequest.model_validate(payload)
    assert (req.stream.url, req.stream.offset, req.stream.length) == ("https://c/corpus.bin", 1024, 4096)
    assert req.stream.inline_b64 is None


def test_decompress_request_validates():
    payload = ChutesRunner._decompress_request(ART, "s0", "QkxPQg==", CAPS)
    req = DecompressRequest.model_validate(payload)
    assert req.stream_id == "s0"
    assert req.blob_b64 == "QkxPQg=="
    assert req.artifact == ArtifactSpec(repo="org/codec", rev="r1", sha256="abc")


def test_chute_materializes_the_inline_shape_the_runner_sends():
    payload = ChutesRunner._compress_request(ART, StreamInput("s0", data=b"abc"), CAPS)
    src = StreamSource.model_validate(payload["stream"])
    assert _materialize(src) == b"abc"


# --- reply side: the runner reads what the chutes return -------------------------

def test_compress_reply_carries_blob_and_hashes():
    dump = CompressResultModel(
        stream_id="s0", raw_bytes=100, compressed_bytes=40, compress_secs=0.2,
        blob_b64="QQ==", blob_hash="bh", source_sha256="sh",
    ).model_dump()
    # exactly the fields ChutesRunner.run_stream reads off the compress reply
    for key in ("raw_bytes", "compressed_bytes", "compress_secs", "blob_b64", "blob_hash", "source_sha256"):
        assert key in dump
    assert dump["blob_b64"] == "QQ==" and dump["source_sha256"] == "sh"


def test_decompress_reply_carries_output_hash():
    dump = DecompressResultModel(
        stream_id="s0", raw_bytes=100, decompress_secs=0.1, output_sha256="oh",
    ).model_dump()
    assert dump["output_sha256"] == "oh"
    assert dump["decompress_secs"] == 0.1
