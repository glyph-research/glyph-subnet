"""The split compress/decompress chute contract, pinned offline (#14).

Binds the validator side (``eval.runner_chutes.ChutesRunner``) to the deployed chutes
(``eval.chute_app`` + ``eval.glyph_eval_runner``) without a live GPU: the requests the runner
builds must validate as the chute's single ``EvalRequest`` model, and the runner's actual reply
dict shape (``run_compress``/``run_decompress``) must carry the fields ``ChutesRunner.run_stream``
reads. A live split round-trip lives in ``scripts/smoke_chute.py``.

NOTE: chute_app.py deliberately defines only ONE pydantic BaseModel (``EvalRequest``, flat dict
fields for artifact/stream) -- the chutes-TEE code-verification (aegis cllmv) trips on a module
with several small request/response models, so the per-field typed models this test used to bind
against (``ArtifactSpec``/``StreamSource``/``CompressRequest``/etc.) were collapsed. See
chute_app.py's module docstring for the full diagnosis.
"""

import base64

from eval.chute_app import EvalRequest
from eval.glyph_eval_runner import _materialize, run_compress, run_decompress
from eval.runner import ArtifactRef, ResourceCaps, StreamInput
from eval.runner_chutes import ChutesRunner
from eval.streams import RangeSource

ART = ArtifactRef(repo="org/codec", rev="r1", sha256="abc")
CAPS = ResourceCaps(wall_clock_secs=12.0)


# --- request side: runner payloads validate as the chute's EvalRequest model ------

def test_compress_request_inline_validates():
    payload = ChutesRunner._compress_request(ART, StreamInput("s0", data=b"hi"), CAPS)
    req = EvalRequest.model_validate(payload)
    assert req.artifact == {"repo": "org/codec", "rev": "r1", "sha256": "abc"}
    assert req.stream["inline_b64"] == base64.b64encode(b"hi").decode("ascii")
    assert "url" not in req.stream
    assert req.wall_clock_secs == 12.0


def test_compress_request_url_validates():
    payload = ChutesRunner._compress_request(
        ART, StreamInput("s1", source=RangeSource("https://c/corpus.bin", 1024, 4096)), CAPS
    )
    req = EvalRequest.model_validate(payload)
    assert (req.stream["url"], req.stream["offset"], req.stream["length"]) == ("https://c/corpus.bin", 1024, 4096)
    assert "inline_b64" not in req.stream


def test_decompress_request_validates():
    payload = ChutesRunner._decompress_request(ART, "s0", "QkxPQg==", CAPS)
    req = EvalRequest.model_validate(payload)
    assert req.stream_id == "s0"
    assert req.blob_b64 == "QkxPQg=="
    assert req.artifact == {"repo": "org/codec", "rev": "r1", "sha256": "abc"}


def test_chute_materializes_the_inline_shape_the_runner_sends():
    payload = ChutesRunner._compress_request(ART, StreamInput("s0", data=b"abc"), CAPS)
    req = EvalRequest.model_validate(payload)
    assert _materialize(req.stream) == b"abc"


# --- reply side: the runner reads what glyph_eval_runner actually returns --------
# _prepare_artifact always network-fetches first; stub it so this stays offline (per this
# module's docstring) while still exercising the real run_compress/run_decompress reply shape.

def test_compress_reply_carries_blob_and_hashes(monkeypatch):
    monkeypatch.setattr(
        "eval.glyph_eval_runner._prepare_artifact",
        lambda artifact: (_ for _ in ()).throw(ValueError("boom")),
    )
    dump = run_compress(
        {"artifact": {"repo": "org/codec", "rev": "r1"}, "stream": {"stream_id": "s0", "inline_b64": "aGk="}}
    )
    for key in ("raw_bytes", "compressed_bytes", "compress_secs", "blob_b64", "blob_hash", "source_sha256", "error"):
        assert key in dump
    assert dump["error"] == "boom"  # exact reply shape ChutesRunner.run_stream parses off comp.get("error")


def test_decompress_reply_carries_output_hash(monkeypatch):
    monkeypatch.setattr(
        "eval.glyph_eval_runner._prepare_artifact",
        lambda artifact: (_ for _ in ()).throw(ValueError("boom")),
    )
    dump = run_decompress({"artifact": {"repo": "org/codec", "rev": "r1"}, "stream_id": "s0", "blob_b64": "QQ=="})
    for key in ("raw_bytes", "decompress_secs", "output_sha256", "error"):
        assert key in dump
    assert dump["error"] == "boom"
