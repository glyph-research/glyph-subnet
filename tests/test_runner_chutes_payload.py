"""ChutesRunner request building + evaluator/corpus wiring for the split chutes (#3, #14).

Compress and decompress hit two separate chutes; the validator anchors the round-trip hash
from the trusted corpus. These pin the request shapes and the evaluator decision offline.
"""

import base64
import hashlib

import pytest

from eval.corpus import StaticLocalProvider
from eval.evaluator import _prepare_stream
from eval.runner import ArtifactRef, LocalSubprocessRunner, ResourceCaps, RunnerError, StreamInput
from eval.runner_chutes import ChutesRunner
from eval.streams import RangeSource, StreamSpec

ART = ArtifactRef(repo="org/codec", rev="abc123", sha256="deadbeef")
CAPS = ResourceCaps(wall_clock_secs=42.0)


# --- compress request: stream shape selection (inline vs URL/range) --------------

def test_compress_request_inline_when_stream_has_bytes():
    payload = ChutesRunner._compress_request(ART, StreamInput("s0", data=b"hello world"), CAPS)
    stream = payload["stream"]
    assert stream["stream_id"] == "s0"
    assert stream["inline_b64"] == base64.b64encode(b"hello world").decode("ascii")
    assert "url" not in stream
    assert payload["artifact"] == {"repo": "org/codec", "rev": "abc123", "sha256": "deadbeef"}
    assert payload["wall_clock_secs"] == 42.0


def test_compress_request_url_when_stream_has_source():
    stream_in = StreamInput("s1", source=RangeSource("https://c/corpus.bin", 1024, 4096))
    stream = ChutesRunner._compress_request(ART, stream_in, CAPS)["stream"]
    assert stream == {"stream_id": "s1", "url": "https://c/corpus.bin", "offset": 1024, "length": 4096}
    assert "inline_b64" not in stream


def test_stream_payload_requires_data_or_source():
    with pytest.raises(RunnerError):
        ChutesRunner._stream_payload(StreamInput("s2"))


def test_decompress_request_carries_blob():
    payload = ChutesRunner._decompress_request(ART, "s0", "BLOB==", CAPS)
    assert payload == {
        "artifact": {"repo": "org/codec", "rev": "abc123", "sha256": "deadbeef"},
        "stream_id": "s0",
        "blob_b64": "BLOB==",
        "wall_clock_secs": 42.0,
    }


def test_runner_remote_preference_flags():
    assert ChutesRunner.prefers_remote_source is True
    assert LocalSubprocessRunner.prefers_remote_source is False


def test_separate_chute_base_urls(monkeypatch):
    monkeypatch.setenv("CHUTES_API_KEY", "cpk_test")
    runner = ChutesRunner(
        compress_base_url="https://acct-glyph-compressor.chutes.ai/",
        decompress_base_url="https://acct-glyph-decompressor.chutes.ai/",
    )
    assert runner.compress_base_url == "https://acct-glyph-compressor.chutes.ai"
    assert runner.decompress_base_url == "https://acct-glyph-decompressor.chutes.ai"


# --- corpus exposes a remote source only when a base_url is configured ------------

def _corpus(tmp_path, *, base_url=None):
    (tmp_path / "a.txt").write_bytes(b"x" * 100)
    return StaticLocalProvider(tmp_path, base_url=base_url)


def test_stream_source_none_without_base_url(tmp_path):
    assert _corpus(tmp_path).stream_source(StreamSpec("s0", 0, 50)) is None


def test_stream_source_present_with_base_url(tmp_path):
    provider = _corpus(tmp_path, base_url="https://c/corpus.bin")
    assert provider.stream_source(StreamSpec("s0", 10, 30)) == RangeSource(
        url="https://c/corpus.bin", offset=10, length=30
    )


# --- evaluator wires runner + corpus capability and anchors the hash --------------

class _RemoteRunner:
    prefers_remote_source = True


class _LocalRunner:
    prefers_remote_source = False


def test_prepare_stream_remote_anchors_hash(tmp_path):
    provider = _corpus(tmp_path, base_url="https://c/corpus.bin")
    spec = StreamSpec("s0", 10, 30)
    stream_in = _prepare_stream(_RemoteRunner(), provider, spec)
    assert stream_in.data is None
    assert stream_in.source == RangeSource(url="https://c/corpus.bin", offset=10, length=30)
    # the validator computes the trusted hash locally even on the remote path
    assert stream_in.expected_sha256 == hashlib.sha256(provider.materialize(spec)).hexdigest()


def test_prepare_stream_inline_when_corpus_has_no_url(tmp_path):
    provider = _corpus(tmp_path)
    stream_in = _prepare_stream(_RemoteRunner(), provider, StreamSpec("s0", 0, 50))
    assert stream_in.source is None
    assert stream_in.data == b"x" * 50
    assert stream_in.expected_sha256 == hashlib.sha256(b"x" * 50).hexdigest()


def test_prepare_stream_inline_for_local_runner_even_with_url(tmp_path):
    provider = _corpus(tmp_path, base_url="https://c/corpus.bin")
    stream_in = _prepare_stream(_LocalRunner(), provider, StreamSpec("s0", 0, 50))
    assert stream_in.source is None
    assert stream_in.data == b"x" * 50
