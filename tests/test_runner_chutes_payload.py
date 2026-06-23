"""Chutes dispatch: inline-bytes vs URL/range payload selection (issue #3).

Production runs let the deployed chute range-fetch each stream from the published corpus
URL instead of the validator inlining (and re-uploading) the 256 MiB sample. These tests
pin the request-payload selection and the evaluator/corpus wiring that drives it.
"""

import base64

import pytest

from eval.corpus import StaticLocalProvider
from eval.evaluator import _prepare_stream
from eval.runner import ArtifactRef, LocalSubprocessRunner, ResourceCaps, RunnerError, StreamInput
from eval.runner_chutes import ChutesRunner
from eval.streams import RangeSource, StreamSpec

ARTIFACT = ArtifactRef(repo="org/codec", rev="abc123", sha256="deadbeef")
CAPS = ResourceCaps(wall_clock_secs=42.0)


# --- ChutesRunner._build_payload selects the stream shape -------------------------

def test_payload_inline_when_stream_has_bytes():
    payload = ChutesRunner._build_payload(ARTIFACT, StreamInput("s0", data=b"hello world"), CAPS)
    stream = payload["stream"]
    assert stream["stream_id"] == "s0"
    assert stream["inline_b64"] == base64.b64encode(b"hello world").decode("ascii")
    assert "url" not in stream
    assert payload["artifact"] == {"repo": "org/codec", "rev": "abc123", "sha256": "deadbeef"}
    assert payload["wall_clock_secs"] == 42.0


def test_payload_url_when_stream_has_source():
    stream_in = StreamInput("s1", source=RangeSource(url="https://c/corpus.bin", offset=1024, length=4096))
    stream = ChutesRunner._build_payload(ARTIFACT, stream_in, CAPS)["stream"]
    assert stream == {"stream_id": "s1", "url": "https://c/corpus.bin", "offset": 1024, "length": 4096}
    assert "inline_b64" not in stream


def test_payload_requires_data_or_source():
    with pytest.raises(RunnerError):
        ChutesRunner._build_payload(ARTIFACT, StreamInput("s2"), CAPS)


def test_runner_remote_preference_flags():
    assert ChutesRunner.prefers_remote_source is True
    assert LocalSubprocessRunner.prefers_remote_source is False


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


# --- evaluator wires runner capability + corpus capability ------------------------

class _RemoteRunner:
    prefers_remote_source = True


class _LocalRunner:
    prefers_remote_source = False


def test_prepare_stream_remote_when_runner_and_corpus_support_it(tmp_path):
    provider = _corpus(tmp_path, base_url="https://c/corpus.bin")
    stream_in = _prepare_stream(_RemoteRunner(), provider, StreamSpec("s0", 10, 30))
    assert stream_in.data is None
    assert stream_in.source == RangeSource(url="https://c/corpus.bin", offset=10, length=30)
    assert stream_in.raw_len == 30  # length known without materializing


def test_prepare_stream_inline_when_corpus_has_no_url(tmp_path):
    stream_in = _prepare_stream(_RemoteRunner(), _corpus(tmp_path), StreamSpec("s0", 0, 50))
    assert stream_in.source is None
    assert stream_in.data == b"x" * 50
    assert stream_in.raw_len == 50


def test_prepare_stream_inline_for_local_runner_even_with_url(tmp_path):
    provider = _corpus(tmp_path, base_url="https://c/corpus.bin")
    stream_in = _prepare_stream(_LocalRunner(), provider, StreamSpec("s0", 0, 50))
    assert stream_in.source is None
    assert stream_in.data == b"x" * 50
