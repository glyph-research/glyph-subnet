import json
from pathlib import Path

import pytest

from core.artifact import hash_artifact, load_manifest
from eval.runner import ArtifactRef, LocalSubprocessRunner, ResourceCaps, RunnerError, StreamInput

REFERENCE_CODEC = Path(__file__).resolve().parents[1] / "reference_codec"


def test_reference_codec_round_trips():
    runner = LocalSubprocessRunner()
    artifact = ArtifactRef(repo="glyph/ref", rev="local", local_path=str(REFERENCE_CODEC))
    data = (b"the quick brown fox jumps over the lazy dog\n" * 4000)
    result = runner.run_stream(artifact, StreamInput("s0", data), caps=ResourceCaps())
    assert result.roundtrip_ok is True
    assert result.raw_bytes == len(data)
    assert 0 < result.compressed_bytes < result.raw_bytes  # actually compresses
    assert result.blob_hash


def test_manifest_and_hash_load():
    manifest = load_manifest(REFERENCE_CODEC)
    assert manifest.entrypoints.compress[0] == "python3"
    assert manifest.placeholder_issues() == []
    digest, total = hash_artifact(REFERENCE_CODEC)
    assert len(digest) == 64 and total > 0


def _write_codec(directory: Path, decompress_body: str):
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "test-codec",
                "entrypoints": {
                    "compress": ["python3", "compress.py", "--input", "{input}", "--output", "{output}"],
                    "decompress": ["python3", "decompress.py", "--input", "{input}", "--output", "{output}"],
                },
                "license": "MIT",
            }
        )
    )
    # store-only compress: copy input to output verbatim
    (directory / "compress.py").write_text(
        "import argparse\n"
        "p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output')\n"
        "a=p.parse_args()\n"
        "open(a.output,'wb').write(open(a.input,'rb').read())\n"
    )
    (directory / "decompress.py").write_text(decompress_body)


def test_broken_codec_fails_roundtrip_gate(tmp_path):
    # decompress writes wrong bytes -> not bit-exact -> roundtrip_ok must be False
    _write_codec(
        tmp_path,
        "import argparse\n"
        "p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output')\n"
        "a=p.parse_args()\n"
        "open(a.output,'wb').write(b'corrupted output')\n",
    )
    runner = LocalSubprocessRunner()
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    result = runner.run_stream(artifact, StreamInput("s0", b"hello world" * 100), caps=ResourceCaps())
    assert result.roundtrip_ok is False


def test_crashing_codec_raises_runner_error(tmp_path):
    _write_codec(
        tmp_path,
        "import sys\nsys.exit(3)\n",
    )
    runner = LocalSubprocessRunner()
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    with pytest.raises(RunnerError):
        runner.run_stream(artifact, StreamInput("s0", b"data" * 100), caps=ResourceCaps())
