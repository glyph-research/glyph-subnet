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


def test_runner_scrubs_secret_environment(monkeypatch, tmp_path):
    _write_codec(
        tmp_path,
        "import argparse\n"
        "p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output')\n"
        "a=p.parse_args()\n"
        "open(a.output,'wb').write(open(a.input,'rb').read())\n",
    )
    (tmp_path / "compress.py").write_text(
        "import argparse, os, sys\n"
        "if os.environ.get('CHUTES_API_KEY') or os.environ.get('HF_TOKEN'):\n"
        "    sys.exit(7)\n"
        "p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output')\n"
        "a=p.parse_args()\n"
        "open(a.output,'wb').write(open(a.input,'rb').read())\n"
    )
    monkeypatch.setenv("CHUTES_API_KEY", "cpk_secret")
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    runner = LocalSubprocessRunner()
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    result = runner.run_stream(artifact, StreamInput("s0", b"data" * 100), caps=ResourceCaps())
    assert result.roundtrip_ok is True


def test_strict_runner_fails_closed_without_network_isolation(monkeypatch, tmp_path):
    _write_codec(
        tmp_path,
        "import argparse\n"
        "p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output')\n"
        "a=p.parse_args()\n"
        "open(a.output,'wb').write(open(a.input,'rb').read())\n",
    )
    monkeypatch.setattr("eval.runner.shutil.which", lambda name: None)
    runner = LocalSubprocessRunner(strict_sandbox=True, require_network_isolation=True)
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    with pytest.raises(RunnerError, match="network isolation unavailable"):
        runner.run_stream(artifact, StreamInput("s0", b"data" * 100), caps=ResourceCaps())


# --- split compress/decompress: separate sandboxes defeat the stash cheat (#14) --

def test_compress_then_decompress_reconstructs_via_blob_only(tmp_path):
    runner = LocalSubprocessRunner()
    artifact = ArtifactRef(repo="glyph/ref", rev="local", local_path=str(REFERENCE_CODEC))
    raw = b"the quick brown fox " * 2000
    comp = runner.compress(artifact, raw, caps=ResourceCaps())
    assert comp.raw_bytes == len(raw)
    assert 0 < comp.compressed_bytes < len(raw)
    assert comp.source_hash and comp.blob_hash
    # the blob alone (no shared state with the compress sandbox) reconstructs the original
    decomp = runner.decompress(artifact, comp.blob, caps=ResourceCaps())
    assert decomp.output_hash == comp.source_hash
    assert decomp.raw_bytes == len(raw)


def test_filesystem_stash_across_compress_decompress_is_defeated(tmp_path):
    # A codec that stashes the raw input to $HOME during compress and reads it back during
    # decompress would fake a ~zero ratio if both phases shared a sandbox. With the split they
    # get separate $HOME dirs, so the stash is gone and the cheat fails the bit-exact gate.
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "stash-cheat",
                "entrypoints": {
                    "compress": ["python3", "compress.py", "--input", "{input}", "--output", "{output}"],
                    "decompress": ["python3", "decompress.py", "--input", "{input}", "--output", "{output}"],
                },
                "license": "MIT",
            }
        )
    )
    (tmp_path / "compress.py").write_text(
        "import argparse, os\n"
        "p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output')\n"
        "a=p.parse_args()\n"
        "raw=open(a.input,'rb').read()\n"
        "open(os.path.join(os.environ['HOME'],'glyph_stash.bin'),'wb').write(raw)\n"
        "open(a.output,'wb').write(b'X')\n"  # 1-byte blob: ratio ~0 if the stash were readable
    )
    (tmp_path / "decompress.py").write_text(
        "import argparse, os\n"
        "p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output')\n"
        "a=p.parse_args()\n"
        "stash=os.path.join(os.environ['HOME'],'glyph_stash.bin')\n"
        "data=open(stash,'rb').read() if os.path.exists(stash) else open(a.input,'rb').read()\n"
        "open(a.output,'wb').write(data)\n"
    )
    runner = LocalSubprocessRunner()
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    raw = b"secret payload " * 500
    result = runner.run_stream(artifact, StreamInput("s0", raw), caps=ResourceCaps())
    assert result.compressed_bytes == 1  # the cheat produced a 1-byte blob...
    assert result.roundtrip_ok is False  # ...but the isolated decompress can't recover the raw
