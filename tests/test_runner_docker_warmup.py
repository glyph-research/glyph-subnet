"""issue #48: a manifest-declared ``image`` runs a networked warmup -> seal -> benchmark
lifecycle instead of the original single-shot ``--network none``-from-start path. These
tests exercise the real ``DockerRunner`` against real Docker (skipped if unavailable) using
``python:3.12-slim``'s own pulled RepoDigest, so no image build/push is needed."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from eval.runner import ArtifactRef, ResourceCaps, RunnerError
from eval.runner_docker import DockerRunner

pytestmark = pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")


def _local_image_digest(tag: str) -> str | None:
    proc = subprocess.run(
        ["docker", "inspect", "--format={{index .RepoDigests 0}}", tag],
        capture_output=True, text=True,
    )
    digest = proc.stdout.strip()
    return digest if proc.returncode == 0 and digest and digest != "<no value>" else None


_DIGEST = _local_image_digest("python:3.12-slim")
requires_digest = pytest.mark.skipif(
    _DIGEST is None, reason="python:3.12-slim has no local RepoDigest (never pulled from a registry)"
)


def _manifest(image, warmup=None, compress_body="", decompress_body=None):
    return {
        "schema_version": 1,
        "name": "networked-test-codec",
        "entrypoints": {
            "compress": ["python3", "compress.py", "--input", "{input}", "--output", "{output}"],
            "decompress": ["python3", "decompress.py", "--input", "{input}", "--output", "{output}"],
        },
        "license": "MIT",
        "image": image,
        **({"warmup": warmup} if warmup is not None else {}),
    }


_PASSTHROUGH = (
    "import argparse\n"
    "p = argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output')\n"
    "a = p.parse_args()\n"
    "open(a.output, 'wb').write(open(a.input, 'rb').read())\n"
)


def _write_networked_codec(directory: Path, manifest: dict, compress_body: str = _PASSTHROUGH):
    directory.joinpath("manifest.json").write_text(json.dumps(manifest))
    directory.joinpath("compress.py").write_text(compress_body)
    directory.joinpath("decompress.py").write_text(_PASSTHROUGH)


# --- digest required, fail closed -------------------------------------------------------


@requires_digest
def test_mutable_tag_image_fails_closed(tmp_path):
    _write_networked_codec(tmp_path, _manifest(image="python:3.12-slim"))  # tag, not digest
    runner = DockerRunner()
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    with pytest.raises(RunnerError, match="digest"):
        runner.compress(artifact, b"hello world" * 100, caps=ResourceCaps())


@requires_digest
def test_digest_pinned_image_is_accepted(tmp_path):
    _write_networked_codec(
        tmp_path, _manifest(image=_DIGEST, warmup={"command": ["true"], "timeout_secs": 30})
    )
    runner = DockerRunner()
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    raw = b"digest pinned round trip" * 50
    comp = runner.compress(artifact, raw, caps=ResourceCaps())
    decomp = runner.decompress(artifact, comp.blob, caps=ResourceCaps())
    assert decomp.output_hash == hashlib.sha256(raw).hexdigest()


# --- corpus absent during the networked warmup window ------------------------------------


@requires_digest
def test_eval_input_not_present_during_warmup(tmp_path):
    # Warmup command asserts /scratch/in.bin does NOT exist yet -- fails (exit 1) if the
    # implementation ever wrote the eval input before sealing the network.
    warmup_check = ["python3", "-c", "import os,sys; sys.exit(1 if os.path.exists('/scratch/in.bin') else 0)"]
    _write_networked_codec(
        tmp_path, _manifest(image=_DIGEST, warmup={"command": warmup_check, "timeout_secs": 30})
    )
    runner = DockerRunner()
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    # If in.bin were present during warmup, the warmup command above would exit 1 and
    # _await_warmup would raise -- reaching a successful compress() proves it was absent.
    result = runner.compress(artifact, b"secret eval payload" * 20, caps=ResourceCaps())
    assert result.compressed_bytes > 0


# --- network actually severed before the scored benchmark run ----------------------------


@requires_digest
def test_network_severed_before_benchmark_entrypoint_runs(tmp_path):
    # The (scored) compress entrypoint itself tries to reach the network; if the seal didn't
    # actually happen, the connect would succeed and the entrypoint would exit 9 (isolation
    # failure) instead of writing output -- run_stream would then raise, failing the test.
    probe_body = (
        "import argparse, socket, sys\n"
        "p = argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output')\n"
        "a = p.parse_args()\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "s.settimeout(3)\n"
        "try:\n"
        "    s.connect(('8.8.8.8', 53))\n"
        "    sys.exit(9)\n"  # reached the network during the sealed benchmark -- isolation failed
        "except OSError:\n"
        "    open(a.output, 'wb').write(open(a.input, 'rb').read())\n"
    )
    _write_networked_codec(
        tmp_path,
        _manifest(image=_DIGEST, warmup={"command": ["true"], "timeout_secs": 30}),
        compress_body=probe_body,
    )
    runner = DockerRunner()
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    comp = runner.compress(artifact, b"data" * 100, caps=ResourceCaps())
    assert comp.compressed_bytes > 0  # only reachable if the network connect raised OSError


# --- warmup timeout kills the container and fails closed ---------------------------------


@requires_digest
def test_warmup_timeout_kills_container_and_raises(tmp_path):
    _write_networked_codec(
        tmp_path,
        _manifest(image=_DIGEST, warmup={"command": ["sleep", "30"], "timeout_secs": 2}),
    )
    runner = DockerRunner()
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    with pytest.raises(RunnerError, match="warmup"):
        runner.compress(artifact, b"data" * 50, caps=ResourceCaps())
    # No leaked container left running under our generated name prefix.
    ps = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "name=glyph-runner-"], capture_output=True, text=True
    )
    assert ps.stdout.strip() == ""


@requires_digest
def test_ready_file_polling_protocol_times_out_when_never_created(tmp_path):
    # No warmup.command -> _await_warmup polls for the container's own CMD/ENTRYPOINT to
    # create ready_file. Exercising the *success* path needs a custom image whose CMD creates
    # the file (python:3.12-slim's stock CMD never will, and locally-built images have no
    # `name@sha256:...` digest to reference without a registry push -- see PR description for
    # this known coverage gap). This test covers the polling loop's fail-closed side: it must
    # still time out and raise, not hang, when nothing ever creates the file.
    _write_networked_codec(
        tmp_path, _manifest(image=_DIGEST, warmup={"ready_file": "/scratch/.never_created", "timeout_secs": 2})
    )
    runner = DockerRunner()
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    with pytest.raises(RunnerError, match="warmup"):
        runner.compress(artifact, b"data" * 50, caps=ResourceCaps())
