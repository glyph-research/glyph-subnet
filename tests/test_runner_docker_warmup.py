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
    # Non-vacuous (review feedback on PR #53): first PROVE real egress exists during warmup
    # -- a warmup command that itself does the same connect and raises (failing the whole
    # test loudly) if it doesn't, so a host with no outbound internet can't make this test
    # pass for the wrong reason -- THEN prove the scored entrypoint's own connect attempt
    # fails once sealed.
    warmup_probe = (
        "import socket\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "s.settimeout(3)\n"
        "s.connect(('8.8.8.8', 53))\n"  # raises (failing warmup, and the test) if no real egress
    )
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
        _manifest(image=_DIGEST, warmup={"command": ["python3", "-c", warmup_probe], "timeout_secs": 30}),
        compress_body=probe_body,
    )
    runner = DockerRunner()
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    # If warmup's own connect had failed, _await_warmup would raise here -- reaching a
    # successful compress() is itself proof warmup genuinely had working network.
    comp = runner.compress(artifact, b"data" * 100, caps=ResourceCaps())
    assert comp.compressed_bytes > 0  # only reachable if the POST-seal connect raised OSError


@requires_digest
def test_established_connection_stops_delivering_after_seal(tmp_path):
    """A connection opened during warmup and held open across the seal.

    Ground-truth check, not a send()-return-value check: manual investigation (see PR #53
    review discussion) found ``send()`` on an already-established socket keeps reporting
    success after ``docker network disconnect`` -- TCP buffers writes locally regardless of
    whether the interface is gone, so a bare "does send() raise" test would be misleading.
    What actually matters is whether bytes are ever genuinely DELIVERED: this test holds one
    real DNS-over-TCP connection to 8.8.8.8:53 open across the seal and requires an actual
    valid response (a full round trip) before warmup is allowed to complete, then keeps
    probing afterward. If any round trip still completes after a timeout was observed, the
    seal has a real residual exfiltration channel and this test fails.
    """

    dns_probe = (
        "import socket, struct, time\n"
        "def query():\n"
        "    header = struct.pack('>HHHHHH', 0x1234, 0x0100, 1, 0, 0, 0)\n"
        "    qname = b''.join(bytes([len(p)]) + p.encode() for p in 'example.com'.split('.')) + b'\\x00'\n"
        "    body = header + qname + struct.pack('>HH', 1, 1)\n"
        "    return struct.pack('>H', len(body)) + body\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "s.settimeout(1.5)\n"
        "log = []\n"
        "s.connect(('8.8.8.8', 53))\n"
        "log.append('connect_ok')\n"
        "s.send(query())\n"
        "resp = s.recv(512)\n"
        "log.append(f'0 recv_ok len={len(resp)}')\n"
        "open('/scratch/dns_probe.log', 'w').write(chr(10).join(log))\n"
        # Only NOW (after a genuine pre-seal round trip already completed) signal the
        # spawner it may let warmup finish -- this removes the race entirely: the seal
        # cannot happen until a real round trip is already proven to have worked.
        "open('/scratch/dns_probe_marker', 'w').write('ok')\n"
        "for i in range(1, 8):\n"
        "    time.sleep(0.5)\n"
        "    try:\n"
        "        s.send(query())\n"
        "    except OSError as e:\n"
        "        log.append(f'{i} send_failed:{e}')\n"
        "        open('/scratch/dns_probe.log', 'w').write(chr(10).join(log))\n"
        "        continue\n"
        "    try:\n"
        "        resp = s.recv(512)\n"
        "        log.append(f'{i} recv_ok len={len(resp)}')\n"
        "    except socket.timeout:\n"
        "        log.append(f'{i} recv_timeout')\n"
        "    except OSError as e:\n"
        "        log.append(f'{i} recv_failed:{e}')\n"
        "    open('/scratch/dns_probe.log', 'w').write(chr(10).join(log))\n"
    )
    spawn_probe = (
        "import subprocess, time, os\n"
        "subprocess.Popen(['python3', '/artifact/dns_probe.py'], start_new_session=True)\n"
        "deadline = time.time() + 15\n"
        "while time.time() < deadline:\n"
        "    if os.path.exists('/scratch/dns_probe_marker'):\n"
        "        break\n"
        "    time.sleep(0.2)\n"
        "else:\n"
        "    raise SystemExit('pre-seal round trip never completed -- no real egress during warmup?')\n"
    )
    compress_body = (
        "import argparse, time\n"
        "p = argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output')\n"
        "a = p.parse_args()\n"
        "time.sleep(6)\n"  # let the still-running background probe attempt several post-seal round trips
        "log = open('/scratch/dns_probe.log', 'rb').read()\n"
        "open(a.output, 'wb').write(log)\n"
    )
    tmp_path.joinpath("manifest.json").write_text(
        json.dumps(_manifest(image=_DIGEST, warmup={"command": ["python3", "/artifact/spawn_probe.py"], "timeout_secs": 30}))
    )
    tmp_path.joinpath("spawn_probe.py").write_text(spawn_probe)
    tmp_path.joinpath("dns_probe.py").write_text(dns_probe)
    tmp_path.joinpath("compress.py").write_text(compress_body)
    tmp_path.joinpath("decompress.py").write_text(_PASSTHROUGH)

    runner = DockerRunner()
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    comp = runner.compress(artifact, b"eval data", caps=ResourceCaps(wall_clock_secs=60))
    log_lines = comp.blob.decode().splitlines()

    assert any("recv_ok" in line for line in log_lines), (
        f"expected at least one genuine pre-seal round trip; log:\n{comp.blob.decode()}"
    )
    first_timeout = next(i for i, line in enumerate(log_lines) if "recv_timeout" in line)
    assert not any("recv_ok" in line for line in log_lines[first_timeout:]), (
        "a round trip completed AFTER a recv_timeout was already observed -- the seal has a "
        f"residual delivery channel; log:\n{comp.blob.decode()}"
    )


# --- scratch disk cap applies to the networked lifecycle too (issue #54) --------------------
# The same scratch mount backs both the networked warmup (downloads/deps) and the sealed
# benchmark, so a single cap must cover both phases. scratch_cap_bytes is overridden tiny here
# so the test doesn't need to write real gigabytes.

_DISK_HOG_LOOP = (
    "for i in range(20):\n"
    "    open(f'/scratch/hog_{i}.bin', 'wb').write(b'x' * (8 * 1024))\n"  # 8 KiB per file
    "    time.sleep(0.3)\n"
)


@requires_digest
def test_networked_lifecycle_disk_cap_kills_codec_during_benchmark(tmp_path):
    # 20 x 8 KiB = 160 KiB total, each file individually well under the 64 KiB cap -- only the
    # watchdog's cumulative check can catch this.
    compress_body = (
        "import argparse, time\n"
        "p = argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output')\n"
        "a = p.parse_args()\n" + _DISK_HOG_LOOP + "open(a.output, 'wb').write(open(a.input, 'rb').read())\n"
    )
    _write_networked_codec(
        tmp_path,
        _manifest(image=_DIGEST, warmup={"command": ["true"], "timeout_secs": 30}),
        compress_body=compress_body,
    )
    runner = DockerRunner(scratch_cap_bytes=64 * 1024)
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    with pytest.raises(RunnerError, match="disk cap"):
        runner.compress(artifact, b"data" * 100, caps=ResourceCaps())


@requires_digest
def test_networked_lifecycle_disk_cap_kills_codec_during_warmup(tmp_path):
    # A runaway warmup (e.g. an unexpectedly huge pip install / weights download) must be
    # capped too -- the same scratch mount backs HOME/TMPDIR/XDG_CACHE_HOME during warmup.
    warmup_hog = "import time\n" + _DISK_HOG_LOOP
    _write_networked_codec(
        tmp_path,
        _manifest(image=_DIGEST, warmup={"command": ["python3", "-c", warmup_hog], "timeout_secs": 30}),
    )
    runner = DockerRunner(scratch_cap_bytes=64 * 1024)
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    with pytest.raises(RunnerError, match="disk cap"):
        runner.compress(artifact, b"data" * 100, caps=ResourceCaps())


@requires_digest
def test_networked_lifecycle_codec_under_disk_cap_still_roundtrips(tmp_path):
    _write_networked_codec(
        tmp_path, _manifest(image=_DIGEST, warmup={"command": ["true"], "timeout_secs": 30})
    )
    runner = DockerRunner(scratch_cap_bytes=16 * 1024 * 1024)  # 16 MiB -- comfortably clear
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    raw = b"under the cap, networked lifecycle" * 50
    comp = runner.compress(artifact, raw, caps=ResourceCaps())
    decomp = runner.decompress(artifact, comp.blob, caps=ResourceCaps())
    assert decomp.output_hash == hashlib.sha256(raw).hexdigest()


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
