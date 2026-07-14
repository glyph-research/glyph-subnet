import json
import shutil
from pathlib import Path

import pytest

from eval.runner import ArtifactRef, ResourceCaps, RunnerError, StreamInput
from eval.runner_docker import DockerRunner

REFERENCE_CODEC = Path(__file__).resolve().parents[1] / "reference_codec"
# Built via `docker build -f docker/glyph-runner-default.Dockerfile -t glyph-runner-default:latest .`
IMAGE = "glyph-runner-default:latest"

pytestmark = pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")


def _has_image(name: str) -> bool:
    import subprocess

    return subprocess.run(["docker", "image", "inspect", name], capture_output=True).returncode == 0


requires_image = pytest.mark.skipif(
    not _has_image(IMAGE), reason=f"{IMAGE} not built; see docker/glyph-runner-default.Dockerfile"
)


@requires_image
def test_reference_codec_round_trips():
    runner = DockerRunner(image=IMAGE)
    artifact = ArtifactRef(repo="glyph/ref", rev="local", local_path=str(REFERENCE_CODEC))
    data = b"the quick brown fox jumps over the lazy dog\n" * 4000
    result = runner.run_stream(artifact, StreamInput("s0", data), caps=ResourceCaps())
    assert result.roundtrip_ok is True
    assert result.raw_bytes == len(data)
    assert 0 < result.compressed_bytes < result.raw_bytes
    assert result.blob_hash


@requires_image
def test_compress_then_decompress_reconstructs_via_blob_only():
    runner = DockerRunner(image=IMAGE)
    artifact = ArtifactRef(repo="glyph/ref", rev="local", local_path=str(REFERENCE_CODEC))
    raw = b"the quick brown fox " * 2000
    comp = runner.compress(artifact, raw, caps=ResourceCaps())
    assert comp.raw_bytes == len(raw)
    assert 0 < comp.compressed_bytes < len(raw)
    decomp = runner.decompress(artifact, comp.blob, caps=ResourceCaps())
    assert decomp.output_hash == comp.source_hash


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
    (directory / "compress.py").write_text(
        "import argparse\n"
        "p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output')\n"
        "a=p.parse_args()\n"
        "open(a.output,'wb').write(open(a.input,'rb').read())\n"
    )
    (directory / "decompress.py").write_text(decompress_body)


def test_broken_codec_fails_roundtrip_gate(tmp_path):
    _write_codec(
        tmp_path,
        "import argparse\n"
        "p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output')\n"
        "a=p.parse_args()\n"
        "open(a.output,'wb').write(b'corrupted output')\n",
    )
    runner = DockerRunner()
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    result = runner.run_stream(artifact, StreamInput("s0", b"hello world" * 100), caps=ResourceCaps())
    assert result.roundtrip_ok is False


def test_crashing_codec_raises_runner_error(tmp_path):
    _write_codec(tmp_path, "import sys\nsys.exit(3)\n")
    runner = DockerRunner()
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    with pytest.raises(RunnerError):
        runner.run_stream(artifact, StreamInput("s0", b"data" * 100), caps=ResourceCaps())


# --- split compress/decompress: separate containers defeat the stash cheat (#14) --


def test_filesystem_stash_across_compress_decompress_is_defeated(tmp_path):
    # A codec that stashes the raw input during compress and reads it back during decompress
    # would fake a ~zero ratio if both phases shared a container. Each phase is a brand-new
    # `docker run --rm` container with its own filesystem, so the stash never crosses over.
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
    runner = DockerRunner()
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    raw = b"secret payload " * 500
    result = runner.run_stream(artifact, StreamInput("s0", raw), caps=ResourceCaps())
    assert result.compressed_bytes == 1  # the cheat produced a 1-byte blob...
    assert result.roundtrip_ok is False  # ...but the isolated decompress can't recover the raw


def test_network_is_dropped_during_untrusted_execution(tmp_path):
    # ResourceCaps.network defaults False -> --network none. A codec trying to reach the
    # network during execution must fail, not silently succeed.
    _write_codec(
        tmp_path,
        "import argparse, socket, sys\n"
        "p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output')\n"
        "a=p.parse_args()\n"
        "s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "s.settimeout(3)\n"
        "try:\n"
        "    s.connect(('8.8.8.8', 53))\n"
        "    sys.exit(9)\n"  # reached the network -- isolation failed
        "except OSError:\n"
        "    open(a.output,'wb').write(open(a.input,'rb').read())\n",
    )
    runner = DockerRunner()
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    result = runner.run_stream(artifact, StreamInput("s0", b"data" * 100), caps=ResourceCaps())
    assert result.roundtrip_ok is True  # only reachable if the network connect raised OSError


def test_docker_binary_missing_raises_runner_error(monkeypatch):
    monkeypatch.setattr("eval.runner_docker.shutil.which", lambda name: None)
    with pytest.raises(RunnerError, match="docker binary not found"):
        DockerRunner()


# --- scratch disk cap: a codec can't fill the validator host's disk (issue #54) -------------
# scratch_cap_bytes is overridden tiny here so the test doesn't need to write real gigabytes;
# production uses core.constants.SCRATCH_CAP_BYTES (117 GiB) by default.


def _write_disk_hog_codec(directory: Path, num_files: int = 20, chunk_bytes: int = 8 * 1024, sleep_secs: float = 0.3):
    """A codec that writes ``num_files`` separate ``chunk_bytes``-sized files, spaced out so
    the background disk watchdog (2s poll interval) gets a real chance to catch a cumulative
    overage mid-write rather than the whole loop completing before the first poll. Each
    individual file stays well under any reasonable per-file (``--ulimit fsize``) cap -- this
    exercises the watchdog's total-directory-size check specifically, the many-small-files
    case a per-file ulimit alone wouldn't catch.
    """

    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "disk-hog-codec",
                "entrypoints": {
                    "compress": ["python3", "compress.py", "--input", "{input}", "--output", "{output}"],
                    "decompress": ["python3", "decompress.py", "--input", "{input}", "--output", "{output}"],
                },
                "license": "MIT",
            }
        )
    )
    (directory / "compress.py").write_text(
        "import argparse, time\n"
        "p = argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output')\n"
        "a = p.parse_args()\n"
        f"for i in range({num_files}):\n"
        f"    open(f'/scratch/hog_{{i}}.bin', 'wb').write(b'x' * {chunk_bytes})\n"
        f"    time.sleep({sleep_secs})\n"
        "open(a.output, 'wb').write(open(a.input, 'rb').read())\n"
    )
    (directory / "decompress.py").write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output')\n"
        "a = p.parse_args()\n"
        "open(a.output, 'wb').write(open(a.input, 'rb').read())\n"
    )


def test_codec_exceeding_disk_cap_is_killed(tmp_path):
    # 20 files x 8 KiB = 160 KiB total, each individually well under the 64 KiB cap (so the
    # per-file --ulimit fsize doesn't fire first) -- only the watchdog's cumulative check can
    # catch this, after roughly the 8th file.
    _write_disk_hog_codec(tmp_path, num_files=20, chunk_bytes=8 * 1024, sleep_secs=0.3)
    runner = DockerRunner(scratch_cap_bytes=64 * 1024)
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    with pytest.raises(RunnerError, match="disk cap"):
        runner.compress(artifact, b"data" * 100, caps=ResourceCaps())


def test_single_file_over_cap_fails_via_ulimit(tmp_path):
    # A single write past --ulimit fsize fails instantly at the kernel level (EFBIG), before
    # the watchdog's first poll -- the complementary per-file enforcement.
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entrypoints": {
                    "compress": ["python3", "compress.py", "--input", "{input}", "--output", "{output}"],
                    "decompress": ["python3", "decompress.py", "--input", "{input}", "--output", "{output}"],
                },
                "license": "MIT",
            }
        )
    )
    (tmp_path / "compress.py").write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output')\n"
        "a = p.parse_args()\n"
        "open('/scratch/hog.bin', 'wb').write(b'x' * (1024 * 1024))\n"  # 1 MiB in one file
        "open(a.output, 'wb').write(open(a.input, 'rb').read())\n"
    )
    (tmp_path / "decompress.py").write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output')\n"
        "a = p.parse_args()\n"
        "open(a.output, 'wb').write(open(a.input, 'rb').read())\n"
    )
    runner = DockerRunner(scratch_cap_bytes=64 * 1024)  # 64 KiB -- the 1 MiB write exceeds it
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    with pytest.raises(RunnerError, match="entrypoint exited"):
        runner.compress(artifact, b"data" * 100, caps=ResourceCaps())


def test_codec_under_disk_cap_still_roundtrips(tmp_path):
    _write_disk_hog_codec(tmp_path, num_files=2, chunk_bytes=8 * 1024, sleep_secs=0.1)  # 16 KiB total
    runner = DockerRunner(scratch_cap_bytes=16 * 1024 * 1024)  # 16 MiB -- comfortably clear
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    raw = b"under the cap" * 50
    comp = runner.compress(artifact, raw, caps=ResourceCaps())
    decomp = runner.decompress(artifact, comp.blob, caps=ResourceCaps())
    assert decomp.output_hash == comp.source_hash


# --- GPU model pin: every validator's --docker-gpu must be the same card --------------------
# core.constants.DOCKER_REFERENCE_GPU = "RTX 4090"; DockerRunner(gpu=True) must fail closed on
# anything else. Mocked here (no real GPU needed); the live RTX 4090 box confirms the real
# nvidia-smi path separately.


def _which_stub(has_nvidia_smi: bool):
    def _which(name):
        if name == "nvidia-smi":
            return "/usr/bin/nvidia-smi" if has_nvidia_smi else None
        return f"/usr/bin/{name}"  # docker (and anything else DockerRunner checks) is "present"

    return _which


def _run_stub(stdout: str, returncode: int = 0):
    def _run(cmd, **kwargs):
        import subprocess as _sp

        return _sp.CompletedProcess(cmd, returncode, stdout=stdout, stderr="" if returncode == 0 else "boom")

    return _run


def test_gpu_flag_accepts_matching_reference_gpu(monkeypatch):
    monkeypatch.setattr("eval.runner_docker.shutil.which", _which_stub(True))
    monkeypatch.setattr("eval.runner_docker.subprocess.run", _run_stub("NVIDIA GeForce RTX 4090\n"))
    runner = DockerRunner(gpu=True)
    assert runner.gpu is True


def test_gpu_flag_rejects_mismatched_gpu(monkeypatch):
    monkeypatch.setattr("eval.runner_docker.shutil.which", _which_stub(True))
    monkeypatch.setattr("eval.runner_docker.subprocess.run", _run_stub("NVIDIA A100-SXM4-80GB\n"))
    with pytest.raises(RunnerError, match="requires GPU model containing 'RTX 4090'"):
        DockerRunner(gpu=True)


def test_gpu_flag_rejects_when_nvidia_smi_missing(monkeypatch):
    monkeypatch.setattr("eval.runner_docker.shutil.which", _which_stub(False))
    with pytest.raises(RunnerError, match="nvidia-smi"):
        DockerRunner(gpu=True)


def test_gpu_flag_rejects_when_nvidia_smi_reports_no_gpus(monkeypatch):
    monkeypatch.setattr("eval.runner_docker.shutil.which", _which_stub(True))
    monkeypatch.setattr("eval.runner_docker.subprocess.run", _run_stub(""))
    with pytest.raises(RunnerError, match="no GPUs"):
        DockerRunner(gpu=True)


def test_gpu_device_filter_is_passed_to_nvidia_smi(monkeypatch):
    monkeypatch.setattr("eval.runner_docker.shutil.which", _which_stub(True))
    seen_cmds = []

    def _run(cmd, **kwargs):
        import subprocess as _sp

        seen_cmds.append(cmd)
        return _sp.CompletedProcess(cmd, 0, stdout="NVIDIA GeForce RTX 4090\n", stderr="")

    monkeypatch.setattr("eval.runner_docker.subprocess.run", _run)
    DockerRunner(gpu=True, gpu_device="0")
    assert "-i" in seen_cmds[0] and "0" in seen_cmds[0]


def test_no_gpu_flag_skips_the_check_entirely(monkeypatch):
    # gpu=False (the default) must not touch nvidia-smi at all -- CPU-only codecs shouldn't
    # need a GPU present, matching, and available.
    monkeypatch.setattr("eval.runner_docker.shutil.which", _which_stub(False))
    DockerRunner(gpu=False)  # would raise if the GPU check ran despite gpu=False


# --- GPU-memory preflight (issue #105): fail closed before launching a codec's container if
# its declared resources["vram_gb"] (+ headroom) doesn't fit in currently-free VRAM, so this
# surfaces as a distinct host-capacity error rather than an opaque "entrypoint exited ...".


def _manifest_with_vram(vram_gb):
    from core.artifact import Entrypoints, Manifest

    resources = {} if vram_gb is None else {"vram_gb": vram_gb}
    return Manifest(
        entrypoints=Entrypoints(
            compress=["true", "{input}", "{output}"], decompress=["true", "{input}", "{output}"]
        ),
        resources=resources,
    )


def test_free_gpu_memory_gb_reports_the_tightest_gpu(monkeypatch):
    from eval.runner_docker import _free_gpu_memory_gb

    monkeypatch.setattr("eval.runner_docker.shutil.which", _which_stub(True))
    monkeypatch.setattr("eval.runner_docker.subprocess.run", _run_stub("2048\n40960\n"))
    assert _free_gpu_memory_gb(None) == pytest.approx(2.0)


def test_free_gpu_memory_gb_none_when_nvidia_smi_missing(monkeypatch):
    from eval.runner_docker import _free_gpu_memory_gb

    monkeypatch.setattr("eval.runner_docker.shutil.which", _which_stub(False))
    assert _free_gpu_memory_gb(None) is None


def test_free_gpu_memory_gb_none_on_query_failure(monkeypatch):
    from eval.runner_docker import _free_gpu_memory_gb

    monkeypatch.setattr("eval.runner_docker.shutil.which", _which_stub(True))
    monkeypatch.setattr("eval.runner_docker.subprocess.run", _run_stub("", returncode=1))
    assert _free_gpu_memory_gb(None) is None


def test_check_gpu_memory_skips_when_vram_gb_not_declared(monkeypatch):
    from eval.runner_docker import _check_gpu_memory

    def _boom(gpu_device):
        raise AssertionError("must not query free memory when nothing is declared to check")

    monkeypatch.setattr("eval.runner_docker._free_gpu_memory_gb", _boom)
    _check_gpu_memory(_manifest_with_vram(None), None)  # no raise


def test_check_gpu_memory_skips_when_free_memory_unknown(monkeypatch):
    from eval.runner_docker import _check_gpu_memory

    monkeypatch.setattr("eval.runner_docker._free_gpu_memory_gb", lambda gpu_device: None)
    _check_gpu_memory(_manifest_with_vram(20), None)  # can't verify -> don't block


def test_check_gpu_memory_raises_when_insufficient(monkeypatch):
    from eval.runner import InsufficientGpuMemoryError
    from eval.runner_docker import _check_gpu_memory

    monkeypatch.setattr("eval.runner_docker._free_gpu_memory_gb", lambda gpu_device: 5.0)
    monkeypatch.setattr("eval.runner_docker._total_gpu_memory_gb", lambda gpu_device: None)
    with pytest.raises(InsufficientGpuMemoryError, match="need ~22.0 GiB"):
        _check_gpu_memory(_manifest_with_vram(20), None)


def test_check_gpu_memory_passes_with_enough_headroom(monkeypatch):
    from eval.runner_docker import _check_gpu_memory

    monkeypatch.setattr("eval.runner_docker._free_gpu_memory_gb", lambda gpu_device: 30.0)
    monkeypatch.setattr("eval.runner_docker._total_gpu_memory_gb", lambda gpu_device: None)
    _check_gpu_memory(_manifest_with_vram(20), None)  # no raise


# --- requirement capped at the GPU's actual total capacity (issue #123) ------------------
#
# declared + headroom was never capped against what the card physically has: a codec
# honestly declaring vram_gb at/near the reference GPU's capacity (24 GiB declared -> 26 GiB
# required vs ~23.99 GiB total) could NEVER pass -- permanent silent exclusion on that
# hardware, not the transient deferral the preflight is for. Numbers below mirror the live
# observation on the reference RTX 4090: total 23.99 GiB, idle free 23.30 GiB.


def test_check_gpu_memory_caps_requirement_at_total_minus_reserve(monkeypatch):
    from eval.runner_docker import _check_gpu_memory

    # Idle reference card: declared 24 + 2 headroom = 26 GiB raw requirement, but the cap
    # (23.99 - 0.75 = 23.24) fits in the 23.30 GiB an idle card actually has free.
    monkeypatch.setattr("eval.runner_docker._free_gpu_memory_gb", lambda gpu_device: 23.30)
    monkeypatch.setattr("eval.runner_docker._total_gpu_memory_gb", lambda gpu_device: 23.99)
    _check_gpu_memory(_manifest_with_vram(24), None)  # no raise


def test_check_gpu_memory_capped_requirement_still_blocks_a_busy_gpu(monkeypatch):
    from eval.runner import InsufficientGpuMemoryError
    from eval.runner_docker import _check_gpu_memory

    # The cap makes a max declaration satisfiable on an IDLE card -- it must not neuter the
    # check when the GPU genuinely is occupied (free well below total - reserve).
    monkeypatch.setattr("eval.runner_docker._free_gpu_memory_gb", lambda gpu_device: 5.0)
    monkeypatch.setattr("eval.runner_docker._total_gpu_memory_gb", lambda gpu_device: 23.99)
    with pytest.raises(InsufficientGpuMemoryError, match="capped at GPU total"):
        _check_gpu_memory(_manifest_with_vram(24), None)


def test_check_gpu_memory_modest_declaration_unaffected_by_the_cap(monkeypatch):
    from eval.runner import InsufficientGpuMemoryError
    from eval.runner_docker import _check_gpu_memory

    # Well under the cap: min(20 + 2, 23.24) = 22 -- identical requirement to before #123.
    monkeypatch.setattr("eval.runner_docker._total_gpu_memory_gb", lambda gpu_device: 23.99)
    monkeypatch.setattr("eval.runner_docker._free_gpu_memory_gb", lambda gpu_device: 23.30)
    _check_gpu_memory(_manifest_with_vram(20), None)  # no raise
    monkeypatch.setattr("eval.runner_docker._free_gpu_memory_gb", lambda gpu_device: 5.0)
    with pytest.raises(InsufficientGpuMemoryError, match="need ~22.0 GiB"):
        _check_gpu_memory(_manifest_with_vram(20), None)


def test_check_gpu_memory_unknown_total_leaves_requirement_uncapped(monkeypatch):
    from eval.runner import InsufficientGpuMemoryError
    from eval.runner_docker import _check_gpu_memory

    # Same fail-open posture as the free-memory query: an unknown total just skips the cap.
    monkeypatch.setattr("eval.runner_docker._total_gpu_memory_gb", lambda gpu_device: None)
    monkeypatch.setattr("eval.runner_docker._free_gpu_memory_gb", lambda gpu_device: 23.30)
    with pytest.raises(InsufficientGpuMemoryError, match="need ~26.0 GiB"):
        _check_gpu_memory(_manifest_with_vram(24), None)


def test_total_gpu_memory_gb_parses_and_fails_open(monkeypatch):
    from eval.runner_docker import _total_gpu_memory_gb

    monkeypatch.setattr("eval.runner_docker.shutil.which", _which_stub(True))
    monkeypatch.setattr("eval.runner_docker.subprocess.run", _run_stub("24564\n"))
    assert _total_gpu_memory_gb(None) == pytest.approx(24564 / 1024.0)

    monkeypatch.setattr("eval.runner_docker.shutil.which", _which_stub(False))
    assert _total_gpu_memory_gb(None) is None


def test_execute_checks_gpu_memory_before_launching_any_container(monkeypatch, tmp_path):
    from eval.runner import ArtifactRef, InsufficientGpuMemoryError

    monkeypatch.setattr("eval.runner_docker.shutil.which", _which_stub(True))
    seen_cmds = []

    def _run(cmd, **kwargs):
        import subprocess as _sp

        seen_cmds.append(cmd)
        return _sp.CompletedProcess(cmd, 0, stdout="NVIDIA GeForce RTX 4090\n", stderr="")

    monkeypatch.setattr("eval.runner_docker.subprocess.run", _run)
    runner = DockerRunner(gpu=True)  # one subprocess.run call already happened (GPU model check)
    monkeypatch.setattr("eval.runner_docker._free_gpu_memory_gb", lambda gpu_device: 1.0)

    artifact_dir = tmp_path / "codec"
    artifact_dir.mkdir()
    _write_manifest_and_scripts(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["resources"] = {"vram_gb": 20}
    manifest_path.write_text(json.dumps(manifest))
    artifact = ArtifactRef(repo="t/codec", rev="local", local_path=str(artifact_dir))

    with pytest.raises(InsufficientGpuMemoryError):
        runner.compress(artifact, b"hello", caps=ResourceCaps())

    # No "docker run"/"docker exec" call reached -- only the constructor's GPU-model query.
    assert all("run" not in cmd and "exec" not in cmd for cmd in seen_cmds)


# --- issue #66: a snapshot_download of a real HF repo returns symlinks-into-blobs/ by default;
# DockerRunner bind-mounts ONLY the snapshot dir, so those symlinks dangle inside the container.
# local_snapshot_dir()-based downloads (the fix, wired in reign_worker.service.artifact_ref /
# validation.precheck.precheck_codec) materialize real files instead -- these tests prove BOTH
# the failure mode and the fix using DockerRunner directly (no mocking of DockerRunner itself).


def _write_manifest_and_scripts(directory: Path) -> None:
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "symlink-regression",
                "entrypoints": {
                    "compress": ["python3", "compress.py", "--input", "{input}", "--output", "{output}"],
                    "decompress": ["python3", "decompress.py", "--input", "{input}", "--output", "{output}"],
                },
                "license": "MIT",
            }
        )
    )
    (directory / "compress.py").write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output')\n"
        "a = p.parse_args()\n"
        "open(a.output, 'wb').write(open(a.input, 'rb').read())\n"
    )
    (directory / "decompress.py").write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output')\n"
        "a = p.parse_args()\n"
        "open(a.output, 'wb').write(open(a.input, 'rb').read())\n"
    )


def test_dangling_symlink_snapshot_layout_fails_inside_container(tmp_path):
    # Mimics huggingface_hub's default cache-based download layout: the "snapshot" dir
    # contains only symlinks into a SEPARATE "blobs" dir that lives outside it -- exactly
    # what DockerRunner's bind-mount (only the snapshot dir) can't see.
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    _write_manifest_and_scripts(blobs)
    for name in ("manifest.json", "compress.py", "decompress.py"):
        (snapshot / name).symlink_to(blobs / name)

    runner = DockerRunner()
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(snapshot))
    with pytest.raises(RunnerError, match="No such file or directory|can't open file"):
        runner.compress(artifact, b"data" * 100, caps=ResourceCaps())


def test_materialized_real_files_snapshot_layout_succeeds(tmp_path):
    # Same content, but as real files (what local_snapshot_dir()-based downloads produce) --
    # this is the fix: DockerRunner works correctly once there's nothing to dangle.
    _write_manifest_and_scripts(tmp_path)
    runner = DockerRunner()
    artifact = ArtifactRef(repo="t/c", rev="local", local_path=str(tmp_path))
    raw = b"data" * 100
    comp = runner.compress(artifact, raw, caps=ResourceCaps())
    assert comp.compressed_bytes == len(raw)  # store-only passthrough, but it actually ran
