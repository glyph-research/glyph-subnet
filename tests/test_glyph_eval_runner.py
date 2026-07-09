"""issue #57: the Chutes eval runner must drop root privileges for the untrusted codec
(matching the Docker path's --user 65534:65534 posture) and must not let a miner-controlled
artifact tree listing escape the download destination (zip-slip)."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from eval.glyph_eval_runner import (
    ResourceCaps,
    RunnerError,
    _hash_artifact,
    _iter_artifact_files,
    _run_codec,
    _safe_join,
    _setpriv_prefix,
)


def _setpriv_usable() -> bool:
    """True iff this host can both find AND actually use setpriv to change uid/gid (review
    feedback on PR #65): GitHub Actions runners have the binary but lack the capability to
    setuid/setgid (unprivileged container), so `shutil.which` alone gives a false pass there
    and the test hard-fails with "Operation not permitted" instead of skipping."""

    if shutil.which("setpriv") is None:
        return False
    try:
        proc = subprocess.run(
            ["setpriv", "--reuid", "65534", "--regid", "65534", "--clear-groups", "true"],
            capture_output=True,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _unshare_net_usable() -> bool:
    if shutil.which("unshare") is None:
        return False
    try:
        return subprocess.run(["unshare", "--net", "true"], capture_output=True).returncode == 0
    except Exception:
        return False


# --- privilege drop is requested by default, no env vars needed -----------------------------


def test_setpriv_prefix_defaults_to_nonroot_uid_without_env_vars(monkeypatch):
    monkeypatch.delenv("GLYPH_CODEC_PRIVDROP_UID", raising=False)
    monkeypatch.delenv("GLYPH_CODEC_PRIVDROP_GID", raising=False)
    monkeypatch.delenv("GLYPH_CODEC_ALLOW_UNPRIVILEGED", raising=False)
    monkeypatch.setattr("eval.glyph_eval_runner.shutil.which", lambda name: f"/usr/bin/{name}")
    prefix = _setpriv_prefix()
    assert "--reuid" in prefix and prefix[prefix.index("--reuid") + 1] == "65534"
    assert "--regid" in prefix and prefix[prefix.index("--regid") + 1] == "65534"
    assert "--no-new-privs" in prefix
    assert "--bounding-set=-all" in prefix


def test_setpriv_prefix_respects_custom_uid_gid_override(monkeypatch):
    monkeypatch.setenv("GLYPH_CODEC_PRIVDROP_UID", "1234")
    monkeypatch.setenv("GLYPH_CODEC_PRIVDROP_GID", "5678")
    monkeypatch.setattr("eval.glyph_eval_runner.shutil.which", lambda name: f"/usr/bin/{name}")
    prefix = _setpriv_prefix()
    assert prefix[prefix.index("--reuid") + 1] == "1234"
    assert prefix[prefix.index("--regid") + 1] == "5678"


def test_setpriv_prefix_fails_closed_when_setpriv_unavailable(monkeypatch):
    monkeypatch.delenv("GLYPH_CODEC_ALLOW_UNPRIVILEGED", raising=False)
    monkeypatch.setattr("eval.glyph_eval_runner.shutil.which", lambda name: None)
    with pytest.raises(RunnerError, match="setpriv unavailable"):
        _setpriv_prefix()


def test_setpriv_prefix_allow_unprivileged_opt_out(monkeypatch):
    monkeypatch.setenv("GLYPH_CODEC_ALLOW_UNPRIVILEGED", "1")
    monkeypatch.setattr("eval.glyph_eval_runner.shutil.which", lambda name: None)
    assert _setpriv_prefix() == []


def _write_uid_reporting_codec(directory):
    # tempfile.mkdtemp()-created dirs are 0700 by default, so _run_codec must make this tree
    # readable for the dropped-privilege uid itself (issue #57) -- otherwise the codec can't
    # even open its own entrypoint script once root is actually dropped.
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entrypoints": {
                    "compress": ["python3", "compress.py", "--input", "{input}", "--output", "{output}"],
                    "decompress": ["python3", "decompress.py", "--input", "{input}", "--output", "{output}"],
                },
            }
        )
    )
    body = (
        "import argparse, os\n"
        "p = argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output')\n"
        "a = p.parse_args()\n"
        "open(a.output, 'wb').write(str(os.getuid()).encode())\n"
    )
    (directory / "compress.py").write_text(body)
    (directory / "decompress.py").write_text(body)


def _mkdtemp_under_tmp():
    # Real _prepare_artifact uses tempfile.mkdtemp() directly under /tmp (world-traversable,
    # sticky bit) -- pytest's own tmp_path fixture nests under a 0700-owner-only directory
    # chain, which would make ANY dropped-uid test fail for a reason unrelated to the code
    # under test. Match production's actual directory placement instead.
    return Path(tempfile.mkdtemp(prefix="glyph-eval-runner-test-"))


def test_run_codec_executes_untrusted_codec_as_nonroot_uid():
    # Real setpriv/unshare, not mocked -- the codec must observe a non-root uid and must be
    # able to read its own entrypoint despite the artifact dir defaulting to 0700.
    if not _setpriv_usable() or not _unshare_net_usable():
        pytest.skip("setpriv/unshare --net not permitted on this host")
    artifact_dir = _mkdtemp_under_tmp()
    try:
        _write_uid_reporting_codec(artifact_dir)
        caps = ResourceCaps(wall_clock_secs=10)
        out, _secs = _run_codec(artifact_dir, "compress", b"irrelevant", caps)
        assert out != b"0"  # not root
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


def test_run_codec_raises_if_codec_somehow_still_root(monkeypatch):
    # Sanity-check the test above actually asserts something -- force the "still root" branch
    # by disabling both privilege drop and network isolation.
    monkeypatch.setattr("eval.glyph_eval_runner.shutil.which", lambda name: None)
    monkeypatch.setenv("GLYPH_CODEC_ALLOW_UNPRIVILEGED", "1")
    artifact_dir = _mkdtemp_under_tmp()
    try:
        _write_uid_reporting_codec(artifact_dir)
        (artifact_dir / "compress.py").write_text(
            "import argparse\n"
            "p = argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output')\n"
            "a = p.parse_args()\n"
            "open(a.output, 'wb').write(b'0')\n"  # simulate still running as root
        )
        caps = ResourceCaps(wall_clock_secs=10, network=True)  # skip the unshare requirement too
        out, _secs = _run_codec(artifact_dir, "compress", b"irrelevant", caps)
        assert out == b"0"  # demonstrates the assertion above would have caught a real regression
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


# --- zip-slip: a miner-controlled tree listing can't escape the download destination --------


def test_safe_join_accepts_normal_relative_paths(tmp_path):
    assert _safe_join(tmp_path, "weights.bin") == (tmp_path / "weights.bin").resolve()
    assert _safe_join(tmp_path, "sub/dir/file.txt") == (tmp_path / "sub/dir/file.txt").resolve()


@pytest.mark.parametrize("rel", ["../../etc/passwd", "../sibling.bin", "/etc/passwd", "a/../../b"])
def test_safe_join_rejects_escaping_paths(tmp_path, rel):
    with pytest.raises(ValueError, match="escapes destination"):
        _safe_join(tmp_path, rel)


# --- symlink escape in artifact hashing (issue #95): this module's own copy of
# core.artifact.iter_artifact_files/hash_artifact, kept in sync by hand -------------------


def test_iter_artifact_files_rejects_a_symlink(tmp_path):
    target = tmp_path / "outside.txt"
    target.write_text("host secret")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "evil.py").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        list(_iter_artifact_files(artifact))


def test_hash_artifact_does_not_read_through_a_symlink(tmp_path):
    target = tmp_path / "outside.bin"
    target.write_bytes(b"host secret bytes")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "link.bin").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        _hash_artifact(artifact)
