"""issue #48 review: the image-digest regex must reject trailing whitespace/newlines, leading
whitespace, and an embedded/second '@sha256:...' being swallowed into the name portion --
not just accept the happy path."""

import tempfile
from pathlib import Path

import pytest

from core.artifact import Manifest, is_image_digest_pinned, local_snapshot_dir

_DIGEST = "a" * 64
_VALID = f"ghcr.io/user/mycodec@sha256:{_DIGEST}"


def test_valid_digest_reference_is_pinned():
    assert is_image_digest_pinned(_VALID) is True


def test_mutable_tag_is_not_pinned():
    assert is_image_digest_pinned("ghcr.io/user/mycodec:latest") is False


def test_trailing_newline_is_rejected():
    assert is_image_digest_pinned(_VALID + "\n") is False


def test_leading_whitespace_is_rejected():
    assert is_image_digest_pinned(" " + _VALID) is False


def test_trailing_whitespace_is_rejected():
    assert is_image_digest_pinned(_VALID + " ") is False


def test_double_digest_marker_is_rejected():
    # A second '@sha256:...' must not be swallowed into the name portion -- a reference
    # syntactically has exactly one '@'.
    assert is_image_digest_pinned(f"{_VALID}@sha256:{'b' * 64}") is False


def test_uppercase_hex_digest_is_rejected():
    assert is_image_digest_pinned(f"ghcr.io/user/mycodec@sha256:{'A' * 64}") is False


def test_short_digest_is_rejected():
    assert is_image_digest_pinned(f"ghcr.io/user/mycodec@sha256:{'a' * 63}") is False


def _manifest(image=None):
    return Manifest.model_validate(
        {
            "entrypoints": {
                "compress": ["python3", "c.py", "--input", "{input}", "--output", "{output}"],
                "decompress": ["python3", "d.py", "--input", "{input}", "--output", "{output}"],
            },
            "image": image,
        }
    )


def test_manifest_with_no_image_has_no_image_issues():
    assert _manifest(image=None).image_issues() == []


def test_manifest_image_issues_flags_mutable_tag():
    issues = _manifest(image="ghcr.io/user/mycodec:latest").image_issues()
    assert len(issues) == 1
    assert "digest" in issues[0]


def test_manifest_image_issues_accepts_valid_digest():
    assert _manifest(image=_VALID).image_issues() == []


# --- local_snapshot_dir: a stable per-(repo, revision) path, not a fresh temp dir (#66) -----


def test_local_snapshot_dir_is_stable_for_same_repo_and_revision():
    a = local_snapshot_dir("veterandad/glyph-codec-strong", "8027fb9ea90b8d5b6fefb5e57ff91f3385c931f8")
    b = local_snapshot_dir("veterandad/glyph-codec-strong", "8027fb9ea90b8d5b6fefb5e57ff91f3385c931f8")
    assert a == b


def test_local_snapshot_dir_differs_by_repo_or_revision():
    base = local_snapshot_dir("org/repo", "abc123def456")
    assert local_snapshot_dir("org/other-repo", "abc123def456") != base
    assert local_snapshot_dir("org/repo", "000000000000") != base


def test_local_snapshot_dir_is_a_single_path_component_no_traversal():
    # repo/revision are commitment-controlled strings and must never be trusted as raw path
    # components -- a single flat directory name (hash-suffixed) sidesteps both path
    # traversal ("../../etc") and nested-slash surprises entirely.
    path = local_snapshot_dir("../../etc/passwd", "../../also-evil")
    assert ".." not in path.parts
    base = Path(tempfile.gettempdir()) / "glyph-artifact-snapshots"
    assert path.parent == base
    assert path.is_relative_to(base)


def test_local_snapshot_dir_does_not_collide_across_naive_sanitization():
    # "a/b" and "a_b" would sanitize to the same string under a naive char-replace scheme --
    # the hash suffix must still keep them distinct.
    assert local_snapshot_dir("a/b", "rev") != local_snapshot_dir("a_b", "rev")


# --- real (not mocked) network test: local_dir=local_snapshot_dir(...) produces real files,
# never the cache's default symlinks-into-blobs/ layout (issue #66) --------------------------


def _hf_reachable() -> bool:
    import urllib.request

    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)  # noqa: S310
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _hf_reachable(), reason="huggingface.co not reachable from this host")
def test_snapshot_download_with_local_snapshot_dir_produces_no_symlinks(tmp_path, monkeypatch):
    from huggingface_hub import snapshot_download

    # Route HF's own cache through tmp_path so this doesn't pollute/depend on this host's
    # real ~/.cache/huggingface, then verify the SAME local_snapshot_dir() this repo's
    # download call sites use produces real files, not the cache's default symlinks.
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    dest = local_snapshot_dir("hf-internal-testing/tiny-random-gpt2", "main")
    try:
        local = Path(snapshot_download("hf-internal-testing/tiny-random-gpt2", revision="main", local_dir=dest))
        files = [p for p in local.rglob("*") if p.is_file()]
        assert files, "expected at least one downloaded file"
        assert not any(p.is_symlink() for p in files), "local_dir download must materialize real files"
    finally:
        import shutil

        shutil.rmtree(dest, ignore_errors=True)
