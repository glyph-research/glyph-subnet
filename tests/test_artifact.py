"""issue #48 review: the image-digest regex must reject trailing whitespace/newlines, leading
whitespace, and an embedded/second '@sha256:...' being swallowed into the name portion --
not just accept the happy path."""

from core.artifact import Manifest, is_image_digest_pinned

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
