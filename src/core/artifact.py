"""Codec artifact contract: manifest schema, argv resolution, deterministic hashing.

A Glyph codec artifact is a directory (a HuggingFace repo in production) containing a
``manifest.json`` that declares two entrypoints, ``compress`` and ``decompress``, each an
argv template using the ``{input}`` and ``{output}`` placeholders. The same module is
used by the validator-side precheck (hashing, validation) and by the runner (execution),
so the contract has a single source of truth.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, Field

MANIFEST_NAME = "manifest.json"
PLACEHOLDER_INPUT = "{input}"
PLACEHOLDER_OUTPUT = "{output}"
_EXCLUDED_PARTS = {"__pycache__", ".git", ".cache"}
# A digest-pinned OCI image reference: name@sha256:<64 lowercase hex chars>. The name portion
# excludes '@' (so an embedded/second '@sha256:...' can't be swallowed as part of the name --
# a reference syntactically has exactly one '@') and whitespace (rejects leading/trailing
# whitespace or a trailing newline). Deliberately excludes bare tags (":latest", ":1.0") --
# see is_image_digest_pinned()/Manifest.image_issues().
_IMAGE_DIGEST_RE = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")
DEFAULT_WARMUP_READY_FILE = "/scratch/.glyph_ready"
DEFAULT_WARMUP_TIMEOUT_SECS = 300.0


def is_image_digest_pinned(image: str) -> bool:
    """True iff ``image`` is pinned by an OCI sha256 digest, not a mutable tag (issue #48)."""

    return bool(_IMAGE_DIGEST_RE.fullmatch(image))


class Entrypoints(BaseModel):
    compress: list[str] = Field(min_length=1)
    decompress: list[str] = Field(min_length=1)


class Warmup(BaseModel):
    """Networked warmup config for a manifest-declared ``image`` (issue #48).

    Only meaningful when ``Manifest.image`` is set: a miner-published image may need
    network access once to install/download deps or weights before the sealed, offline
    benchmark run. Two mutually exclusive readiness protocols, chosen by whether
    ``command`` is set:

    - ``command`` set (recommended, simpler): the container's own CMD/ENTRYPOINT is
      overridden with a lightweight keep-alive, and ``command`` is run via ``docker exec``
      as the actual warmup step -- readiness = it exits 0.
    - ``command`` unset: the image's own CMD/ENTRYPOINT runs unmodified and must be a
      long-running process that does its own initialization and creates ``ready_file``
      when done (it must not exit, or there is nothing left to ``docker exec`` the scored
      entrypoint into afterward) -- polled from the host up to ``timeout_secs``.

    Either way this is a hard, fail-closed deadline: a codec that never signals ready is
    killed and fails the round rather than running unbounded.
    """

    command: list[str] | None = None
    ready_file: str = DEFAULT_WARMUP_READY_FILE
    timeout_secs: float = DEFAULT_WARMUP_TIMEOUT_SECS


class Manifest(BaseModel):
    schema_version: int = 1
    name: str = "codec"
    entrypoints: Entrypoints
    resources: dict = Field(default_factory=dict)
    license: str = "unknown"
    # Digest-pinned Docker image the codec runs in (e.g. "ghcr.io/user/mycodec@sha256:...").
    # None (default) -> the operator-supplied --docker-image / DEFAULT_DOCKER_IMAGE, the
    # existing ephemeral --network-none-from-start path with no warmup. Set -> the
    # warmup(network on) -> seal(network off) -> benchmark lifecycle in runner_docker.py.
    image: str | None = None
    warmup: Warmup | None = None

    def placeholder_issues(self) -> list[str]:
        issues: list[str] = []
        for label, argv in (("compress", self.entrypoints.compress), ("decompress", self.entrypoints.decompress)):
            joined = " ".join(argv)
            if PLACEHOLDER_INPUT not in joined:
                issues.append(f"{label} entrypoint is missing the {PLACEHOLDER_INPUT} placeholder")
            if PLACEHOLDER_OUTPUT not in joined:
                issues.append(f"{label} entrypoint is missing the {PLACEHOLDER_OUTPUT} placeholder")
        return issues

    def image_issues(self) -> list[str]:
        """Fail-closed: a declared ``image`` MUST be pinned by digest, never a mutable tag.

        All validators must run byte-identical containers or scores diverge on-chain (issue
        #48) -- a floating tag could resolve to different bytes depending on when/where each
        validator pulled it.
        """

        if self.image is None:
            return []
        if not is_image_digest_pinned(self.image):
            return [
                f"manifest image {self.image!r} must be pinned by digest "
                "(e.g. 'repo@sha256:<64 hex chars>'), not a mutable tag"
            ]
        return []


_UNSAFE_PATH_CHARS_RE = re.compile(r"[^A-Za-z0-9_.-]")


def local_snapshot_dir(repo: str, revision: str) -> Path:
    """A stable, per-(repo, revision) directory for a materialized HuggingFace snapshot.

    Passing this as ``snapshot_download``'s ``local_dir`` makes it write real files instead
    of the cache's default symlinks-into-``blobs/`` layout -- those symlinks dangle once the
    snapshot directory is bind-mounted into a container alone, without the separate ``blobs/``
    directory they point into (issue #66). ``revision`` is a specific pinned commit, so the
    same (repo, revision) always resolves to identical content -- safe to reuse indefinitely
    and share across rounds (the incumbent is re-evaluated every round) rather than
    re-downloading into a fresh temp dir on every call.

    Keyed off a hash of the exact (repo, revision) pair, not a naive char-sanitized path: two
    different repos (e.g. ``"a/b"`` and ``"a_b"``) could otherwise sanitize to the same
    directory name, and repo/revision are commitment-controlled strings that must not be
    trusted as raw path components (path traversal via ``..``). The sanitized prefix is kept
    only for human debuggability -- the hash suffix is what actually guarantees uniqueness.
    """

    digest = hashlib.sha256(f"{repo}@{revision}".encode("utf-8")).hexdigest()
    readable_prefix = _UNSAFE_PATH_CHARS_RE.sub("_", f"{repo}@{revision}")[:80]
    return Path(tempfile.gettempdir()) / "glyph-artifact-snapshots" / f"{readable_prefix}-{digest}"


def manifest_path(artifact_dir: str | Path) -> Path:
    path = Path(artifact_dir)
    return path / MANIFEST_NAME if path.is_dir() else path


def load_manifest(artifact_dir: str | Path) -> Manifest:
    data = json.loads(manifest_path(artifact_dir).read_text())
    return Manifest.model_validate(data)


def resolve_argv(template: list[str], input_path: str | Path, output_path: str | Path) -> list[str]:
    """Substitute the {input}/{output} placeholders in an entrypoint argv template."""

    resolved: list[str] = []
    for token in template:
        token = token.replace(PLACEHOLDER_INPUT, str(input_path))
        token = token.replace(PLACEHOLDER_OUTPUT, str(output_path))
        resolved.append(token)
    return resolved


class ArtifactSymlinkError(ValueError):
    """A codec artifact contains a symlink (issue #95).

    ``Path.is_file()``/``read_bytes()`` follow symlinks with no check that the resolved
    target stays inside the artifact root -- a miner-supplied symlink could point anywhere
    readable on the host doing the hashing (e.g. ``/proc/self/environ``), and since the
    target depends on the local filesystem of whichever validator hashes it, different
    validators could compute different hashes for the identical on-chain commitment.
    Codec artifacts have no legitimate need for symlinks, so any is rejected outright rather
    than silently followed or skipped.
    """


def iter_artifact_files(root: str | Path) -> Iterator[Path]:
    root = Path(root)
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        # Exclude reserved dirs by their position *within the artifact*, not the absolute
        # path (e.g. a HuggingFace snapshot lives under ~/.cache/huggingface/...).
        rel_parts = path.relative_to(root).parts
        if _EXCLUDED_PARTS.intersection(rel_parts):
            continue
        if path.is_symlink():
            raise ArtifactSymlinkError(
                f"artifact contains a symlink at {path.relative_to(root).as_posix()!r}; "
                "symlinks are not allowed in codec artifacts"
            )
        if path.is_file():
            yield path


def hash_artifact(root: str | Path) -> tuple[str, int]:
    """Deterministic (path, file-hash) tree sha256 over the artifact, plus total bytes.

    This is the canonical artifact hash used for the on-chain-commitment match (recomputed
    at precheck) and for the cross-hotkey duplicate-hash disqualification (anti-copy gate).
    """

    root = Path(root)
    digest = hashlib.sha256()
    total = 0
    for path in iter_artifact_files(root):
        rel = path.relative_to(root).as_posix()
        data = path.read_bytes()
        total += len(data)
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest(), total
