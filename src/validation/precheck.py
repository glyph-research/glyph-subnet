"""Codec artifact validation and hashing (validator + miner side).

Unlike a model-benchmark subnet, a Glyph codec artifact *is* executable code by design --
the miner ships the decompressor -- so there is no "no remote code" ban here. Isolation of
untrusted artifacts is the runner/sandbox's responsibility (DESIGN §6). Precheck instead
validates the manifest/entrypoint contract, enforces the artifact-size cap, and computes
the canonical artifact hash used for the on-chain match and duplicate-hash disqualification.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from core.artifact import (
    MANIFEST_NAME,
    hash_artifact,
    iter_artifact_files,
    load_manifest,
)
from core.constants import DEFAULT_MAX_ARTIFACT_BYTES


@dataclass
class PrecheckResult:
    repo: str
    revision: str
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifact_hash: str | None = None
    artifact_bytes: int | None = None
    name: str | None = None
    license: str | None = None

    def raise_for_status(self) -> None:
        if not self.ok:
            raise ValueError("; ".join(self.errors) or "codec precheck failed")


def _entrypoint_script_warnings(directory: Path, manifest) -> list[str]:
    """Warn if an entrypoint references a local script that is not in the artifact."""

    present = {p.relative_to(directory).as_posix() for p in iter_artifact_files(directory)}
    warnings: list[str] = []
    for argv in (manifest.entrypoints.compress, manifest.entrypoints.decompress):
        for token in argv:
            if token.startswith("-") or token in {"{input}", "{output}"}:
                continue
            if token.endswith((".py", ".sh", ".bin")) and not token.startswith("/"):
                if token not in present:
                    warnings.append(f"entrypoint references missing local file: {token}")
    return warnings


def precheck_artifact_dir(
    directory: str | Path,
    repo: str = "",
    revision: str = "",
    *,
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> PrecheckResult:
    """Validate a codec artifact that is already present on local disk."""

    result = PrecheckResult(repo=repo, revision=revision, ok=False)
    path = Path(directory)

    if not (path / MANIFEST_NAME).exists():
        result.errors.append(f"missing {MANIFEST_NAME}")
        return result

    try:
        manifest = load_manifest(path)
    except Exception as exc:
        result.errors.append(f"invalid {MANIFEST_NAME}: {exc}")
        return result

    result.name = manifest.name
    result.license = manifest.license
    result.errors.extend(manifest.placeholder_issues())
    result.warnings.extend(_entrypoint_script_warnings(path, manifest))

    try:
        digest, total = hash_artifact(path)
        result.artifact_hash = digest
        result.artifact_bytes = total
        if total == 0:
            result.errors.append("artifact is empty")
        elif total > max_artifact_bytes:
            result.errors.append(f"artifact is {total:,} bytes; cap is {max_artifact_bytes:,}")
    except Exception as exc:
        result.errors.append(f"cannot hash artifact: {exc}")

    result.ok = not result.errors
    return result


def precheck_codec(
    repo: str,
    revision: str,
    *,
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    download: bool = True,
) -> PrecheckResult:
    """Validate a HuggingFace-hosted codec artifact at a pinned revision.

    With ``download=True`` the whole artifact is snapshotted and hashed. With
    ``download=False`` only the manifest is fetched (fast dry run); the artifact hash and
    size are left unverified.
    """

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    result = PrecheckResult(repo=repo, revision=revision, ok=False)

    try:
        HfApi().repo_info(repo_id=repo, revision=revision)
    except Exception as exc:
        result.errors.append(f"repo unavailable: {exc}")
        return result

    if not download:
        try:
            manifest_file = hf_hub_download(repo_id=repo, revision=revision, filename=MANIFEST_NAME)
        except Exception as exc:
            result.errors.append(f"cannot fetch {MANIFEST_NAME}: {exc}")
            return result
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp)
            (staged / MANIFEST_NAME).write_bytes(Path(manifest_file).read_bytes())
            try:
                manifest = load_manifest(staged)
            except Exception as exc:
                result.errors.append(f"invalid {MANIFEST_NAME}: {exc}")
                return result
        result.name = manifest.name
        result.license = manifest.license
        result.errors.extend(manifest.placeholder_issues())
        result.warnings.append("artifact download skipped; hash/size not computed")
        result.ok = not result.errors
        return result

    local = snapshot_download(repo_id=repo, revision=revision)
    return precheck_artifact_dir(local, repo, revision, max_artifact_bytes=max_artifact_bytes)
