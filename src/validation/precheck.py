"""Codec artifact validation and hashing (validator + miner side).

A Glyph codec artifact is executable code by design: the miner ships the compressor and
decompressor. Precheck therefore validates both the artifact contract and obvious
exfiltration paths before the runner executes the artifact in an isolated worker.
"""

from __future__ import annotations

import ast
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from core.artifact import (
    MANIFEST_NAME,
    hash_artifact,
    iter_artifact_files,
    load_manifest,
    local_snapshot_dir,
)
from core.constants import DEFAULT_MAX_ARTIFACT_BYTES

_SOURCE_SUFFIXES = {
    ".py",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".pl",
    ".rb",
    ".js",
    ".ts",
    ".mjs",
    ".cjs",
    ".lua",
    ".r",
    ".jl",
    ".php",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".hh",
    ".rs",
    ".go",
    ".java",
    ".kt",
}
_SOURCE_NAMES = {"makefile", "dockerfile"}
_SHELL_SOURCE_SUFFIXES = {".sh", ".bash", ".zsh", ".fish"}
_SHELL_SOURCE_NAMES = {"makefile", "dockerfile"}
_MAX_REVIEWABLE_SOURCE_BYTES = 2 * 1024 * 1024

_NETWORK_IMPORTS = {
    "aiohttp",
    "azure",
    "boto3",
    "botocore",
    "dropbox",
    "ftplib",
    "google.cloud",
    "grpc",
    "h11",
    "h2",
    "http.client",
    "http.server",
    "httpx",
    "huggingface_hub",
    "imaplib",
    "paramiko",
    "poplib",
    "requests",
    "scp",
    "sftp",
    "smtplib",
    "socket",
    "ssl",
    "telnetlib",
    "urllib",
    "websocket",
    "websockets",
    "xmlrpc",
}
_SHELL_COMMAND_APIS = {
    "os.system",
    "os.popen",
    "popen2.popen2",
    "popen2.popen3",
    "popen2.popen4",
}
_SUBPROCESS_APIS = {
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
    "subprocess.run",
}
_DYNAMIC_IMPORT_APIS = {"__import__", "importlib.import_module"}
_NETWORK_COMMANDS = {
    "aws",
    "azcopy",
    "curl",
    "dig",
    "gcloud",
    "gsutil",
    "host",
    "nc",
    "netcat",
    "nslookup",
    "rclone",
    "rsync",
    "scp",
    "sftp",
    "ssh",
    "telnet",
    "wget",
}
_INLINE_CODE_RUNTIMES = {"bash", "dash", "fish", "node", "perl", "python", "python3", "ruby", "sh", "zsh"}
_URL_RE = re.compile(r"\b(?:https?|ftp|s3|gs)://", re.IGNORECASE)
_DEV_TCP_RE = re.compile(r"/dev/(?:tcp|udp)/", re.IGNORECASE)
_NETWORK_COMMAND_RE = re.compile(
    r"(?<![\w.-])(" + "|".join(re.escape(cmd) for cmd in sorted(_NETWORK_COMMANDS)) + r")(?![\w.-])"
)


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
    # True only when HF answered definitively that the repo/revision does not exist
    # (deleted, renamed, made private) -- never for transient network/5xx/rate-limit
    # failures. Lets the validator distinguish "genuinely gone, stop retrying eventually"
    # from "worth retrying next round" (issue #128).
    repo_not_found: bool = False
    # True when the failure happened at the HF fetch itself (either flavor above), as
    # opposed to a content problem with a successfully fetched artifact (manifest/security/
    # size/duplicate). A fetch failure says nothing bad about the codec -- the crown must
    # not change hands over one (issue #135).
    repo_unreachable: bool = False

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


def _module_blocked(name: str) -> bool:
    lowered = name.lower()
    return any(lowered == blocked or lowered.startswith(f"{blocked}.") for blocked in _NETWORK_IMPORTS)


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _truthy_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _first_string_arg(node: ast.Call) -> str | None:
    if not node.args:
        return None
    first = node.args[0]
    return first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else None


def _literal_command_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return Path(node.value.split()[0]).name.lower() if node.value.split() else None
    if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
        first = node.elts[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return Path(first.value).name.lower()
    return None


def _python_docstring_node_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    docstring_owners = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in ast.walk(tree):
        if not isinstance(node, docstring_owners) or not node.body:
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


def _strip_shell_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        quote: str | None = None
        escaped = False
        kept: list[str] = []
        for char in line:
            if escaped:
                kept.append(char)
                escaped = False
                continue
            if char == "\\":
                kept.append(char)
                escaped = True
                continue
            if quote:
                kept.append(char)
                if char == quote:
                    quote = None
                continue
            if char in {"'", '"'}:
                quote = char
                kept.append(char)
                continue
            if char == "#":
                break
            kept.append(char)
        lines.append("".join(kept))
    return "\n".join(lines)


class _PythonSecurityVisitor(ast.NodeVisitor):
    def __init__(self, rel: str, *, docstring_node_ids: set[int]):
        self.rel = rel
        self.docstring_node_ids = docstring_node_ids
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            self._check_import(alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.module:
            self._check_import(node.module, node.lineno)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = _qualified_name(node.func)
        if name in _SHELL_COMMAND_APIS:
            self.errors.append(f"{self.rel}:{node.lineno}: shell command API {name} is not allowed")
        if name in _SUBPROCESS_APIS:
            command = _literal_command_name(node.args[0]) if node.args else None
            if command in _NETWORK_COMMANDS:
                self.errors.append(
                    f"{self.rel}:{node.lineno}: subprocess network command {command!r} is not allowed"
                )
            for keyword in node.keywords:
                if keyword.arg == "shell" and _truthy_literal(keyword.value):
                    self.errors.append(f"{self.rel}:{node.lineno}: subprocess shell=True is not allowed")
        if name in _DYNAMIC_IMPORT_APIS:
            imported = _first_string_arg(node)
            if imported and _module_blocked(imported):
                self.errors.append(
                    f"{self.rel}:{node.lineno}: dynamic import of network module {imported!r} is not allowed"
                )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if isinstance(node.value, str) and id(node) not in self.docstring_node_ids:
            if _URL_RE.search(node.value):
                self.errors.append(f"{self.rel}:{node.lineno}: external URL/protocol literal is not allowed")
            if _DEV_TCP_RE.search(node.value):
                self.errors.append(f"{self.rel}:{node.lineno}: /dev/tcp or /dev/udp network access is not allowed")

    def _check_import(self, name: str, lineno: int) -> None:
        if _module_blocked(name):
            self.errors.append(f"{self.rel}:{lineno}: network/cloud import {name!r} is not allowed")


def _is_reviewable_source(path: Path) -> bool:
    return path.suffix.lower() in _SOURCE_SUFFIXES or path.name.lower() in _SOURCE_NAMES


def _is_shell_like_source(path: Path) -> bool:
    return path.suffix.lower() in _SHELL_SOURCE_SUFFIXES or path.name.lower() in _SHELL_SOURCE_NAMES


def _entrypoint_security_errors(manifest) -> list[str]:
    errors: list[str] = []
    for label, argv in (("compress", manifest.entrypoints.compress), ("decompress", manifest.entrypoints.decompress)):
        if not argv:
            continue
        command = Path(argv[0]).name.lower()
        if command in _NETWORK_COMMANDS:
            errors.append(f"{label} entrypoint uses network command {command!r}")
        if command in _INLINE_CODE_RUNTIMES and any(token in {"-c", "-e"} for token in argv[1:]):
            errors.append(f"{label} entrypoint uses inline code execution; use a reviewable script file")
        for token in argv:
            if _URL_RE.search(token):
                errors.append(f"{label} entrypoint contains external URL/protocol")
                break
    return errors


def _source_security_errors(root: Path) -> list[str]:
    errors: list[str] = []
    for path in iter_artifact_files(root):
        if not _is_reviewable_source(path):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            size = path.stat().st_size
            if size > _MAX_REVIEWABLE_SOURCE_BYTES:
                errors.append(
                    f"{rel}: source file is {size:,} bytes; split or remove generated code for review"
                )
                continue
            text = path.read_text("utf-8", errors="replace")
        except Exception as exc:
            errors.append(f"{rel}: cannot read source for security review: {exc}")
            continue

        if path.suffix.lower() == ".py":
            try:
                tree = ast.parse(text, filename=rel)
            except SyntaxError as exc:
                errors.append(f"{rel}:{exc.lineno or 0}: invalid Python source: {exc.msg}")
                continue
            visitor = _PythonSecurityVisitor(rel, docstring_node_ids=_python_docstring_node_ids(tree))
            visitor.visit(tree)
            errors.extend(visitor.errors)
        else:
            scanned_text = _strip_shell_comments(text) if _is_shell_like_source(path) else text
            if _URL_RE.search(scanned_text):
                errors.append(f"{rel}: external URL/protocol literals are not allowed")
            if _DEV_TCP_RE.search(scanned_text):
                errors.append(f"{rel}: /dev/tcp or /dev/udp network access is not allowed")

        if _is_shell_like_source(path):
            match = _NETWORK_COMMAND_RE.search(_strip_shell_comments(text))
            if match:
                errors.append(f"{rel}: network command {match.group(1)!r} is not allowed")
    return errors


def artifact_security_errors(directory: str | Path, manifest=None) -> list[str]:
    """Return pre-execution security issues that should block a codec artifact."""

    path = Path(directory)
    if manifest is None:
        manifest = load_manifest(path)
    return [*_entrypoint_security_errors(manifest), *_source_security_errors(path)]


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
    result.errors.extend(manifest.image_issues())
    try:
        result.warnings.extend(_entrypoint_script_warnings(path, manifest))
        result.errors.extend(artifact_security_errors(path, manifest))
    except Exception as exc:
        result.errors.append(f"cannot review artifact: {exc}")

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
    from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

    result = PrecheckResult(repo=repo, revision=revision, ok=False)

    try:
        HfApi().repo_info(repo_id=repo, revision=revision)
    except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
        # A definitive "does not exist" answer from HF (covers deleted/renamed/made-private
        # repos, and GatedRepoError via subclassing), unlike the transient failures below.
        result.repo_not_found = True
        result.repo_unreachable = True
        result.errors.append(f"repo unavailable: {exc}")
        return result
    except Exception as exc:
        result.repo_unreachable = True
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
        result.errors.extend(manifest.image_issues())
        result.warnings.append("artifact download skipped; hash/size not computed")
        result.ok = not result.errors
        return result

    # local_dir materializes real files rather than the cache's default symlinks-into-blobs/
    # layout (issue #66) -- and, since it's the same stable per-(repo, revision) path
    # reign_worker.service.artifact_ref() downloads into, this also lets that later download
    # skip re-fetching content precheck already verified here.
    local = snapshot_download(repo_id=repo, revision=revision, local_dir=local_snapshot_dir(repo, revision))
    return precheck_artifact_dir(local, repo, revision, max_artifact_bytes=max_artifact_bytes)
