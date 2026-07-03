"""Heavy eval-runner helpers for the glyph eval chutes. Baked into the chute IMAGE (site-packages)
by chute_app.build_image() and imported lazily inside the cords. Keeping this code OUT of the uploaded
chute entry module keeps that entry small enough that the chutes-TEE code-verification (aegis cllmv)
does not trip -- an entry module carrying the full runner verifies (aegis) but never becomes routable
(every invocation 500s "No infrastructure available"). Verified empirically 2026-06/07."""
import base64
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

DEFAULT_MAX_ARTIFACT_BYTES = 10 * 2**30
VRAM_CAP_BYTES = 24 * 2**30
RAM_CAP_BYTES = 32 * 2**30
COMPRESS_BUDGET_SECS = (32 * 2**20) / (10 * 1024)
MANIFEST_NAME = "manifest.json"
PLACEHOLDER_INPUT = "{input}"
PLACEHOLDER_OUTPUT = "{output}"
_EXCLUDED_PARTS = {"__pycache__", ".git", ".cache"}
_ENV_ALLOWLIST = {
    "CUDA_VISIBLE_DEVICES", "GLYPH_TS_ZIP_DEVICE", "GLYPH_TS_ZIP_THREADS", "LANG", "LC_ALL",
    "LD_LIBRARY_PATH", "NVIDIA_DRIVER_CAPABILITIES", "NVIDIA_VISIBLE_DEVICES", "PATH", "PYTHONPATH",
}
_HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_artifact_files(root: Path):
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        rel_parts = path.relative_to(root).parts
        if path.is_file() and not _EXCLUDED_PARTS.intersection(rel_parts):
            yield path


def _hash_artifact(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    for path in _iter_artifact_files(root):
        rel = path.relative_to(root).as_posix()
        data = path.read_bytes()
        total += len(data)
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest(), total


def _load_manifest(artifact_dir: Path) -> dict:
    data = json.loads((artifact_dir / MANIFEST_NAME).read_text())
    eps = data.get("entrypoints") or {}
    for label in ("compress", "decompress"):
        argv = eps.get(label)
        if not isinstance(argv, list) or not argv:
            raise ValueError(f"manifest entrypoints.{label} missing/empty")
        joined = " ".join(argv)
        if PLACEHOLDER_INPUT not in joined or PLACEHOLDER_OUTPUT not in joined:
            raise ValueError(f"{label} entrypoint missing {{input}}/{{output}} placeholder")
    return data


def _resolve_argv(template: list[str], input_path: Path, output_path: Path) -> list[str]:
    out = []
    for token in template:
        token = token.replace(PLACEHOLDER_INPUT, str(input_path)).replace(PLACEHOLDER_OUTPUT, str(output_path))
        out.append(token)
    return out


class RunnerError(Exception):
    pass


@dataclass
class ResourceCaps:
    wall_clock_secs: float = COMPRESS_BUDGET_SECS
    ram_bytes: int = RAM_CAP_BYTES
    vram_bytes: int = VRAM_CAP_BYTES
    artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES
    network: bool = False


def _subprocess_env(home: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k in _ENV_ALLOWLIST}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["HOME"] = str(home)
    env["XDG_CACHE_HOME"] = str(home / ".cache")
    env["TMPDIR"] = str(home)
    env["NO_PROXY"] = "*"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    return env


def _exec(argv: list[str], cwd: Path, caps: ResourceCaps, home: Path) -> float:
    # Untrusted codec runs here with the network dropped (unshare --net) while the stream is
    # present; trusted prep (download) runs before this with no untrusted code.
    wrapped = argv
    if not caps.network:
        if shutil.which("unshare"):
            wrapped = ["unshare", "--net", "--", *argv]
        else:
            raise RunnerError("network isolation unavailable for untrusted artifact execution")
    start = time.perf_counter()
    try:
        proc = subprocess.run(wrapped, cwd=str(cwd), env=_subprocess_env(home),
                              timeout=caps.wall_clock_secs, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired as exc:
        raise RunnerError(f"entrypoint timed out after {caps.wall_clock_secs:.0f}s") from exc
    if proc.returncode != 0:
        raise RunnerError(f"entrypoint exited {proc.returncode}: {proc.stderr.decode('utf-8','replace')[-500:]}")
    return time.perf_counter() - start


def _run_codec(artifact_dir: Path, which: str, data: bytes, caps: ResourceCaps) -> tuple[bytes, float]:
    """Run the compress|decompress entrypoint in a fresh network-dropped sandbox; return (out, secs)."""
    manifest = _load_manifest(artifact_dir)
    argv_tpl = manifest["entrypoints"][which]
    with tempfile.TemporaryDirectory(prefix=f"glyph-{which}-") as tmp:
        td = Path(tmp)
        infile = td / "in.bin"
        outfile = td / "out.bin"
        infile.write_bytes(data)
        secs = _exec(_resolve_argv(argv_tpl, infile, outfile), artifact_dir, caps, td)
        if not outfile.exists():
            raise RunnerError(f"{which} produced no output")
        return outfile.read_bytes(), secs


def _hf_snapshot(repo: str, rev: str, dest: Path) -> Path:
    """stdlib-only equivalent of huggingface_hub.snapshot_download for a public repo.

    Lists the model tree via the HF API then downloads each file via /resolve, preserving relative
    paths. Uses urllib (stdlib) so the chute image needs NO huggingface_hub/requests -- it can be the
    SAME minimal `pip install zstandard` image as our working control chute (bundling huggingface_hub
    and its transitive deps changes the image filesystem and breaks the chutes-TEE gateway relay)."""
    import urllib.parse
    import urllib.request

    def _get(url: str, want_json: bool = False):
        req = urllib.request.Request(url, headers={"User-Agent": "glyph-eval/1"})
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            body = resp.read()
        return json.loads(body) if want_json else body

    quoted = urllib.parse.quote(repo, safe="/")
    tree = _get(f"{_HF_ENDPOINT}/api/models/{quoted}/tree/{urllib.parse.quote(rev, safe='')}?recursive=true", True)
    files = [e["path"] for e in tree if e.get("type") == "file"]
    if not files:
        raise ValueError("artifact repo has no files")
    for rel in files:
        url = f"{_HF_ENDPOINT}/{quoted}/resolve/{urllib.parse.quote(rev, safe='')}/{urllib.parse.quote(rel)}"
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_get(url))
    return dest


def _prepare_artifact(artifact: dict) -> tuple[Path, str]:
    """HF snapshot (stdlib urllib) + manifest validation + tree hash. `artifact` is a plain dict
    {repo, rev, sha256?} (NOTE: reduced precheck -- manifest/placeholder validation + hash. The full
    source-scan precheck from validation/precheck.py must be re-inlined before scoring untrusted
    miner codecs in prod)."""
    repo, rev, sha = artifact["repo"], artifact["rev"], artifact.get("sha256")
    try:
        local = _hf_snapshot(repo, rev, Path(tempfile.mkdtemp(prefix="artifact-")))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"artifact fetch failed for {repo}@{rev}: {exc}") from exc
    _load_manifest(local)  # raises ValueError on a bad manifest
    digest, total = _hash_artifact(local)
    if total > DEFAULT_MAX_ARTIFACT_BYTES:
        raise ValueError(f"artifact too large: {total} bytes")
    if sha and digest != sha:
        raise ValueError(f"artifact hash mismatch: got {digest}, expected {sha}")
    return local, digest


def _materialize(src: dict) -> bytes:
    """`src` is a plain dict {stream_id, inline_b64?, url?, offset?, length?}."""
    inline = src.get("inline_b64")
    if inline is not None:
        return base64.b64decode(inline)
    url = src.get("url")
    if url:
        import urllib.request

        headers = {"User-Agent": "glyph-eval/1"}
        length = src.get("length") or 0
        if length:
            offset = src.get("offset") or 0
            headers["Range"] = f"bytes={offset}-{offset + length - 1}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            return resp.read()
    raise ValueError("stream source needs inline_b64 or url")


def _compress(req) -> dict:
    stream = req.stream or {}
    data = _materialize(stream)
    sid = stream.get("stream_id", "")
    try:
        local, digest = _prepare_artifact(req.artifact)
    except ValueError as exc:
        return {"stream_id": sid, "raw_bytes": len(data), "compressed_bytes": 0, "compress_secs": 0.0,
                "blob_b64": "", "blob_hash": "", "source_sha256": "", "artifact_hash": None, "error": str(exc)}
    src_hash = hashlib.sha256(data).hexdigest()
    caps = ResourceCaps(wall_clock_secs=req.wall_clock_secs)
    try:
        blob, secs = _run_codec(local, "compress", data, caps)
    except RunnerError as exc:
        return {"stream_id": sid, "raw_bytes": len(data), "compressed_bytes": 0, "compress_secs": 0.0,
                "blob_b64": "", "blob_hash": "", "source_sha256": "", "artifact_hash": digest, "error": str(exc)}
    return {"stream_id": sid, "raw_bytes": len(data), "compressed_bytes": len(blob), "compress_secs": secs,
            "blob_b64": base64.b64encode(blob).decode("ascii"), "blob_hash": hashlib.sha256(blob).hexdigest(),
            "source_sha256": src_hash, "artifact_hash": digest, "error": None}


def _decompress(req) -> dict:
    sid = req.stream_id or ""
    try:
        local, digest = _prepare_artifact(req.artifact)
    except ValueError as exc:
        return {"stream_id": sid, "raw_bytes": 0, "decompress_secs": 0.0, "output_sha256": "",
                "artifact_hash": None, "error": str(exc)}
    caps = ResourceCaps(wall_clock_secs=req.wall_clock_secs)
    try:
        blob = base64.b64decode(req.blob_b64 or "")
        out, secs = _run_codec(local, "decompress", blob, caps)
    except RunnerError as exc:
        return {"stream_id": sid, "raw_bytes": 0, "decompress_secs": 0.0, "output_sha256": "",
                "artifact_hash": digest, "error": str(exc)}
    return {"stream_id": sid, "raw_bytes": len(out), "decompress_secs": secs,
            "output_sha256": hashlib.sha256(out).hexdigest(), "artifact_hash": digest, "error": None}


def run_compress(req: dict) -> dict:
    ns = SimpleNamespace(artifact=req["artifact"], stream=req.get("stream"),
                         stream_id=req.get("stream_id"), blob_b64=req.get("blob_b64"),
                         wall_clock_secs=req.get("wall_clock_secs", 3600.0))
    return _compress(ns)


def run_decompress(req: dict) -> dict:
    ns = SimpleNamespace(artifact=req["artifact"], stream=req.get("stream"),
                         stream_id=req.get("stream_id"), blob_b64=req.get("blob_b64"),
                         wall_clock_secs=req.get("wall_clock_secs", 3600.0))
    return _decompress(ns)
