"""Local miner preflight: validate a codec artifact and round-trip it locally.

Lets a miner self-benchmark against the public incumbent and the zstd baseline before
paying to commit (DESIGN §3.3). Accepts either a HuggingFace repo or a local artifact
directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.constants import BASELINE_LEVEL, DEFAULT_MAX_ARTIFACT_BYTES, THROUGHPUT_FLOOR_BPS
from validation.precheck import precheck_artifact_dir, precheck_codec
from eval.runner import ArtifactRef, LocalSubprocessRunner, ResourceCaps, RunnerError, StreamInput
from eval.scoring import zstd_baseline_ratio

_BUNDLED_CORPUS = Path(__file__).resolve().parents[1] / "samples" / "corpus"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check and benchmark a Glyph codec locally")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-repo", help="HuggingFace codec repo id")
    source.add_argument("--local-path", help="Local codec artifact directory")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-artifact-bytes", type=int, default=DEFAULT_MAX_ARTIFACT_BYTES)
    parser.add_argument("--sample-file", default=None, help="File to round-trip; defaults to bundled corpus")
    parser.add_argument("--sample-bytes", type=int, default=65536)
    parser.add_argument("--floor-bps", type=float, default=THROUGHPUT_FLOOR_BPS)
    parser.add_argument("--baseline-level", type=int, default=BASELINE_LEVEL)
    parser.add_argument("--skip-roundtrip", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _load_sample(args) -> bytes:
    if args.sample_file:
        return Path(args.sample_file).read_bytes()[: args.sample_bytes]
    if _BUNDLED_CORPUS.exists():
        data = bytearray()
        for path in sorted(_BUNDLED_CORPUS.iterdir()):
            if path.is_file():
                data += path.read_bytes()
            if len(data) >= args.sample_bytes:
                break
        return bytes(data[: args.sample_bytes])
    return b"glyph sample text " * 4096


def main() -> None:
    args = build_parser().parse_args()

    if args.local_path:
        result = precheck_artifact_dir(args.local_path, max_artifact_bytes=args.max_artifact_bytes)
        local_path = args.local_path
    else:
        from huggingface_hub import HfApi, snapshot_download

        revision = args.revision or HfApi().repo_info(repo_id=args.model_repo, revision=args.revision).sha
        result = precheck_codec(args.model_repo, revision, max_artifact_bytes=args.max_artifact_bytes)
        local_path = snapshot_download(repo_id=args.model_repo, revision=revision) if result.ok else None

    report: dict = {
        "ok": result.ok,
        "errors": result.errors,
        "warnings": result.warnings,
        "artifact_hash": result.artifact_hash,
        "artifact_bytes": result.artifact_bytes,
    }

    if result.ok and not args.skip_roundtrip and local_path:
        sample = _load_sample(args)
        runner = LocalSubprocessRunner()
        artifact = ArtifactRef(repo=args.model_repo or "local", rev="local", local_path=local_path)
        try:
            stream_result = runner.run_stream(
                artifact, StreamInput("smoke", sample), caps=ResourceCaps()
            )
            ratio = stream_result.compressed_bytes / max(stream_result.raw_bytes, 1)
            baseline = zstd_baseline_ratio([sample], level=args.baseline_level)
            report["roundtrip"] = {
                "roundtrip_ok": stream_result.roundtrip_ok,
                "ratio": ratio,
                "baseline_ratio": baseline,
                "beats_baseline": stream_result.roundtrip_ok and ratio < baseline,
                "decompress_throughput_bps": stream_result.decompress_throughput_bps,
                "meets_floor": stream_result.decompress_throughput_bps >= args.floor_bps,
            }
        except RunnerError as exc:
            report["roundtrip"] = {"roundtrip_ok": False, "error": str(exc)}

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"ok={report['ok']}")
        if report.get("artifact_hash"):
            print(f"artifact_hash={report['artifact_hash']} bytes={report['artifact_bytes']:,}")
        for warning in report["warnings"]:
            print(f"warning: {warning}")
        for error in report["errors"]:
            print(f"error: {error}")
        rt = report.get("roundtrip")
        if rt:
            print(f"roundtrip_ok={rt.get('roundtrip_ok')}")
            if "ratio" in rt:
                print(
                    f"ratio={rt['ratio']:.4f} baseline(zstd-{args.baseline_level})={rt['baseline_ratio']:.4f} "
                    f"beats_baseline={rt['beats_baseline']} meets_floor={rt['meets_floor']}"
                )

    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
