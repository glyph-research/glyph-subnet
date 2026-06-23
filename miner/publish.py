"""Validate (and optionally upload) a local codec artifact to HuggingFace.

A miner builds a codec directory (manifest.json + compress/decompress + weights), validates
it locally with ``check``, then publishes it here and commits the pinned revision.
"""

from __future__ import annotations

import argparse

from core.constants import DEFAULT_MAX_ARTIFACT_BYTES
from validation.precheck import precheck_artifact_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and publish a Glyph codec artifact")
    parser.add_argument("--path", required=True, help="Local codec artifact directory")
    parser.add_argument("--repo", help="HuggingFace repo id to upload to (namespace/name)")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--max-artifact-bytes", type=int, default=DEFAULT_MAX_ARTIFACT_BYTES)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = precheck_artifact_dir(args.path, max_artifact_bytes=args.max_artifact_bytes)
    for warning in result.warnings:
        print(f"warning: {warning}")
    if not result.ok:
        for error in result.errors:
            print(f"error: {error}")
        raise SystemExit(1)
    print(f"valid codec: hash={result.artifact_hash} bytes={result.artifact_bytes:,}")

    if not args.repo:
        print("no --repo given; validation only")
        return
    if args.dry_run:
        print(f"dry-run: would upload {args.path} -> {args.repo}")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=args.repo, private=args.private, exist_ok=True)
    api.upload_folder(folder_path=args.path, repo_id=args.repo)
    info = api.repo_info(repo_id=args.repo)
    print(f"uploaded {args.path} -> {args.repo}")
    print(f"pin this revision when committing: {info.sha}")


if __name__ == "__main__":
    main()
