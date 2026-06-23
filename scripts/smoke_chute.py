#!/usr/bin/env python3
"""Live smoke test of the deployed glyph-runner chute's /run_stream invocation contract.

Drives the *exact* production dispatch path (``eval.runner_chutes.ChutesRunner``) against a
live deployed chute, for both stream shapes -- inline bytes and URL/range -- and asserts a
bit-exact round-trip. Use it right after deploying to confirm the live URL/auth/request/
response contract end-to-end (issue #7).

Prerequisites:
  * a deployed glyph-runner chute:  chutes deploy eval.chute_app:chute --accept-fee
  * an invocation key:              CHUTES_API_KEY=cpk_... (or --chutes-key-file)
  * a codec artifact on HuggingFace (the chute snapshot_downloads repo@rev). Publish the
    bundled reference codec with `glyph-miner publish`, then pass --repo/--rev.
  * for the URL/range path only: the corpus served as one contiguous blob at --corpus-url.

Examples:
  CHUTES_API_KEY=cpk_... python scripts/smoke_chute.py \
      --repo you/glyph-ref-codec --rev main \
      --chute-url https://<acct>-glyph-runner.chutes.ai
  # add --corpus-url https://<host>/corpus.bin to also exercise the production range-fetch path
"""

from __future__ import annotations

import argparse
import os
import sys

# Work whether or not the package is installed editable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from eval.runner import ArtifactRef, ResourceCaps, StreamInput  # noqa: E402
from eval.runner_chutes import ChutesRunner  # noqa: E402
from eval.streams import RangeSource  # noqa: E402


def _report(label: str, result) -> bool:
    ratio = (result.compressed_bytes / result.raw_bytes) if result.raw_bytes else 0.0
    status = "OK" if result.roundtrip_ok else "FAIL"
    print(
        f"  [{status}] {label}: roundtrip_ok={result.roundtrip_ok} raw={result.raw_bytes} "
        f"compressed={result.compressed_bytes} ratio={ratio:.3f} "
        f"compress_s={result.compress_secs:.3f} decompress_s={result.decompress_secs:.3f}"
    )
    return bool(result.roundtrip_ok)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="HuggingFace repo of the codec artifact")
    parser.add_argument("--rev", default="main", help="HF revision/commit of the codec artifact")
    parser.add_argument("--chute-url", default=None, help="deployed chute base URL (or GLYPH_CHUTE_URL)")
    parser.add_argument("--chutes-key-file", default=None, help="file with the cpk_ key (or CHUTES_API_KEY)")
    parser.add_argument("--corpus-url", default=None, help="corpus blob URL; exercises the URL/range path")
    parser.add_argument("--stream-bytes", type=int, default=4096, help="bytes per smoke stream")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    runner = ChutesRunner(key_file=args.chutes_key_file, base_url=args.chute_url, timeout=args.timeout)
    artifact = ArtifactRef(repo=args.repo, rev=args.rev)
    caps = ResourceCaps()
    print(f"chute:    {runner.base_url}/run_stream")
    print(f"artifact: {args.repo}@{args.rev}")

    ok = True

    # Inline path: the validator sends the bytes in the request body.
    sample = (b"glyph live smoke test. " * (args.stream_bytes // 23 + 1))[: args.stream_bytes]
    ok &= _report("inline", runner.run_stream(artifact, StreamInput("smoke-inline", data=sample), caps=caps))

    # URL/range path: the chute range-fetches the corpus itself.
    if args.corpus_url:
        source = RangeSource(url=args.corpus_url, offset=0, length=args.stream_bytes)
        ok &= _report("url/range", runner.run_stream(artifact, StreamInput("smoke-range", source=source), caps=caps))
    else:
        print("  [skip] url/range: pass --corpus-url to exercise the production range-fetch path")

    print("RESULT:", "all paths bit-exact" if ok else "a path failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
