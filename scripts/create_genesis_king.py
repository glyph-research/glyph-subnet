"""Seed validator state with a genesis king (initial incumbent).

At launch the king-of-the-hill needs an incumbent so challengers have something to beat.
This writes a ``validator_state.json`` whose winner is a committed codec (the reference
codec by default). Run from the repo root after ``pip install -e .``.

  python scripts/create_genesis_king.py --local-path ./reference_codec --hotkey 5GENESIS...
  python scripts/create_genesis_king.py --repo user/glyph-codec --rev <sha> --hotkey 5...
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.state import CommitmentState, ScoreState, ValidatorState, save_state
from core.weights import WinnerEntry
from validation.precheck import precheck_artifact_dir, precheck_codec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed a genesis king into validator state")
    parser.add_argument("--state-dir", default="./state")
    parser.add_argument("--hotkey", required=True, help="ss58 hotkey of the genesis king")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--local-path", help="Local codec artifact directory")
    src.add_argument("--repo", help="HuggingFace codec repo id")
    parser.add_argument("--rev", default="genesis", help="Revision (for --repo)")
    parser.add_argument("--ratio", type=float, default=0.99, help="Genesis incumbent ratio")
    parser.add_argument("--commit-block", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.local_path:
        result = precheck_artifact_dir(args.local_path)
        repo, rev = "genesis/local", "local"
    else:
        result = precheck_codec(args.repo, args.rev)
        repo, rev = args.repo, args.rev
    result.raise_for_status()

    state = ValidatorState()
    key = f"{args.hotkey}:{repo}@{rev}"
    state.commitments[key] = CommitmentState(
        hotkey=args.hotkey,
        repo=repo,
        revision=rev,
        block=args.commit_block,
        artifact_hash=result.artifact_hash,
        artifact_bytes=result.artifact_bytes,
        valid=True,
    )
    state.scores[key] = ScoreState(
        hotkey=args.hotkey,
        repo=repo,
        revision=rev,
        ratio=args.ratio,
        roundtrip_ok=True,
        throughput_bps=0.0,
        valid=True,
        commit_block=args.commit_block,
    )
    state.winner_history = [WinnerEntry(args.hotkey, repo, rev, args.ratio, args.commit_block)]
    if result.artifact_hash:
        state.duplicate_hash_owner[result.artifact_hash] = args.hotkey

    state_path = Path(args.state_dir) / "validator_state.json"
    save_state(state_path, state)
    print(f"genesis king written: {args.hotkey} {repo}@{rev} ratio={args.ratio}")
    print(f"state -> {state_path}")


if __name__ == "__main__":
    main()
