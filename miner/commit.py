"""One-shot miner codec commitment CLI.

Commits exactly one codec artifact (a HuggingFace repo at a pinned revision) for this
hotkey. Commitments are permanent: to submit a different codec, register a new hotkey.
"""

from __future__ import annotations

import argparse
import secrets

from chain.chain import BittensorChain, ChainConfig
from core.commitments import (
    CodecCommitment,
    commitment_digest,
    parse_commitment,
    serialize_commit_phase,
    serialize_reveal_phase,
)
from core.constants import DEFAULT_MAX_ARTIFACT_BYTES, DEFAULT_NETUID
from validation.precheck import precheck_codec


def _resolve_revision(repo: str, revision: str | None) -> str:
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo_id=repo, revision=revision)
    if not info.sha:
        raise ValueError("HuggingFace did not return a pinned revision SHA")
    return info.sha


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Commit one Glyph codec for this hotkey")
    parser.add_argument("--network", default="finney")
    parser.add_argument("--netuid", type=int, default=DEFAULT_NETUID)
    parser.add_argument("--wallet-name", "--wallet.name", dest="wallet_name", default="default")
    parser.add_argument("--hotkey-name", "--wallet.hotkey", dest="hotkey_name", default="default")
    parser.add_argument("--wallet-path", "--wallet.path", dest="wallet_path", default=None)
    parser.add_argument("--model-repo", required=True, help="HuggingFace codec repo id")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-artifact-bytes", type=int, default=DEFAULT_MAX_ARTIFACT_BYTES)
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    revision = _resolve_revision(args.model_repo, args.revision)
    commitment = CodecCommitment(repo=args.model_repo, rev=revision)
    salt = secrets.token_hex(8)
    digest = commitment_digest(commitment.repo, commitment.rev, salt)
    commit_value = serialize_commit_phase(digest)
    reveal_value = serialize_reveal_phase(commitment.repo, commitment.rev, salt)

    if not args.skip_model_check:
        result = precheck_codec(args.model_repo, revision, max_artifact_bytes=args.max_artifact_bytes)
        for warning in result.warnings:
            print(f"warning: {warning}")
        if not result.ok:
            for error in result.errors:
                print(f"error: {error}")
            raise SystemExit(1)
        print(f"precheck ok: artifact_hash={result.artifact_hash} bytes={result.artifact_bytes:,}")

    chain = BittensorChain(
        ChainConfig(
            netuid=args.netuid,
            network=args.network,
            wallet_name=args.wallet_name,
            hotkey_name=args.hotkey_name,
            wallet_path=args.wallet_path,
        )
    )

    existing = chain.get_my_commitment()
    if existing:
        try:
            existing_commitment, _ = parse_commitment(existing)
            print(f"hotkey already committed {existing_commitment.repo}@{existing_commitment.rev}")
        except Exception:
            print("hotkey already has non-empty commitment metadata; refusing to overwrite")
        raise SystemExit(1)

    print(f"hotkey={chain.hotkey}")
    print(f"netuid={args.netuid}")
    print(f"commit phase = {commit_value}")
    print(f"reveal phase = {reveal_value}")

    if args.dry_run:
        print("dry-run: not submitting commitment")
        return

    if not args.yes:
        answer = input("This commitment is permanent for this hotkey. Submit? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("aborted")
            return

    # Two-phase commit-reveal in one shot (exploit vector #9). Phase 1 publishes only the
    # digest; set_commitment waits for finalization, so phase 2's reveal lands on a later
    # block. The earliest-commit tie-break keys off the phase-1 block, so a mempool watcher
    # who only learns repo|rev at reveal time cannot land an earlier commit.
    print("submitting commit phase (hiding digest)...")
    commit_response = chain.set_commitment(commit_value)
    if not getattr(commit_response, "success", True):
        print(f"commit phase failed, not revealing: {commit_response}")
        raise SystemExit(1)

    print("commit included; submitting reveal phase...")
    reveal_response = chain.set_commitment(reveal_value)
    if getattr(reveal_response, "success", False):
        print(f"commitment revealed (keep this salt for your records): salt={salt}")
    else:
        print(f"reveal response: {reveal_response}  salt={salt}")


if __name__ == "__main__":
    main()
