"""Top-level Glyph validator orchestrator.

Composes the reign worker (evaluate + king-of-the-hill) and the weight setter
(temporal-burn weights) in one process. The same pieces can run as separate PM2 services
(``reign_worker``, ``weight_setter``); this orchestrator is the all-in-one path and the
offline M0 demo.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from bittensor.utils.btlogging import logging as bt_logging

from chain.chain import BittensorChain, ChainConfig
from core.commitments import (
    WinnerCommitment,
    parse_commit_phase_by_hotkey,
    parse_commitments_by_hotkey,
    prune_commit_phase_seen,
    serialize_winner_commitment,
)
from core.constants import (
    BASELINE_LEVEL,
    BURN_UID,
    COMMIT_PHASE_MAX_AGE_BLOCKS,
    COMMIT_POLL_INTERVAL_SECS,
    COMPRESS_BUDGET_SECS,
    CONVICTION_TRACKING_START_BLOCK,
    DEFAULT_MAX_ARTIFACT_BYTES,
    DEFAULT_NETUID,
    DEFAULT_WIN_MARGIN,
    EVAL_BENCHMARK_SOURCE,
    EVAL_BENCHMARK_STREAMS,
    EVAL_SOURCE,
    EVAL_STREAM_BYTES,
    EVAL_STREAMS,
    PRECHECK_FULL_RECHECK_INTERVAL_BLOCKS,
    REPO_NOT_FOUND_EXCLUDE_STREAK,
    REFERENCE_SKU,
    SCORING_VERSION,
    THROUGHPUT_FLOOR_BPS,
    WINDOW_ANCHOR_BLOCK,
    WINNER_LIMIT,
)
from core.log_config import add_logging_args
from core.conviction import conviction_report, ledger_catchup, ledger_grid
from core.state import (
    CommitmentState,
    ValidatorState,
    load_state,
    save_state,
    score_is_comparable,
)
from core.version import assert_weights_version_matches, local_version_key
from core.wandb_logger import WandbLogger, build_round_metrics, build_weights_metrics, make_wandb_logger
from core.weights import WinnerEntry, compact_history, promote_winner, rank_key, should_promote
from eval.corpus import StaticLocalProvider
from eval.evaluator import paired_eval
from eval.live_bench import (
    LivePrefetcher,
    LiveSnapshotStore,
    SnapshotAppendedProvider,
    live_benchmark_spec,
)
from eval.live_corpus import resolve_live_corpus
from eval.runner import ArtifactRef, LocalSubprocessRunner, ResourceCaps
from eval.scoring import source_ratio_breakdown, stream_ratio, zstd_baseline_ratio
from eval.streams import derive_seed, sample_source_streams
from validation.precheck import precheck_artifact_dir, precheck_codec
from reign_worker.service import run_round
from weight_setter.service import decide_weights, resolve_force_burn

if TYPE_CHECKING:
    from eval.runner_chutes import ChutesRunner
    from eval.runner_docker import DockerRunner

__all__ = ["build_parser", "run_once", "run_reign_only", "run_offline_demo", "decide_weights", "main"]

# Bundled tiny sample corpus used by --offline-demo when --corpus-dir isn't passed (issue #71
# retired the shared, owner-published mixed corpus a real round used to read from here).
DEFAULT_DEMO_CORPUS_DIR = Path(__file__).resolve().parents[2] / "samples" / "corpus"

# The CLI's own default --docker-image, built by scripts/install_deps.sh. Deliberately not
# eval.runner_docker.DEFAULT_DOCKER_IMAGE, which stays a generic pullable image for
# DockerRunner's own unit tests -- an operator who omits --docker-image should get the real
# zstandard-enabled runner, not a bare python image that lacks it.
DEFAULT_VALIDATOR_DOCKER_IMAGE = "glyph-runner-default:latest"


# Argument parsing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Glyph compression validator")
    parser.add_argument("--network", default="finney")
    parser.add_argument("--netuid", type=int, default=DEFAULT_NETUID)
    parser.add_argument("--wallet-name", "--wallet.name", dest="wallet_name", default="default")
    parser.add_argument("--hotkey-name", "--wallet.hotkey", dest="hotkey_name", default="default")
    parser.add_argument("--wallet-path", "--wallet.path", dest="wallet_path", default=None)
    parser.add_argument("--state-dir", default="./state")
    parser.add_argument("--salt-file", default=None)
    parser.add_argument(
        "--runner",
        choices=["local", "chutes", "docker"],
        default="docker",
        help="'docker' (default, product direction) runs compress/decompress as ephemeral local "
        "Docker containers on operator-controlled hardware/GPU -- see --docker-gpu below. "
        "'chutes' dispatches to the deployed Chutes eval chutes instead (requires the "
        "platform-mandated pro_6000 SKU, subject to Chutes availability). 'local' runs "
        "on this host's own OS user, network-isolated by default (see --unsafe-local-no-sandbox).",
    )
    parser.add_argument(
        "--unsafe-local-no-sandbox",
        action="store_true",
        help="Refused for --runner local: this path always evaluates real on-chain "
        "commitments (even on testnet), never your own local codec, so it is always "
        "network-isolated and this flag is rejected outright. To test your OWN codec "
        "unsandboxed, use --offline-demo --local-codec instead (no chain, no real "
        "commitments involved).",
    )
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help=(
            "StaticLocalProvider corpus directory for --offline-demo ONLY (default: the "
            f"bundled {DEFAULT_DEMO_CORPUS_DIR.name!r} sample). A real round (no "
            "--offline-demo) always builds its corpus live from HuggingFace, keyed by the "
            "round's chain beacon -- see eval.live_corpus.resolve_live_corpus (issue #71) -- "
            "and refuses this flag outright."
        ),
    )
    parser.add_argument("--reference-sku", default=REFERENCE_SKU)
    parser.add_argument("--chutes-key-file", default=None)
    parser.add_argument(
        "--blockmachine-key-file",
        default=None,
        help="Optional file containing a blockmachine.io API key (Standard plan) -- makes it "
        "the preferred archive source for the conviction-ledger backfill (~10x faster than "
        "the public archive node, issue #151). Defaults to the BLOCKMACHINE_API_KEY env "
        "var; leave both unset to use the public archive node.",
    )
    parser.add_argument("--compress-chute-url", default=None, help="Deployed glyph-compressor chute base URL")
    parser.add_argument("--decompress-chute-url", default=None, help="Deployed glyph-decompressor chute base URL")
    parser.add_argument(
        "--docker-image",
        default=DEFAULT_VALIDATOR_DOCKER_IMAGE,
        help="Image used for --runner docker compress/decompress containers "
        f"(default: {DEFAULT_VALIDATOR_DOCKER_IMAGE!r}, built by scripts/install_deps.sh). "
        "Pre-pull/build it -- a cold pull runs inside the timed budget.",
    )
    parser.add_argument(
        "--docker-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --gpus to the docker containers and enforce the RTX 4090 gate (requires "
        "nvidia-container-toolkit on this host). Default ON: GPU execution is the network-wide "
        "default eval path now, so a validator without a matching GPU fails closed by design. "
        "Pass --no-docker-gpu for CPU-only codecs / testnet where that's not available.",
    )
    parser.add_argument(
        "--docker-gpu-device",
        default=None,
        metavar="ID",
        help="Specific GPU device id(s) for --docker-gpu, e.g. '0' or '0,1' (default: all visible GPUs).",
    )
    parser.add_argument(
        "--docker-seccomp-profile",
        default=None,
        metavar="PATH",
        help="Optional seccomp profile JSON passed to Docker codec containers. When omitted, "
        "Docker's default seccomp profile remains active.",
    )
    parser.add_argument(
        "--wandb.off",
        dest="wandb_off",
        action="store_true",
        help="Disable Weights & Biases logging (default: ON). Pure observability -- disabling "
        "it changes nothing about scoring/promotion/weights/burn.",
    )
    parser.add_argument("--wandb.project", dest="wandb_project", default="text-compression")
    parser.add_argument(
        "--wandb.entity", dest="wandb_entity", default="glyph-research-org",
        help="wandb entity (team/org). Defaults to the glyph-research-org team run.",
    )
    parser.add_argument(
        "--wandb.name", dest="wandb_name", default=None,
        help="Override the wandb run name. Defaults to this coldkey's on-chain identity name "
        "('btcli wallet set-identity'), or its hotkey ss58 if no identity is set, so multiple "
        "validators are distinguishable at a glance in the shared project.",
    )
    parser.add_argument(
        "--wandb.offline", dest="wandb_offline", action="store_true",
        help="Log locally only, no network (for tests/CI -- no WANDB_API_KEY needed).",
    )
    parser.add_argument("--wandb.notes", dest="wandb_notes", default=None)
    parser.add_argument(
        "--wandb.restart_interval", dest="wandb_restart_interval", type=float, default=24.0,
        metavar="HOURS",
        help="Finish and reopen the wandb run after this many hours, so long-lived validator "
        "processes don't accumulate one unbounded run. 0 disables the restart.",
    )
    # Per-source eval (issue #10): score on EVAL_STREAMS windows per source.
    parser.add_argument("--eval-source", default=EVAL_SOURCE,
                        help="comma-separated provenance sources to score (must name at least one)")
    parser.add_argument("--eval-streams", type=int, default=EVAL_STREAMS)
    parser.add_argument("--eval-stream-bytes", type=int, default=EVAL_STREAM_BYTES)
    parser.add_argument("--eval-benchmark-source", default=EVAL_BENCHMARK_SOURCE,
                        help="provenance source to run for benchmark display only")
    parser.add_argument("--eval-benchmark-streams", type=int, default=EVAL_BENCHMARK_STREAMS)
    parser.add_argument("--floor-bps", type=float, default=THROUGHPUT_FLOOR_BPS)
    parser.add_argument("--compress-budget-secs", type=float, default=COMPRESS_BUDGET_SECS)
    parser.add_argument("--max-artifact-bytes", type=int, default=DEFAULT_MAX_ARTIFACT_BYTES)
    parser.add_argument("--win-margin", type=float, default=DEFAULT_WIN_MARGIN)
    parser.add_argument(
        "--baseline-level",
        type=int,
        default=BASELINE_LEVEL,
        help="zstd level for the vacant-crown baseline floor (production default 19)",
    )
    parser.add_argument("--burn-uid", type=int, default=BURN_UID)
    parser.add_argument("--window-anchor", type=int, default=WINDOW_ANCHOR_BLOCK)
    parser.add_argument(
        "--loop", action="store_true",
        help="Deprecated, no-op: continuous looping is now the default (issue #79). Kept only "
        "so an existing invocation that already passes --loop doesn't break. Use --once to run "
        "a single round and exit instead.",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single round and exit, instead of looping continuously (the default). For "
        "testing/CI -- a real validator should not pass this.",
    )
    parser.add_argument("--sleep", type=int, default=1200)
    parser.add_argument(
        "--commit-poll-interval",
        type=int,
        default=COMMIT_POLL_INTERVAL_SECS,
        help="Seconds between lightweight commitment polls between full rounds, so commit-phase "
        "blocks are observed before their next-block reveal (exploit vector #9).",
    )
    parser.add_argument(
        "--commit-phase-max-age",
        type=int,
        default=COMMIT_PHASE_MAX_AGE_BLOCKS,
        help="Prune commit-phase digests with no matching reveal after this many blocks.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not submit weights")
    parser.add_argument("--offline-demo", action="store_true")
    parser.add_argument(
        "--local-codec",
        action="append",
        default=[],
        metavar="HOTKEY=PATH",
        help="Offline-demo codec, e.g. hkA=./reference_codec (repeatable)",
    )
    parser.add_argument(
        "--local-artifact",
        action="append",
        default=[],
        metavar="HOTKEY=PATH",
        help="Evaluate this on-chain hotkey's codec from a local dir instead of HuggingFace "
        "(testnet/CI when HF upload is unavailable). Repeatable.",
    )
    parser.add_argument(
        "--only-hotkeys",
        action="append",
        default=[],
        metavar="HOTKEY",
        help="Restrict evaluation to these hotkeys (ignore all other on-chain commitments). Repeatable.",
    )
    add_logging_args(parser)
    return parser


def _parse_local_artifacts(args) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in getattr(args, "local_artifact", []) or []:
        hotkey, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--local-artifact must be HOTKEY=PATH, got {item!r}")
        mapping[hotkey] = path
    return mapping


# Small, testable helpers


def _load_salt(state_dir: Path, salt_file: str | None) -> str:
    """Resolve this round's private seed salt.

    Default path (no --salt-file): a FRESH salt every call (issue #116). _load_salt runs at
    the top of every run_once, so this makes both the public (block hash) and private (salt)
    seed components fresh each round -- previously the salt was generated once and reused for
    the validator's entire lifetime, so a single leak of the on-disk value exposed every
    future round's sampling. The file is still written each time, but purely for
    after-the-fact observability (which salt produced a given round's seed) -- it is never
    read back.

    An explicit --salt-file keeps the original read-and-reuse behavior unchanged: that flag
    exists for tests/manual reproducibility, where a fixed, known salt is the point.
    """

    import secrets

    if salt_file:
        path = Path(salt_file)
        if path.exists():
            return path.read_text().strip()
        path.parent.mkdir(parents=True, exist_ok=True)
        salt = secrets.token_hex(32)
        path.write_text(salt)
        path.chmod(0o600)
        return salt

    path = state_dir / "validator_salt.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_hex(32)
    path.write_text(salt)
    path.chmod(0o600)
    return salt


def _local_version_key() -> int:
    return local_version_key()


def _assert_version_key_matches(chain: BittensorChain) -> int:
    return assert_weights_version_matches(chain)


def _apply_precheck(
    state: ValidatorState,
    parsed_commitments,
    max_artifact_bytes: int,
    *,
    block: int | None = None,
    local_artifacts: dict[str, str] | None = None,
) -> None:
    """Precheck commitments, record validity, and disqualify duplicate artifact hashes.

    ``local_artifacts`` maps hotkey -> local codec dir, used when HuggingFace download is
    unavailable (testnet/CI): the on-chain commitment is still authoritative, but the
    artifact is validated/evaluated from disk.
    """

    local_artifacts = local_artifacts or {}
    persisted_owner = dict(state.duplicate_hash_owner)
    entries: dict[str, CommitmentState] = {}
    for parsed in sorted(parsed_commitments, key=lambda p: p.hotkey):
        if parsed.hotkey in state.excluded_hotkeys:
            continue
        key = f"{parsed.hotkey}:{parsed.commitment.key}"
        existing = state.commitments.get(key)
        last_full_check = existing.last_full_check_block if existing else None
        # issue #96: don't let the full security scan + hash stay skipped forever just
        # because a commitment already has a recorded hash -- periodically force a fresh one
        # (also covers state persisted before this field existed, where last_full_check is
        # always None) as a second, independent safety net alongside revision immutability.
        full_check = (
            existing is None
            or not existing.artifact_hash
            or last_full_check is None
            or (block is not None and block - last_full_check >= PRECHECK_FULL_RECHECK_INTERVAL_BLOCKS)
        )
        local_dir = local_artifacts.get(parsed.hotkey)
        if local_dir:
            result = precheck_artifact_dir(
                local_dir, parsed.commitment.repo, parsed.commitment.rev,
                max_artifact_bytes=max_artifact_bytes,
            )
        else:
            result = precheck_codec(
                parsed.commitment.repo,
                parsed.commitment.rev,
                max_artifact_bytes=max_artifact_bytes,
                download=full_check,
            )
        # Tie-break block: prefer the commit-phase block this reveal opens (front-run-proof),
        # then a previously recorded block, then the current block (legacy / commit phase missed).
        seen_digests = state.commit_phase_seen.get(parsed.hotkey, {})
        if parsed.digest and parsed.digest in seen_digests:
            commit_block = seen_digests[parsed.digest]
            # Reveal resolved -> drop the commit-phase record so the map stays bounded (#21).
            del seen_digests[parsed.digest]
            if not seen_digests:
                state.commit_phase_seen.pop(parsed.hotkey, None)
        elif existing and existing.block is not None:
            commit_block = existing.block
        else:
            commit_block = block
        artifact_hash = result.artifact_hash or (existing.artifact_hash if existing else None)
        codec_desc = f"{parsed.hotkey} {parsed.commitment.repo}@{parsed.commitment.rev}"
        if result.ok:
            bt_logging.info(f"precheck: {codec_desc} valid")
        else:
            bt_logging.warning(f"precheck: {codec_desc} invalid: {'; '.join(result.errors)}")
        # issue #128: count only definitive 404s (repo deleted/renamed/private) toward the
        # exclusion streak; any success or differently-failing precheck (transient network/
        # 5xx/rate-limit) resets it, so an honest miner is never blacklisted over an outage.
        prior_streak = existing.consecutive_repo_not_found if existing else 0
        repo_not_found_streak = prior_streak + 1 if getattr(result, "repo_not_found", False) else 0
        if repo_not_found_streak >= REPO_NOT_FOUND_EXCLUDE_STREAK:
            state.excluded_hotkeys.add(parsed.hotkey)
            bt_logging.warning(
                f"precheck: {codec_desc} repo 404'd on {repo_not_found_streak} consecutive "
                f"prechecks (threshold {REPO_NOT_FOUND_EXCLUDE_STREAK}, spanning several "
                f"hours of rounds) -- treating as permanently unavailable and excluding "
                f"the hotkey; it will not be rechecked"
            )
        entries[key] = CommitmentState(
            hotkey=parsed.hotkey,
            repo=parsed.commitment.repo,
            revision=parsed.commitment.rev,
            block=commit_block,
            artifact_hash=artifact_hash,
            artifact_bytes=result.artifact_bytes
            if result.artifact_bytes is not None
            else (existing.artifact_bytes if existing else None),
            valid=result.ok,
            disqualification_reason=None if result.ok else "; ".join(result.errors),
            local_path=local_dir,
            last_full_check_block=block if full_check else last_full_check,
            consecutive_repo_not_found=repo_not_found_streak,
            # A fetch failure not (yet) confirmed permanent -- includes 404s still under
            # the #128 streak -- must not cost the hotkey its crown (issue #135).
            transiently_unreachable=getattr(result, "repo_unreachable", False)
            and repo_not_found_streak < REPO_NOT_FOUND_EXCLUDE_STREAK,
        )

    # Duplicate-artifact ownership: earliest commit_block wins, hotkey only as the final
    # deterministic tie-break for genuinely equal blocks -- NOT hotkey sort order (issue #58).
    # A copier who re-hosts a victim's artifact byte-for-byte and reveals in the same round
    # must not win ownership merely by having a lexicographically-earlier hotkey. A hash's
    # owner, once decided (this round or a previous one), still stays sticky thereafter.
    by_hash: dict[str, list[CommitmentState]] = {}
    for entry in entries.values():
        if entry.valid and entry.artifact_hash:
            by_hash.setdefault(entry.artifact_hash, []).append(entry)
    for artifact_hash, candidates in by_hash.items():
        owner = persisted_owner.get(artifact_hash)
        if owner is None:
            best = min(candidates, key=lambda e: (e.block if e.block is not None else float("inf"), e.hotkey))
            owner = best.hotkey
            persisted_owner[artifact_hash] = owner
        for entry in candidates:
            if entry.hotkey != owner:
                entry.valid = False
                entry.disqualification_reason = f"duplicate artifact; first owner is {owner}"

    state.commitments.update(entries)
    state.duplicate_hash_owner = persisted_owner


def _make_runner(args) -> "LocalSubprocessRunner | ChutesRunner | DockerRunner":
    if args.runner == "local":
        if getattr(args, "unsafe_local_no_sandbox", False):
            # This path is only ever reached while evaluating real on-chain commitments
            # (_evaluate_round -> chain.get_all_commitments()) -- unlike run_offline_demo,
            # which never touches the chain and stays unsandboxed for testing your OWN
            # codec. There is no "safe" way to honor this flag here, so refuse outright
            # (issue #56) rather than silently running an untrusted commitment unisolated.
            raise SystemExit(
                "--unsafe-local-no-sandbox refused: --runner local always evaluates real "
                "on-chain commitments here, never your own local codec. Use "
                "--offline-demo --local-codec instead to test your own codec unsandboxed."
            )
        # Network-wide default, not a per-operator override: an untrusted miner codec must
        # never run as the validator's own OS user with full network + wallet-key access
        # (issue #56). Fails closed (RunnerError) if `unshare` is unavailable rather than
        # silently running unisolated -- see require_network_isolation in eval/runner.py.
        return LocalSubprocessRunner(strict_sandbox=True, require_network_isolation=True)
    if args.runner == "docker":
        from eval.runner_docker import DockerRunner

        return DockerRunner(
            image=args.docker_image,
            gpu=args.docker_gpu,
            gpu_device=args.docker_gpu_device,
            seccomp_profile=args.docker_seccomp_profile,
        )
    from eval.runner_chutes import ChutesRunner

    return ChutesRunner(
        reference_sku=args.reference_sku,
        key_file=args.chutes_key_file,
        compress_base_url=args.compress_chute_url,
        decompress_base_url=args.decompress_chute_url,
    )


def _resolve_demo_corpus_dir(corpus_dir: str | None) -> Path:
    path = Path(corpus_dir) if corpus_dir else DEFAULT_DEMO_CORPUS_DIR
    if not path.is_dir():
        if corpus_dir:
            raise SystemExit(f"corpus directory not found: {path}")
        raise SystemExit(
            f"default demo corpus directory not found: {path}. Pass --corpus-dir explicitly."
        )
    return path


def _make_demo_provider(args) -> StaticLocalProvider:
    """--offline-demo's corpus provider: a local sample directory, never the chain/HF.

    Real rounds never call this -- see ``_evaluate_round``'s live-corpus wiring (issue #71).
    """

    path = _resolve_demo_corpus_dir(getattr(args, "corpus_dir", None))
    provider = StaticLocalProvider(path)
    if provider.total_bytes <= 0:
        raise SystemExit(f"corpus directory has no benchmark data files: {path}")
    return provider


def _parse_sources(value: str) -> list[tuple[str, int | None]]:
    """Parse an --eval-source list into ``(name, streams)`` pairs.

    Each entry is ``name`` or ``name:count`` -- the optional count sets that source's scored
    stream count (issue #112's asymmetric 2x fineweb-edu / 1x pile mix); a bare name yields
    ``None`` and the caller falls back to --eval-streams.
    """

    sources: list[tuple[str, int | None]] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, count = part.partition(":")
        name = name.strip()
        if not count:
            sources.append((name, None))
            continue
        try:
            streams = int(count)
        except ValueError:
            raise SystemExit(f"--eval-source entry {part!r}: stream count must be an integer") from None
        if streams <= 0:
            raise SystemExit(f"--eval-source entry {part!r}: stream count must be positive")
        sources.append((name, streams))
    return sources


def _source_seed(seed: int, source: str) -> int:
    payload = int(seed).to_bytes(8, "big") + source.encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _source_specs(args, provider, seed: int, source: str, *, scored: bool, streams: int):
    if not hasattr(provider, "source_range"):
        raise SystemExit("per-source eval requires a provider with source_range()")
    rng = provider.source_range(source)
    if rng is None:
        raise SystemExit(f"corpus provenance does not contain a contiguous source range for {source!r}")
    start, span = rng
    return sample_source_streams(
        _source_seed(seed, source),
        start,
        span,
        stream_bytes=args.eval_stream_bytes,
        streams=streams,
        source=source,
        scored=scored,
    )


def _select_specs(args, provider, seed):
    """Select the scored fineweb-edu/pile streams plus benchmark-only enwik9 streams."""

    sources = _parse_sources(getattr(args, "eval_source", "") or "")
    if not sources:
        raise SystemExit(
            "--eval-source must name at least one provenance source (e.g. 'fineweb-edu:2,pile:1')"
        )

    specs = []
    for scored_source, source_streams in sources:
        specs.extend(
            _source_specs(
                args, provider, seed, scored_source, scored=True,
                streams=source_streams if source_streams is not None else args.eval_streams,
            )
        )
    benchmark_source = (getattr(args, "eval_benchmark_source", "") or "").strip()
    benchmark_streams = getattr(args, "eval_benchmark_streams", 0)
    if benchmark_source and benchmark_streams > 0:
        specs.extend(
            _source_specs(args, provider, seed, benchmark_source, scored=False, streams=benchmark_streams)
        )
    return specs


def _scored_specs(specs):
    return [spec for spec in specs if spec.scored]


def resolve_blockmachine_key(args) -> str | None:
    """Optional blockmachine.io archive key (issue #151): ``--blockmachine-key-file`` wins,
    else the ``BLOCKMACHINE_API_KEY`` env var (deployment-specific, ``.env.example``).
    Absent -> the public archive node, today's behavior."""

    path = getattr(args, "blockmachine_key_file", None)
    if path:
        return Path(path).read_text().strip() or None
    return os.environ.get("BLOCKMACHINE_API_KEY") or None


def _make_chain(args) -> BittensorChain:
    return BittensorChain(
        ChainConfig(
            netuid=args.netuid,
            network=args.network,
            wallet_name=args.wallet_name,
            hotkey_name=args.hotkey_name,
            wallet_path=args.wallet_path,
            blockmachine_api_key=resolve_blockmachine_key(args),
        )
    )


def _update_conviction_ledger(state: ValidatorState, chain: BittensorChain, block: int, tempo: int) -> None:
    """Advance the persisted earnings ledger to ``block`` (Miner Conviction, issue #141).

    One increment code path (``ledger_catchup``); grid blocks within the live node's
    pruning horizon come from it, anything older (validator downtime, fresh start)
    backfills from the archive endpoint. Best-effort: a failure leaves the ledger at the
    last fully-applied grid block -- gating then runs on slightly-stale totals for a tempo
    rather than blocking weight-setting.
    """

    def emissions_at(grid_block: int) -> dict[str, float]:
        if block - grid_block <= 250:
            return chain.emissions_by_hotkey(grid_block)
        return chain.archive_emissions_by_hotkey(grid_block)

    try:
        # Backfill visibility (issue #154): a fresh sync is 40+ minutes of otherwise total
        # silence on the public archive node -- announce it (preferred endpoint first, key
        # never logged; per-query fallbacks already warn via #152) and log each 20% of the
        # grid. The normal 1-sample steady-state catchup stays quiet.
        grid = ledger_grid(state.conviction_ledger.last_block, block, tempo)
        started = time.monotonic()
        on_applied = None
        if len(grid) > 1:
            key = getattr(getattr(chain, "config", None), "blockmachine_api_key", None)
            source = "blockmachine RPC" if key else "public archive node"
            from_block = max(state.conviction_ledger.last_block, CONVICTION_TRACKING_START_BLOCK)
            bt_logging.info(
                f"conviction: backfilling ledger from block {from_block:,} to {grid[-1]:,} "
                f"({len(grid)} tempo samples) via {source}"
            )

            def on_applied(done: int, total: int, grid_block: int) -> None:
                fifths_now, fifths_before = done * 5 // total, (done - 1) * 5 // total
                if fifths_now > fifths_before:
                    bt_logging.info(
                        f"conviction: backfill {fifths_now * 20}% ({done}/{total} samples, "
                        f"at block {grid_block:,}, elapsed {int(time.monotonic() - started)}s)"
                    )

        applied = ledger_catchup(
            state.conviction_ledger, current_block=block, tempo=tempo,
            emissions_at=emissions_at, on_applied=on_applied,
        )
        if applied:
            bt_logging.info(
                f"conviction: ledger advanced {applied} tempo(s) to block "
                f"{state.conviction_ledger.last_block}"
            )
    except Exception as exc:  # noqa: BLE001 - best-effort; never block weight-setting
        bt_logging.warning(
            f"conviction: ledger catchup stopped at block "
            f"{state.conviction_ledger.last_block}, will resume next tempo: {exc}"
        )


def _conviction_report_for_winners(
    state: ValidatorState, metagraph, block: int, chain: BittensorChain | None = None
) -> dict[str, dict]:
    """Compliance snapshot for the current winner slots, logged every weight-setting."""

    winners = [w.hotkey for w in state.winner_history[:WINNER_LIMIT]]
    # The lock is alpha-only by design: on dTAO metagraphs S is the consensus stake weight
    # (alpha + tao-weighted root stake), so a winner could otherwise satisfy part of the
    # lock with root TAO. Prefer the pure per-hotkey alpha; S only as a fallback.
    stakes = getattr(metagraph, "alpha_stake", None)
    if stakes is None:
        stakes = getattr(metagraph, "S", None)
    staked = (
        {hotkey: float(s) for hotkey, s in zip(metagraph.hotkeys, stakes)}
        if stakes is not None
        else {}
    )
    # Conviction v1.1 (issue #156): read the winners' chain-locked alpha whenever the
    # chain supports it -- also before CONVICTION_LOCK_CHECK_START_BLOCK, so operators
    # can watch winners lock up during the grace window. Best-effort: on failure the
    # report falls back to the v1 staked rule for this tempo (locked <= staked always,
    # so a lock-compliant winner can never be wrongly gated by the fallback).
    locked = None
    read_locked = getattr(chain, "locked_alpha_by_hotkey", None)
    if winners and read_locked is not None:
        try:
            locked = read_locked(winners)
        except Exception as exc:  # noqa: BLE001 - never block weight-setting
            bt_logging.warning(
                f"conviction: lock query failed -- gating on staked alpha this tempo "
                f"(v1 rule): {exc}"
            )
    report = conviction_report(
        state.conviction_ledger, winners, staked, block=block, conviction_by_hotkey=locked
    )
    for hotkey, entry in report.items():
        conviction_part = (
            f" conviction={entry['conviction']:.1f}" if entry["conviction"] is not None else ""
        )
        line = (
            f"conviction: {hotkey} earned={entry['earned']:.1f} staked={entry['staked']:.1f}"
            f"{conviction_part} required_conviction={entry['required_conviction']:.1f} "
            f"compliant={entry['compliant']}"
        )
        if entry["compliant"]:
            bt_logging.info(line)
        else:
            bt_logging.warning(
                f"{line} -- winner is below its required conviction; its share goes to the "
                f"other winner slot this tempo (or burns if no slot meets its conviction) "
                f"and restores automatically once its conviction covers the requirement "
                f"(`btcli lock add --netuid 117`, issues #141/#166)"
            )
    return report


def _publish_winner_commitment(chain: BittensorChain, champion: WinnerEntry) -> None:
    """Best-effort, observability-only publish of the new champion on this validator's own
    commitment slot (issue #103) -- never read back into this validator's own scoring/
    promotion (see WinnerCommitment's docstring), and a publish failure must never crash or
    delay the round.
    """

    winner = WinnerCommitment(
        hotkey=champion.hotkey,
        ratio_ppm=round(champion.ratio * 1_000_000),
        scoring_version=SCORING_VERSION,
    )
    try:
        response = chain.set_commitment(serialize_winner_commitment(winner))
        if getattr(response, "success", True):
            bt_logging.info(f"published winner commitment: {champion.hotkey} ratio={champion.ratio:.4f}")
        else:
            bt_logging.warning(f"failed to publish winner commitment: {response}")
    except Exception as exc:
        bt_logging.warning(f"failed to publish winner commitment: {exc}")


def _wandb_identity_name(chain: BittensorChain) -> str | None:
    """Best-effort wandb run-name fallback (issue #102 follow-up): this validator's on-chain
    identity name if set (``btcli wallet set-identity``), else its hotkey ss58 -- still
    distinguishes validators at a glance in the shared project even though set-identity is
    opt-in and most hotkeys won't have one. An explicit --wandb.name always overrides this
    (see core.wandb_logger.make_wandb_logger)."""

    return chain.identity_name() or chain.hotkey


def _startup_wandb_identity_name(args) -> str | None:
    """Resolve the wandb run-name identity fallback via a throwaway chain connection before
    the wandb run starts (``main()`` only -- ``run_once``/``run_reign_only`` reuse the chain
    they build for the round itself). Unlike that per-round chain, a failure constructing
    this one must never crash/delay startup -- it's a nice-to-have label, and the real
    per-round chain (built fresh inside the retried round loop) surfaces a genuine
    wallet/network problem properly either way."""

    try:
        return _wandb_identity_name(_make_chain(args))
    except Exception:
        return None


def _append_live_benchmark_stream(args, provider, specs):
    """Append the latest complete live snapshot as a benchmark-only stream (issue #139).

    Read-only at round start: never fetches, only picks up what the between-rounds
    prefetch already completed. No snapshot yet (or no state dir, e.g. offline tests) ->
    the round proceeds without a live stream; a stale-but-complete snapshot is preferred
    over none. Returns the (possibly wrapped) provider and spec list.
    """

    state_dir = getattr(args, "state_dir", None)
    if not state_dir:
        return provider, specs
    try:
        store = LiveSnapshotStore(Path(state_dir) / "live_data")
        latest = store.latest()
        if latest is None:
            bt_logging.warning(
                "live benchmark: no complete snapshot yet; skipping the live stream this round"
            )
            return provider, specs
        path, snapshot_block = latest
        snapshot = path.read_bytes()
        spec = live_benchmark_spec(provider, snapshot)
        bt_logging.info(
            f"live benchmark: using snapshot from block {snapshot_block} "
            f"({len(snapshot):,} bytes) as benchmark-only stream {spec.stream_id}"
        )
        return SnapshotAppendedProvider(provider, snapshot), [*specs, spec]
    except Exception as exc:  # noqa: BLE001 - display-only; must never fail the round
        bt_logging.warning(f"live benchmark: skipping live stream: {exc}")
        return provider, specs


_live_prefetcher: LivePrefetcher | None = None


def _start_live_prefetch(args, chain: BittensorChain) -> None:
    """Kick off the between-rounds live-data fetch (issue #139) -- strictly best-effort,
    returns immediately; any failure here degrades to a skipped/stale live stream, never a
    blocked or delayed round."""

    global _live_prefetcher
    state_dir = getattr(args, "state_dir", None)
    if not state_dir:
        return
    try:
        if _live_prefetcher is None:
            _live_prefetcher = LivePrefetcher(LiveSnapshotStore(Path(state_dir) / "live_data"))
        _live_prefetcher.start(chain.current_block())
    except Exception as exc:  # noqa: BLE001
        bt_logging.warning(f"live benchmark: prefetch not started: {exc}")


def _recover_vacant_crown(state: ValidatorState) -> None:
    """Refill an empty ``winner_history`` from persisted scores (issue #135 recovery gap).

    Challenger selection is one-shot (``c.key not in state.scores``), so once every hotkey
    is already scored a vacant crown can never refill itself -- 100% burn indefinitely.
    Promote the best already-scored, currently-valid, non-excluded hotkey. In practice this
    re-crowns an ex-champion whose repo came back after an outage or whose pop later proved
    over-eager; one-shot losers stay out via ``excluded_hotkeys``, and stale-version scores
    never qualify (a score retained across a score-compatible transition, issue #143,
    counts as current -- its ratio means the same thing).
    """

    candidates = [
        score
        for key, score in state.scores.items()
        if score.valid
        and score_is_comparable(score)
        and score.hotkey not in state.excluded_hotkeys
        and (commitment := state.commitments.get(key)) is not None
        and commitment.valid
    ]
    if not candidates:
        return
    best = min(candidates, key=lambda score: rank_key(score.as_winner()))
    bt_logging.warning(
        f"vacant crown: re-promoting best already-scored hotkey {best.hotkey} "
        f"(ratio={best.ratio:.4f}) instead of burning indefinitely"
    )
    state.winner_history = promote_winner(state.winner_history, best.as_winner())


def _evaluate_round(
    args, state: ValidatorState, chain: BittensorChain, salt: str
) -> tuple[int, dict | None, dict[str, str], "object"]:
    """Precheck commitments and, if there are new challengers, run one reign round.

    Returns ``(block, round_metrics, raw_commitments, metagraph)``; ``round_metrics`` is
    ``None`` when no challengers ran this round (nothing new to report). ``round_metrics``
    is a plain dict built by ``core.wandb_logger.build_round_metrics`` purely from what this
    round already decided -- reporting it changes nothing about scoring/promotion (issue
    #41). ``raw_commitments`` is the commitment dict this round already fetched for
    precheck, passed back so ``run_once``'s burn-override check (issue #113) doesn't need a
    second, redundant ``get_all_commitments()`` round-trip moments later. ``metagraph`` is
    fetched here (same de-dup pattern, issue #126) both to label round metrics/logs with
    UIDs alongside hotkeys and for ``run_once``'s weight-setting to reuse -- this also
    deliberately gives ``run_reign_only`` (which never fetched one before) UID labeling in
    its round metrics.
    """

    if getattr(args, "corpus_dir", None):
        # --corpus-dir is offline-demo-only (see run_offline_demo / issue #71): a real round
        # always builds its corpus live from HuggingFace, keyed by this round's chain beacon,
        # so a leftover --corpus-dir would be silently ignored rather than doing anything --
        # refuse outright instead of letting an operator believe it took effect.
        raise SystemExit(
            "--corpus-dir is only valid with --offline-demo; a real round always builds its "
            "corpus live from HuggingFace (eval.live_corpus.resolve_live_corpus, issue #71)."
        )
    block = chain.current_block()
    if state.window_anchor_block is None:
        state.window_anchor_block = args.window_anchor or block
    raw_commitments = chain.get_all_commitments()
    metagraph = chain.metagraph()
    hotkey_to_uid = {hk: int(uid) for hk, uid in zip(metagraph.hotkeys, metagraph.uids)}
    # Record commit-phase digests as we see them so a later reveal can tie-break off the
    # commit-phase block (exploit vector #9). Observing this requires polling during the
    # commit/reveal window; a validator that only catches the reveal degrades to the
    # reveal-observation block, which a copier (who must reveal later) still loses to.
    for hotkey, digest in parse_commit_phase_by_hotkey(raw_commitments).items():
        state.commit_phase_seen.setdefault(hotkey, {}).setdefault(digest, block)
    prune_commit_phase_seen(
        state.commit_phase_seen, block, getattr(args, "commit_phase_max_age", COMMIT_PHASE_MAX_AGE_BLOCKS)
    )
    parsed = parse_commitments_by_hotkey(raw_commitments)
    only = set(getattr(args, "only_hotkeys", []) or [])
    if only:
        parsed = [p for p in parsed if p.hotkey in only]
    _apply_precheck(
        state, parsed, args.max_artifact_bytes, block=block,
        local_artifacts=_parse_local_artifacts(args),
    )
    # History retention uses retained_hotkeys, not eligible_hotkeys (issue #135): during
    # the 2026-07-16 HF outage, a 504 on the champion's precheck made it ineligible for one
    # round and this compaction permanently dethroned it into an indefinite 100% burn. A
    # crown entry now survives transient unreachability and is dropped only when the hotkey
    # is definitively out (confirmed-404 exclusion, content disqualification, or the #67
    # failed-re-eval pop inside run_round).
    retained = state.retained_hotkeys()
    state.winner_history = compact_history(state.winner_history, eligible_hotkeys=retained)
    if not state.winner_history:
        _recover_vacant_crown(state)

    challengers = [
        c
        for c in state.commitments.values()
        if c.valid and c.key not in state.scores and c.hotkey not in state.excluded_hotkeys
    ]
    round_metrics = None
    if challengers:
        seed = derive_seed(chain.block_hash(block), salt, block)
        # Each validator independently builds this round's corpus straight from HuggingFace,
        # keyed by the same beacon-derived seed used for stream-window sampling below -- no
        # shared file, no owner-run oracle process, and every validator lands on byte-identical
        # chunks by construction (issue #71).
        # HF_TOKEN is optional (anonymous streaming still works) but avoids intermittent
        # 403/AccessDenied from HF's Xet-backed CDN throttling fully anonymous traffic
        # (issue #108) -- a free-tier read token is enough, no special dataset permissions.
        provider = resolve_live_corpus(str(seed), token=os.environ.get("HF_TOKEN"))
        specs = _select_specs(args, provider, seed)
        provider, specs = _append_live_benchmark_stream(args, provider, specs)
        scored_specs = _scored_specs(specs)
        baseline = zstd_baseline_ratio(
            [provider.materialize(s) for s in scored_specs],
            level=args.baseline_level,
            sources=[s.source for s in scored_specs],
        )
        runner = _make_runner(args)
        caps = ResourceCaps(wall_clock_secs=args.compress_budget_secs, artifact_bytes=args.max_artifact_bytes)
        champion_before = state.winner_history[0].hotkey if state.winner_history else None
        # uid alongside hotkey (issue #126): following the log otherwise requires manually
        # cross-referencing the metagraph.
        def _with_uid(hotkey: str) -> str:
            return f"{hotkey} (uid {hotkey_to_uid.get(hotkey, '?')})"

        challenger_descs = [_with_uid(c.hotkey) for c in challengers]
        bt_logging.info(
            f"round: evaluating incumbent={_with_uid(champion_before) if champion_before else 'none'}, "
            f"{len(challengers)} challenger(s): {challenger_descs} (baseline zstd ratio={baseline:.4f})"
        )
        outcomes = run_round(
            state, runner, challengers, provider, specs,
            caps=caps, floor_bps=args.floor_bps, budget_secs=args.compress_budget_secs,
            margin=args.win_margin, block=block, eligible_hotkeys=retained, baseline_ratio=baseline,
        )
        champion_after = state.winner_history[0] if state.winner_history else None
        crown_changed = bool(champion_after) and champion_after.hotkey != champion_before
        if crown_changed:
            _publish_winner_commitment(chain, champion_after)
        round_metrics = build_round_metrics(
            block=block,
            baseline_ratio=baseline,
            num_challengers=len(challengers),
            outcomes=outcomes,
            excluded_hotkeys_count=len(state.excluded_hotkeys),
            commit_phase_seen_count=sum(len(v) for v in state.commit_phase_seen.values()),
            winner_hotkey=champion_after.hotkey if champion_after else None,
            winner_ratio=champion_after.ratio if champion_after else None,
            crown_changed=crown_changed,
            hotkey_to_uid=hotkey_to_uid,
        )
    return block, round_metrics, raw_commitments, metagraph


# Production paths (require chain access)


def run_once(args: argparse.Namespace, wandb_logger: WandbLogger | None = None) -> None:
    owns_logger = wandb_logger is None
    try:
        state_dir = Path(args.state_dir)
        state_path = state_dir / "validator_state.json"
        state = load_state(state_path)
        salt = _load_salt(state_dir, args.salt_file)

        chain = _make_chain(args)
        wandb_logger = wandb_logger or make_wandb_logger(args, identity_name=_wandb_identity_name(chain))
        version_key = _local_version_key()
        try:
            bt_logging.info(f"version key ok: {_assert_version_key_matches(chain)}")
        except SystemExit as exc:
            # A version-key mismatch is the expected, transient state during a release
            # rollout until the owner updates the on-chain weights_version hyperparameter --
            # it self-heals with time, unlike genuine misconfiguration. Skip this cycle
            # (nothing scored, no weights set -- same safety property as exiting) and let the
            # outer loop's normal sleep/retry cadence pick up once the chain catches up,
            # instead of letting SystemExit sail past main()'s `except Exception` and kill
            # the process into a pm2 crash-restart loop (issue #120). Scoped to the version
            # check only: any other SystemExit still hard-stops as before.
            bt_logging.warning(f"weights_version mismatch, waiting for on-chain update: {exc}")
            return
        if not chain.commit_reveal_enabled():
            bt_logging.warning("commit-reveal is not enabled on this subnet; anti-copy weights are weaker")

        block, round_metrics, raw_commitments, metagraph = _evaluate_round(args, state, chain, salt)
        if round_metrics is not None:
            wandb_logger.log(round_metrics)
        # Reuse the metagraph _evaluate_round already fetched (issue #126) -- same
        # no-second-round-trip pattern as raw_commitments (issue #113).
        hotkeys = list(metagraph.hotkeys)
        uids = [int(uid) for uid in metagraph.uids]
        hotkey_to_uid = dict(zip(hotkeys, uids))
        champion = state.winner_history[0] if state.winner_history else None
        champion_desc = (
            f"{champion.hotkey} (uid {hotkey_to_uid.get(champion.hotkey, '?')}) ratio={champion.ratio:.4f}"
            if champion
            else "none"
        )
        challengers_desc = (
            f"{round_metrics['round/num_challengers']} challenger(s)" if round_metrics else "0 challengers"
        )
        bt_logging.info(f"round: block={block} champion={champion_desc} {challengers_desc}")
        tempo = chain.tempo()
        anchor = state.window_anchor_block

        # Owner emergency burn override (issue #113), read from the commitment dict
        # _evaluate_round already fetched for precheck -- no second chain round-trip.
        force_burn = (
            resolve_force_burn(raw_commitments, hotkeys[args.burn_uid])
            if 0 <= args.burn_uid < len(hotkeys)
            else False
        )
        if force_burn:
            # Without this line, a forced burn is indistinguishable in the log from an
            # ordinary scheduled burn tempo -- except it happens EVERY tempo, which reads
            # as a bug unless the operator knows the owner override is active.
            bt_logging.warning(
                "Subnet faced an issue and turned into temporal burn "
                "(owner emergency override: on-chain force_burn=true) -- burning 100% this tempo"
            )

        # Miner Conviction (issues #141/#166): advance the earnings ledger, then gate any
        # winner below its required conviction -- its share goes to the compliant winner
        # slot(s), burning only when all occupied slots are gated.
        _update_conviction_ledger(state, chain, block, tempo)
        conviction = _conviction_report_for_winners(state, metagraph, block, chain)
        gated = {hotkey for hotkey, entry in conviction.items() if not entry["compliant"]}

        weights, burn = decide_weights(
            hotkeys, state.winner_history, block=block, tempo=tempo,
            last_round_outputs=state.last_round_outputs, anchor=anchor, burn_uid=args.burn_uid,
            force_burn=force_burn, gated_hotkeys=gated,
        )
        wandb_logger.log(
            build_weights_metrics(
                block=block, tempo=tempo, is_burn_tempo=burn, uids=uids, weights=weights,
                conviction=conviction,
            )
        )
        save_state(state_path, state)

        nonzero = [(uids[i], round(w, 4)) for i, w in enumerate(weights) if w > 0]
        bt_logging.info(f"block={block} tempo={tempo} burn_tempo={burn} weights={nonzero}")
        if args.dry_run:
            bt_logging.info("dry-run: not submitting weights")
            return
        remaining = chain.blocks_until_weights_allowed()
        if remaining and remaining > 0:
            bt_logging.info(f"set_weights: skipped, rate-limited ({remaining} blocks remaining)")
            return
        response = chain.set_weights(uids, weights, version_key=version_key)
        if response.success:
            bt_logging.info("set_weights: success=True")
        else:
            bt_logging.warning(f"set_weights: success=False error={response.error} message={response.message}")
    finally:
        # wandb_logger may still be None here if an exception struck before chain was built
        # (it's now created after chain, to resolve the run-name identity fallback from it).
        if owns_logger and wandb_logger is not None:
            wandb_logger.finish()


def run_reign_only(args: argparse.Namespace, wandb_logger: WandbLogger | None = None) -> None:
    """The reign half only (evaluate + update crown), no weight setting."""

    owns_logger = wandb_logger is None
    try:
        state_dir = Path(args.state_dir)
        state_path = state_dir / "validator_state.json"
        state = load_state(state_path)
        salt = _load_salt(state_dir, args.salt_file)

        chain = _make_chain(args)
        wandb_logger = wandb_logger or make_wandb_logger(args, identity_name=_wandb_identity_name(chain))
        try:
            bt_logging.info(f"version key ok: {_assert_version_key_matches(chain)}")
        except SystemExit as exc:
            # Same as run_once (issue #120): a release-rollout mismatch self-heals once the
            # on-chain weights_version updates -- skip the cycle, don't kill the process.
            bt_logging.warning(f"weights_version mismatch, waiting for on-chain update: {exc}")
            return
        _block, round_metrics, _raw_commitments, _metagraph = _evaluate_round(args, state, chain, salt)
        if round_metrics is not None:
            wandb_logger.log(round_metrics)
        # Keep the conviction ledger warm in the split deployment too (issue #141): the
        # standalone weight-setter reads state but never writes it, so the reign worker is
        # the split deployment's ledger persister.
        _update_conviction_ledger(state, chain, _block, chain.tempo())
        save_state(state_path, state)
        bt_logging.info(f"winner history = {[(w.hotkey, round(w.ratio, 4)) for w in state.winner_history]}")
    finally:
        # wandb_logger may still be None if an exception struck before chain was built.
        if owns_logger and wandb_logger is not None:
            wandb_logger.finish()


# Offline M0 demo (no chain): eval -> score -> king-of-the-hill -> weights end to end


def run_offline_demo(args: argparse.Namespace) -> None:
    codec_specs = args.local_codec or ["winner=./reference_codec"]
    provider = _make_demo_provider(args)
    seed = derive_seed("offline-demo-block", "offline-demo-salt", 0)
    specs = _select_specs(args, provider, seed)
    scored_specs = _scored_specs(specs)
    baseline = zstd_baseline_ratio(
        [provider.materialize(s) for s in scored_specs],
        level=args.baseline_level,
        sources=[s.source for s in scored_specs],
    )

    hotkeys = ["uid0_burn"]
    artifacts: list[tuple[str, ArtifactRef]] = []
    parsed_codecs: list[tuple[str, str]] = []
    for item in codec_specs:
        hotkey, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--local-codec must be HOTKEY=PATH, got {item!r}")
        hotkeys.append(hotkey)
        artifacts.append((hotkey, ArtifactRef(repo=f"{hotkey}/codec", rev="local", local_path=path)))
        parsed_codecs.append((hotkey, path))

    runner = LocalSubprocessRunner()
    caps = ResourceCaps(wall_clock_secs=args.compress_budget_secs)
    outcomes = paired_eval(
        runner, artifacts, provider, specs, caps=caps,
        floor_bps=args.floor_bps, budget_secs=args.compress_budget_secs,
    )

    scored_counts = {}
    benchmark_counts = {}
    for spec in specs:
        counts = scored_counts if spec.scored else benchmark_counts
        counts[spec.source or "whole-corpus"] = counts.get(spec.source or "whole-corpus", 0) + 1
    sampled_bytes = specs[0].length if specs else 0
    bt_logging.info(
        f"streams={len(specs)} stream_bytes={sampled_bytes}"
        f" scored={scored_counts}"
        f" benchmark_only={benchmark_counts} "
        f"baseline zstd-{args.baseline_level} ratio={baseline:.4f}"
    )
    history: list[WinnerEntry] = []
    for block_n, (hotkey, _path) in enumerate(parsed_codecs):
        outcome = outcomes[hotkey]
        beats = outcome.score.valid and outcome.score.ratio < baseline
        status = "valid" if outcome.score.valid else f"INVALID {outcome.score.reasons}"
        scored_breakdown = source_ratio_breakdown(outcome.results)
        benchmark = {
            result.source or result.stream_id: stream_ratio(result)
            for result in outcome.results
            if not result.scored
        }
        bt_logging.info(
            f"  {hotkey}: ratio={outcome.score.ratio:.4f} "
            f"scored_sources={{{', '.join(f'{k}: {v:.4f}' for k, v in scored_breakdown.items())}}} "
            f"benchmark_only={{{', '.join(f'{k}: {v:.4f}' for k, v in benchmark.items())}}} "
            f"min_throughput={outcome.score.throughput_bps_min:.0f} B/s "
            f"beats_baseline={beats} [{status}]"
        )
        if beats and (not history or should_promote(outcome.score.ratio, history[0].ratio, args.win_margin)):
            winner = WinnerEntry(hotkey, f"{hotkey}/codec", "local", outcome.score.ratio, block_n)
            history = promote_winner(history, winner)

    last_round_outputs = outcomes[history[0].hotkey].burn_outputs() if history else []
    bt_logging.info(f"winner history = {[(w.hotkey, round(w.ratio, 4)) for w in history]}")
    from core.constants import BURN_WINDOW_TEMPOS

    bt_logging.info(f"temporal burn schedule (two {BURN_WINDOW_TEMPOS}-tempo windows):")
    for tempo_idx in range(2 * BURN_WINDOW_TEMPOS):
        block = tempo_idx * 360
        weights, burn = decide_weights(
            hotkeys, history, block=block, tempo=360, last_round_outputs=last_round_outputs, anchor=0
        )
        nonzero = [(hotkeys[i], round(w, 3)) for i, w in enumerate(weights) if w > 0]
        bt_logging.info(f"  tempo {tempo_idx} (block {block}): burn={burn} weights={nonzero}")


def poll_commit_phase(args: argparse.Namespace, chain: BittensorChain) -> int:
    """Lightweight commitment poll: record new commit-phase digests, prune stale ones.

    Run between full rounds at ~block cadence (full eval rounds are far too slow to catch
    the ~1-block commit->reveal window) so a reveal tie-breaks off its true commit-phase
    block (#21, exploit vector #9). Returns the count of newly recorded digests.
    """

    state_path = Path(args.state_dir) / "validator_state.json"
    state = load_state(state_path)
    block = chain.current_block()
    new = 0
    for hotkey, digest in parse_commit_phase_by_hotkey(chain.get_all_commitments()).items():
        seen = state.commit_phase_seen.setdefault(hotkey, {})
        if digest not in seen:
            seen[digest] = block
            new += 1
    prune_commit_phase_seen(state.commit_phase_seen, block, args.commit_phase_max_age)
    save_state(state_path, state)
    return new


def _sleep_with_commit_polls(args: argparse.Namespace, chain: BittensorChain) -> None:
    """Wait ~``args.sleep`` until the next full round, polling commit phases each block."""

    end = time.time() + args.sleep
    while time.time() < end:
        time.sleep(max(1, min(args.commit_poll_interval, end - time.time())))
        try:
            poll_commit_phase(args, chain)
        except Exception:
            bt_logging.exception("commit-phase poll failed")


def main() -> None:
    from core.dotenv import load_dotenv
    from core.log_config import configure_logging

    load_dotenv()  # CHUTES_API_KEY etc. from .env (see .env.example)
    args = build_parser().parse_args()
    configure_logging(args)
    if args.offline_demo:
        run_offline_demo(args)
        return
    # One wandb run for the whole process lifetime (not one per --loop iteration) so console
    # capture starts before the first round and metrics from every round land in the same run
    # (see core.wandb_logger's own restart_interval for keeping very long runs bounded).
    wandb_logger = make_wandb_logger(args, identity_name=_startup_wandb_identity_name(args))
    poll_chain = None
    try:
        while True:
            try:
                run_once(args, wandb_logger=wandb_logger)
            except KeyboardInterrupt:
                break
            except Exception:
                bt_logging.exception("round failed")
            if args.once:
                break
            # Poll at block cadence between rounds so a reveal's commit-phase block is captured
            # rather than degrading to the reveal-observation block (exploit vector #9).
            if poll_chain is None:
                poll_chain = _make_chain(args)
            # The between-rounds window is where the next round's live benchmark snapshot
            # gets fetched (issue #139) -- non-blocking, so the commit-poll sleep below
            # still starts immediately.
            _start_live_prefetch(args, poll_chain)
            _sleep_with_commit_polls(args, poll_chain)
    finally:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
