"""Top-level Glyph validator orchestrator.

Composes the reign worker (evaluate + king-of-the-hill) and the weight setter
(temporal-burn weights) in one process. The same pieces can run as separate PM2 services
(``reign_worker``, ``weight_setter``); this orchestrator is the all-in-one path and the
offline M0 demo.
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from pathlib import Path

from chain.chain import BittensorChain, ChainConfig
from core.commitments import (
    parse_commit_phase_by_hotkey,
    parse_commitments_by_hotkey,
    prune_commit_phase_seen,
)
from core.constants import (
    BASELINE_LEVEL,
    BURN_UID,
    COMMIT_PHASE_MAX_AGE_BLOCKS,
    COMMIT_POLL_INTERVAL_SECS,
    COMPRESS_BUDGET_SECS,
    DEFAULT_MAX_ARTIFACT_BYTES,
    DEFAULT_NETUID,
    DEFAULT_WIN_MARGIN,
    REFERENCE_SKU,
    STREAM_BYTES,
    STREAMS_PER_ROUND,
    THROUGHPUT_FLOOR_BPS,
    WINDOW_ANCHOR_BLOCK,
)
from core.state import CommitmentState, ValidatorState, load_state, save_state
from core.version import assert_weights_version_matches, local_version_key
from core.weights import WinnerEntry, compact_history, promote_winner, should_promote
from eval.corpus import StaticLocalProvider
from eval.evaluator import paired_eval
from eval.runner import ArtifactRef, LocalSubprocessRunner, ResourceCaps
from eval.scoring import zstd_baseline_ratio
from eval.streams import derive_seed, sample_streams
from validation.precheck import precheck_artifact_dir, precheck_codec
from reign_worker.service import run_round
from weight_setter.service import decide_weights

__all__ = ["build_parser", "run_once", "run_reign_only", "run_offline_demo", "decide_weights", "main"]

DEFAULT_MIXED_CORPUS_DIR = "/tmp/glyph_mixed_8x2mb"
MIXED_CORPUS_ENV = "GLYPH_MIXED_CORPUS_DIR"


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Glyph compression validator")
    parser.add_argument("--network", default="finney")
    parser.add_argument("--netuid", type=int, default=DEFAULT_NETUID)
    parser.add_argument("--wallet-name", "--wallet.name", dest="wallet_name", default="default")
    parser.add_argument("--hotkey-name", "--wallet.hotkey", dest="hotkey_name", default="default")
    parser.add_argument("--wallet-path", "--wallet.path", dest="wallet_path", default=None)
    parser.add_argument("--state-dir", default="./state")
    parser.add_argument("--salt-file", default=None)
    parser.add_argument("--runner", choices=["local", "chutes"], default="chutes")
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help=(
            "StaticLocalProvider corpus directory; defaults to the mixed launch corpus "
            f"at ${MIXED_CORPUS_ENV} or {DEFAULT_MIXED_CORPUS_DIR}"
        ),
    )
    parser.add_argument(
        "--corpus-url",
        default=None,
        help="Public URL of the corpus served as one contiguous blob (chunk order == sorted "
        "manifest order). Enables the production Chutes path: the runner range-fetches each "
        "stream itself instead of the validator inlining the 256 MiB sample.",
    )
    parser.add_argument("--reference-sku", default=REFERENCE_SKU)
    parser.add_argument("--chutes-key-file", default=None)
    parser.add_argument("--compress-chute-url", default=None, help="Deployed glyph-compressor chute base URL")
    parser.add_argument("--decompress-chute-url", default=None, help="Deployed glyph-decompressor chute base URL")
    parser.add_argument("--streams", type=int, default=STREAMS_PER_ROUND)
    parser.add_argument("--stream-bytes", type=int, default=STREAM_BYTES)
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
    parser.add_argument("--loop", action="store_true")
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
    return parser


def _parse_local_artifacts(args) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in getattr(args, "local_artifact", []) or []:
        hotkey, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--local-artifact must be HOTKEY=PATH, got {item!r}")
        mapping[hotkey] = path
    return mapping


# --------------------------------------------------------------------------------------
# Small, testable helpers
# --------------------------------------------------------------------------------------


def _load_salt(state_dir: Path, salt_file: str | None) -> str:
    path = Path(salt_file) if salt_file else state_dir / "validator_salt.txt"
    if path.exists():
        return path.read_text().strip()
    import secrets

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
    valid_hash_owner = dict(state.duplicate_hash_owner)
    for parsed in sorted(parsed_commitments, key=lambda p: p.hotkey):
        if parsed.hotkey in state.excluded_hotkeys:
            continue
        key = f"{parsed.hotkey}:{parsed.commitment.key}"
        existing = state.commitments.get(key)
        full_check = existing is None or not existing.artifact_hash
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
        entry = CommitmentState(
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
        )
        if entry.valid and artifact_hash:
            owner = valid_hash_owner.get(artifact_hash)
            if owner and owner != parsed.hotkey:
                entry.valid = False
                entry.disqualification_reason = f"duplicate artifact; first owner is {owner}"
            else:
                valid_hash_owner[artifact_hash] = parsed.hotkey
        state.commitments[key] = entry
    state.duplicate_hash_owner = valid_hash_owner


def _make_runner(args) -> object:
    if args.runner == "local":
        return LocalSubprocessRunner()
    from eval.runner_chutes import ChutesRunner

    return ChutesRunner(
        reference_sku=args.reference_sku,
        key_file=args.chutes_key_file,
        compress_base_url=args.compress_chute_url,
        decompress_base_url=args.decompress_chute_url,
    )


def _default_corpus_dir() -> Path:
    return Path(os.environ.get(MIXED_CORPUS_ENV, DEFAULT_MIXED_CORPUS_DIR))


def _resolve_corpus_dir(corpus_dir: str | None) -> Path:
    path = Path(corpus_dir) if corpus_dir else _default_corpus_dir()
    if not path.is_dir():
        if corpus_dir:
            raise SystemExit(f"corpus directory not found: {path}")
        raise SystemExit(
            "default mixed corpus directory not found: "
            f"{path}. Create/populate the mixed launch corpus, set {MIXED_CORPUS_ENV}, "
            "or pass --corpus-dir explicitly."
        )
    return path


def _make_provider(args):
    path = _resolve_corpus_dir(args.corpus_dir)
    provider = StaticLocalProvider(path, base_url=getattr(args, "corpus_url", None))
    if provider.total_bytes <= 0:
        raise SystemExit(f"corpus directory has no benchmark data files: {path}")
    return provider


def _make_chain(args) -> BittensorChain:
    return BittensorChain(
        ChainConfig(
            netuid=args.netuid,
            network=args.network,
            wallet_name=args.wallet_name,
            hotkey_name=args.hotkey_name,
            wallet_path=args.wallet_path,
        )
    )


def _evaluate_round(args, state: ValidatorState, chain: BittensorChain, salt: str) -> int:
    """Precheck commitments and, if there are new challengers, run one reign round."""

    block = chain.current_block()
    if state.window_anchor_block is None:
        state.window_anchor_block = args.window_anchor or block
    raw_commitments = chain.get_all_commitments()
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
    eligible = state.eligible_hotkeys()
    state.winner_history = compact_history(state.winner_history, eligible_hotkeys=eligible)

    challengers = [
        c
        for c in state.commitments.values()
        if c.valid and c.key not in state.scores and c.hotkey not in state.excluded_hotkeys
    ]
    if challengers:
        provider = _make_provider(args)
        seed = derive_seed(chain.block_hash(block), salt, block)
        specs = sample_streams(seed, provider.total_bytes, stream_bytes=args.stream_bytes, streams=args.streams)
        baseline = zstd_baseline_ratio([provider.materialize(s) for s in specs], level=args.baseline_level)
        runner = _make_runner(args)
        caps = ResourceCaps(wall_clock_secs=args.compress_budget_secs, artifact_bytes=args.max_artifact_bytes)
        print(f"round: {len(challengers)} challenger(s), baseline zstd ratio={baseline:.4f}")
        run_round(
            state, runner, challengers, provider, specs,
            caps=caps, floor_bps=args.floor_bps, budget_secs=args.compress_budget_secs,
            margin=args.win_margin, block=block, eligible_hotkeys=eligible, baseline_ratio=baseline,
        )
    return block


# --------------------------------------------------------------------------------------
# Production paths (require chain access)
# --------------------------------------------------------------------------------------


def run_once(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir)
    state_path = state_dir / "validator_state.json"
    state = load_state(state_path)
    salt = _load_salt(state_dir, args.salt_file)

    chain = _make_chain(args)
    version_key = _local_version_key()
    print(f"version key ok: {_assert_version_key_matches(chain)}")
    if not chain.commit_reveal_enabled():
        print("WARNING: commit-reveal is not enabled on this subnet; anti-copy weights are weaker")

    block = _evaluate_round(args, state, chain, salt)
    tempo = chain.tempo()
    anchor = state.window_anchor_block

    metagraph = chain.metagraph()
    hotkeys = list(metagraph.hotkeys)
    uids = [int(uid) for uid in metagraph.uids]

    weights, burn = decide_weights(
        hotkeys, state.winner_history, block=block, tempo=tempo,
        last_round_outputs=state.last_round_outputs, anchor=anchor, burn_uid=args.burn_uid,
    )
    save_state(state_path, state)

    nonzero = [(uids[i], round(w, 4)) for i, w in enumerate(weights) if w > 0]
    print(f"block={block} tempo={tempo} burn_tempo={burn} weights={nonzero}")
    if args.dry_run:
        print("dry-run: not submitting weights")
        return
    print(f"set_weights response: {chain.set_weights(uids, weights, version_key=version_key)}")


def run_reign_only(args: argparse.Namespace) -> None:
    """The reign half only (evaluate + update crown), no weight setting."""

    state_dir = Path(args.state_dir)
    state_path = state_dir / "validator_state.json"
    state = load_state(state_path)
    salt = _load_salt(state_dir, args.salt_file)

    chain = _make_chain(args)
    print(f"version key ok: {_assert_version_key_matches(chain)}")
    _evaluate_round(args, state, chain, salt)
    save_state(state_path, state)
    print(f"winner history = {[(w.hotkey, round(w.ratio, 4)) for w in state.winner_history]}")


# --------------------------------------------------------------------------------------
# Offline M0 demo (no chain): eval -> score -> king-of-the-hill -> weights end to end
# --------------------------------------------------------------------------------------


def run_offline_demo(args: argparse.Namespace) -> None:
    codec_specs = args.local_codec or ["winner=./reference_codec"]
    provider = _make_provider(args)
    seed = derive_seed("offline-demo-block", "offline-demo-salt", 0)
    specs = sample_streams(seed, provider.total_bytes, stream_bytes=args.stream_bytes, streams=args.streams)
    baseline = zstd_baseline_ratio([provider.materialize(s) for s in specs], level=args.baseline_level)

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

    print(
        f"streams={len(specs)} stream_bytes={args.stream_bytes} "
        f"baseline zstd-{args.baseline_level} ratio={baseline:.4f}"
    )
    history: list[WinnerEntry] = []
    for block_n, (hotkey, _path) in enumerate(parsed_codecs):
        outcome = outcomes[hotkey]
        beats = outcome.score.valid and outcome.score.ratio < baseline
        status = "valid" if outcome.score.valid else f"INVALID {outcome.score.reasons}"
        print(
            f"  {hotkey}: ratio={outcome.score.ratio:.4f} "
            f"min_throughput={outcome.score.throughput_bps_min:.0f} B/s "
            f"beats_baseline={beats} [{status}]"
        )
        if beats and (not history or should_promote(outcome.score.ratio, history[0].ratio, args.win_margin)):
            winner = WinnerEntry(hotkey, f"{hotkey}/codec", "local", outcome.score.ratio, block_n)
            history = promote_winner(history, winner)

    last_round_outputs = outcomes[history[0].hotkey].burn_outputs() if history else []
    print(f"winner history = {[(w.hotkey, round(w.ratio, 4)) for w in history]}")
    print("temporal burn schedule (two 4-tempo windows):")
    for tempo_idx in range(8):
        block = tempo_idx * 360
        weights, burn = decide_weights(
            hotkeys, history, block=block, tempo=360, last_round_outputs=last_round_outputs, anchor=0
        )
        nonzero = [(hotkeys[i], round(w, 3)) for i, w in enumerate(weights) if w > 0]
        print(f"  tempo {tempo_idx} (block {block}): burn={burn} weights={nonzero}")


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
            traceback.print_exc()


def main() -> None:
    from core.dotenv import load_dotenv

    load_dotenv()  # CHUTES_API_KEY etc. from .env (see .env.example)
    args = build_parser().parse_args()
    if args.offline_demo:
        run_offline_demo(args)
        return
    poll_chain = None
    while True:
        try:
            run_once(args)
        except KeyboardInterrupt:
            break
        except Exception:
            traceback.print_exc()
        if not args.loop:
            break
        # Use the inter-round wait to poll commitments at block cadence so commit-phase
        # blocks are captured before their reveal (instead of idling for --sleep).
        if poll_chain is None:
            poll_chain = _make_chain(args)
        _sleep_with_commit_polls(args, poll_chain)


if __name__ == "__main__":
    main()
