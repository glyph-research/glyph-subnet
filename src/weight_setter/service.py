"""Temporal-burn weight setting.

``decide_weights`` computes the weights for a single tempo from the winner history and the
burn seed. ``main`` is a standalone service that reads validator state, computes weights
for the current chain tempo, and submits them via commit-reveal -- runnable as its own
PM2 process.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from bittensor.utils.btlogging import logging as bt_logging

from core.burn_schedule import derive_burn_seed, is_burn_tempo
from core.commitments import parse_burn_override
from core.constants import BURN_ENABLED, BURN_UID, DEFAULT_NETUID, WINDOW_ANCHOR_BLOCK
from core.log_config import add_logging_args
from core.state import load_state
from core.version import assert_weights_version_matches, local_version_key
from core.weights import WinnerEntry, compute_weights


def resolve_force_burn(raw_commitments: dict[str, str], owner_hotkey: str) -> bool:
    """True only if ``owner_hotkey`` (whichever hotkey currently occupies BURN_UID on the
    live metagraph) has published a ``force_burn=true`` commitment (issue #113).

    Missing, malformed, or ``force_burn=false`` all resolve to False -- this is strictly
    additive (can only ever force MORE burning), never a way to suppress a scheduled burn.
    """

    raw = raw_commitments.get(owner_hotkey)
    if not raw:
        return False
    override = parse_burn_override(raw)
    return bool(override and override.force_burn)


def decide_weights(
    hotkeys: list[str],
    history: list[WinnerEntry],
    *,
    block: int,
    tempo: int,
    last_round_outputs,
    anchor: int = WINDOW_ANCHOR_BLOCK,
    burn_uid: int = BURN_UID,
    force_burn: bool = False,
    gated_hotkeys: set[str] | None = None,
) -> tuple[list[float], bool]:
    """Compute the weights for the current tempo, applying the temporal burn schedule.

    Short-circuits to "never a burn tempo" when ``core.constants.BURN_ENABLED`` is False
    (issue #43) -- a network-wide, source-committed switch, not a per-operator override.

    ``force_burn`` (issue #113) is the caller's already-resolved on-chain owner override
    (see ``resolve_force_burn``): when True, burn unconditionally regardless of
    ``BURN_ENABLED``/the schedule -- an emergency, owner-controlled kill switch, additive
    only (can force a burn tempo, never suppress one).

    ``gated_hotkeys`` (Miner Conviction, issues #141/#170) is the caller's already-computed
    set of winners below their required conviction this tempo; they are skipped when
    picking payees, so the pot goes to the two most recent compliant winners in the
    retained history and burns only when none qualifies (see ``compute_weights``).
    """

    seed = derive_burn_seed(last_round_outputs)
    burn = force_burn or (BURN_ENABLED and is_burn_tempo(block, tempo, seed, anchor))
    weights = compute_weights(
        hotkeys, history, is_burn_tempo=burn, burn_uid=burn_uid, gated_hotkeys=gated_hotkeys
    )
    return weights, burn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Glyph weight setter")
    parser.add_argument("--network", default="finney")
    parser.add_argument("--netuid", type=int, default=DEFAULT_NETUID)
    parser.add_argument("--wallet-name", "--wallet.name", dest="wallet_name", default="default")
    parser.add_argument("--hotkey-name", "--wallet.hotkey", dest="hotkey_name", default="default")
    parser.add_argument("--wallet-path", "--wallet.path", dest="wallet_path", default=None)
    parser.add_argument("--state-dir", default="./state")
    parser.add_argument("--burn-uid", type=int, default=BURN_UID)
    parser.add_argument(
        "--blockmachine-key-file",
        default=None,
        help="Optional file containing a blockmachine.io API key (Standard plan) -- preferred "
        "archive source for the conviction-ledger backfill (issue #151). Defaults to the "
        "BLOCKMACHINE_API_KEY env var; unset -> public archive node.",
    )
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
        "testing/CI -- a real weight setter should not pass this.",
    )
    parser.add_argument("--sleep", type=int, default=1200)
    parser.add_argument("--dry-run", action="store_true")
    add_logging_args(parser)
    return parser


def run(args: argparse.Namespace) -> None:
    from chain.chain import BittensorChain, ChainConfig

    from validator.service import resolve_blockmachine_key

    state = load_state(Path(args.state_dir) / "validator_state.json")
    chain = BittensorChain(
        ChainConfig(
            netuid=args.netuid,
            network=args.network,
            wallet_name=args.wallet_name,
            hotkey_name=args.hotkey_name,
            wallet_path=args.wallet_path,
            blockmachine_api_key=resolve_blockmachine_key(args),
        )
    )
    version_key = local_version_key()
    try:
        bt_logging.info(f"version key ok: {assert_weights_version_matches(chain)}")
    except SystemExit as exc:
        # A version-key mismatch is the expected, transient state during a release rollout
        # until the owner updates the on-chain weights_version hyperparameter -- skip this
        # cycle (no weights set) and let main()'s sleep/retry loop pick up once the chain
        # catches up, instead of SystemExit sailing past `except Exception` and killing the
        # process into a pm2 crash-restart loop (issue #120). Scoped to the version check
        # only: any other SystemExit still hard-stops as before.
        bt_logging.warning(f"weights_version mismatch, waiting for on-chain update: {exc}")
        return
    block = chain.current_block()
    tempo = chain.tempo()
    anchor = (
        state.window_anchor_block
        if state.window_anchor_block is not None
        else (args.window_anchor or block)
    )
    metagraph = chain.metagraph()
    hotkeys = list(metagraph.hotkeys)
    uids = [int(uid) for uid in metagraph.uids]

    # Owner emergency burn override (issue #113) -- best-effort: a chain hiccup here must
    # not block weight-setting, so it just falls through to the unchanged normal schedule.
    try:
        raw_commitments = chain.get_all_commitments()
    except Exception:
        raw_commitments = {}
    force_burn = (
        resolve_force_burn(raw_commitments, hotkeys[args.burn_uid])
        if 0 <= args.burn_uid < len(hotkeys)
        else False
    )
    if force_burn:
        # Without this line, a forced burn is indistinguishable in the log from an ordinary
        # scheduled burn tempo -- except it happens EVERY tempo, which reads as a bug unless
        # the operator knows the owner override is active.
        bt_logging.warning(
            "Subnet faced an issue and turned into temporal burn "
            "(owner emergency override: on-chain force_burn=true) -- burning 100% this tempo"
        )

    # Miner Conviction (issue #141): top up the persisted ledger in memory (this process
    # never writes state -- the validator/reign worker persists it) and gate any winner
    # below its required stake lock. Best-effort, same posture as the validator path.
    from validator.service import _conviction_report_for_winners, _update_conviction_ledger

    _update_conviction_ledger(state, chain, block, tempo)
    conviction = _conviction_report_for_winners(
        state, metagraph, block, chain, burn_uid=args.burn_uid
    )
    gated = {hotkey for hotkey, entry in conviction.items() if not entry["compliant"]}

    weights, burn = decide_weights(
        hotkeys,
        state.winner_history,
        block=block,
        tempo=tempo,
        last_round_outputs=state.last_round_outputs,
        anchor=anchor,
        burn_uid=args.burn_uid,
        force_burn=force_burn,
        gated_hotkeys=gated,
    )
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


def main() -> None:
    from core.dotenv import load_dotenv
    from core.log_config import configure_logging

    load_dotenv()
    args = build_parser().parse_args()
    configure_logging(args)
    while True:
        try:
            run(args)
        except KeyboardInterrupt:
            break
        except Exception:
            bt_logging.exception("weight setter round failed")
        if args.once:
            break
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
