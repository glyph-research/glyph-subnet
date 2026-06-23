"""Temporal-burn weight setting (DESIGN §6.1).

``decide_weights`` computes the weights for a single tempo from the winner history and the
burn seed. ``main`` is a standalone service that reads validator state, computes weights
for the current chain tempo, and submits them via commit-reveal -- runnable as its own
PM2 process.
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from core.burn_schedule import derive_burn_seed, is_burn_tempo
from core.constants import BURN_UID, DEFAULT_NETUID, WINDOW_ANCHOR_BLOCK
from core.state import load_state
from core.version import assert_weights_version_matches, local_version_key
from core.weights import WinnerEntry, compute_weights


def decide_weights(
    hotkeys: list[str],
    history: list[WinnerEntry],
    *,
    block: int,
    tempo: int,
    last_round_outputs,
    anchor: int = WINDOW_ANCHOR_BLOCK,
    burn_uid: int = BURN_UID,
) -> tuple[list[float], bool]:
    """Compute the weights for the current tempo, applying the temporal burn schedule."""

    seed = derive_burn_seed(last_round_outputs)
    burn = is_burn_tempo(block, tempo, seed, anchor)
    weights = compute_weights(hotkeys, history, is_burn_tempo=burn, burn_uid=burn_uid)
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
    parser.add_argument("--window-anchor", type=int, default=WINDOW_ANCHOR_BLOCK)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--sleep", type=int, default=1200)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    from chain.chain import BittensorChain, ChainConfig

    state = load_state(Path(args.state_dir) / "validator_state.json")
    chain = BittensorChain(
        ChainConfig(
            netuid=args.netuid,
            network=args.network,
            wallet_name=args.wallet_name,
            hotkey_name=args.hotkey_name,
            wallet_path=args.wallet_path,
        )
    )
    version_key = local_version_key()
    print(f"version key ok: {assert_weights_version_matches(chain)}")
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

    weights, burn = decide_weights(
        hotkeys,
        state.winner_history,
        block=block,
        tempo=tempo,
        last_round_outputs=state.last_round_outputs,
        anchor=anchor,
        burn_uid=args.burn_uid,
    )
    nonzero = [(uids[i], round(w, 4)) for i, w in enumerate(weights) if w > 0]
    print(f"block={block} tempo={tempo} burn_tempo={burn} weights={nonzero}")
    if args.dry_run:
        print("dry-run: not submitting weights")
        return
    response = chain.set_weights(uids, weights, version_key=version_key)
    print(f"set_weights response: {response}")


def main() -> None:
    from core.dotenv import load_dotenv

    load_dotenv()
    args = build_parser().parse_args()
    while True:
        try:
            run(args)
        except KeyboardInterrupt:
            break
        except Exception:
            traceback.print_exc()
        if not args.loop:
            break
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
