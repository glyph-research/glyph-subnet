"""Register a hotkey on the Glyph subnet (burned registration).

Each codec commitment is permanent per hotkey, so submitting a new/improved codec means
registering a fresh hotkey -- the registration burn is the spam-proof submission fee.
"""

from __future__ import annotations

import argparse

from core.constants import DEFAULT_NETUID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register a hotkey on the Glyph subnet")
    parser.add_argument("--network", default="finney")
    parser.add_argument("--netuid", type=int, default=DEFAULT_NETUID)
    parser.add_argument("--wallet-name", "--wallet.name", dest="wallet_name", default="default")
    parser.add_argument("--hotkey-name", "--wallet.hotkey", dest="hotkey_name", default="default")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import bittensor as bt

    subtensor = bt.Subtensor(network=args.network)
    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.hotkey_name)
    hotkey = wallet.hotkey.ss58_address

    if subtensor.is_hotkey_registered(netuid=args.netuid, hotkey_ss58=hotkey):
        print(f"hotkey {hotkey} already registered on netuid {args.netuid}")
        return

    try:
        cost = subtensor.get_subnet_burn_cost()
        print(f"hotkey={hotkey} netuid={args.netuid} burn_cost={cost}")
    except Exception:
        print(f"hotkey={hotkey} netuid={args.netuid}")

    if args.dry_run:
        print("dry-run: not registering")
        return
    if not args.yes:
        answer = input("Burn TAO to register this hotkey? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("aborted")
            return

    response = subtensor.burned_register(wallet=wallet, netuid=args.netuid)
    print("registered" if response else f"register response: {response}")


if __name__ == "__main__":
    main()
