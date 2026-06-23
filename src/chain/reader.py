"""Read and print on-chain Glyph commitments (monitoring / debugging).

Runnable as ``glyph-chain-reader`` or ``python -m chain``.
"""

from __future__ import annotations

import argparse

from chain.chain import BittensorChain, ChainConfig
from core.commitments import parse_commitments_by_hotkey
from core.constants import DEFAULT_NETUID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump on-chain Glyph commitments")
    parser.add_argument("--network", default="finney")
    parser.add_argument("--netuid", type=int, default=DEFAULT_NETUID)
    parser.add_argument("--wallet-name", "--wallet.name", dest="wallet_name", default="default")
    parser.add_argument("--hotkey-name", "--wallet.hotkey", dest="hotkey_name", default="default")
    parser.add_argument("--wallet-path", "--wallet.path", dest="wallet_path", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    chain = BittensorChain(
        ChainConfig(
            netuid=args.netuid,
            network=args.network,
            wallet_name=args.wallet_name,
            hotkey_name=args.hotkey_name,
            wallet_path=args.wallet_path,
        )
    )
    parsed = parse_commitments_by_hotkey(chain.get_all_commitments())
    print(
        f"netuid={args.netuid} block={chain.current_block()} tempo={chain.tempo()} "
        f"commit_reveal={chain.commit_reveal_enabled()}"
    )
    for item in sorted(parsed, key=lambda p: p.hotkey):
        print(f"  {item.hotkey}  {item.commitment.repo}@{item.commitment.rev}")
    print(f"total commitments: {len(parsed)}")


if __name__ == "__main__":
    main()
