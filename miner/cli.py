"""Glyph miner CLI dispatcher: commit | check | publish | register.

Usage: ``glyph-miner <subcommand> [options]`` (e.g. ``glyph-miner check --local-path .``).
Each subcommand is also importable/runnable on its own (``python -m miner.commit``).
"""

from __future__ import annotations

import sys

from miner import check, commit, publish, register
from miner.env import load_miner_env

SUBCOMMANDS = {
    "commit": commit,
    "check": check,
    "publish": publish,
    "register": register,
}

_USAGE = """usage: glyph-miner <subcommand> [options]

subcommands:
  check     validate + locally round-trip a codec (self-benchmark before committing)
  publish   validate (and optionally upload) a local codec artifact to HuggingFace
  register  register a hotkey on the subnet (burned registration)
  commit    commit a codec (HF repo@rev) -- permanent for this hotkey
"""


def main() -> None:
    load_miner_env()
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_USAGE)
        raise SystemExit(0)
    sub = sys.argv[1]
    if sub not in SUBCOMMANDS:
        print(f"unknown subcommand: {sub}\n")
        print(_USAGE)
        raise SystemExit(2)
    # Re-point argv so the subcommand's argparse sees only its own arguments.
    sys.argv = [f"glyph-miner {sub}", *sys.argv[2:]]
    SUBCOMMANDS[sub].main()


if __name__ == "__main__":
    main()
