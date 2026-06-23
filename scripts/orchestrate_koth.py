"""Staggered king-of-the-hill run on testnet netuid 509.

Three miners submit one after another (~1 hr apart): bzip2 -> xz -> ppmd. After each
commit the validator (owner) evaluates on enwik8 (4x20 MiB) and sets weights on-chain.
Each stronger codec dethrones the last; weights go 70/30 (current/previous) after a
dethroning. Run from the repo root.

    python scripts/orchestrate_koth.py [start_round] [gap_seconds]
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import bittensor as bt

from core.commitments import CodecCommitment, serialize_commitment

NETUID = 509
GAP = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
START = int(sys.argv[1]) if len(sys.argv) > 1 else 1
STREAM = str(20 * 2**20)

revs = {}
for line in open("/tmp/glyph_test_revs.txt"):
    name, repo, rev = line.split()
    revs[name] = (repo, rev)

st = bt.Subtensor(network="test")
WALLET = {h: bt.Wallet(name="glyph_miner", hotkey=h) for h in ("m3", "m4", "m5")}
SS = {h: WALLET[h].hotkey.ss58_address for h in WALLET}

# (miner hotkey, codec, hotkeys included in this round's evaluation set)
ROUNDS = [
    ("m3", "bzip2", ["m3"]),
    ("m4", "xz", ["m3", "m4"]),
    ("m5", "ppmd", ["m3", "m4", "m5"]),
]


def commit(hk: str, name: str) -> None:
    repo, rev = revs[name]
    data = serialize_commitment(CodecCommitment(repo=repo, rev=rev))
    r = st.set_commitment(wallet=WALLET[hk], netuid=NETUID, data=data,
                          wait_for_inclusion=True, wait_for_finalization=True)
    print(f"[commit] {hk} {name} {data} -> {getattr(r, 'success', r)}", flush=True)


def run_round(only_hks: list[str]) -> None:
    cmd = [
        "glyph-validator", "--network", "test", "--netuid", str(NETUID),
        "--wallet-name", "glyph_owner", "--hotkey-name", "default",
        "--runner", "local", "--corpus-dir", "bench_corpus",
        "--stream-bytes", STREAM, "--streams", "4",
        "--floor-bps", "10240", "--baseline-level", "3", "--state-dir", "./prod_state",
    ]
    for h in only_hks:
        cmd += ["--only-hotkeys", SS[h]]
    env = {**os.environ, "HF_TOKEN": open("/tmp/glyph_hf_token").read().strip()}
    subprocess.run(cmd, env=env)


def main() -> None:
    for i, (hk, name, only) in enumerate(ROUNDS):
        rnd = i + 1
        if rnd < START:
            continue
        print(f"\n===== ROUND {rnd}: miner {hk} submits {name} =====", flush=True)
        commit(hk, name)
        run_round(only)
        if rnd < len(ROUNDS):
            print(f"[wait] sleeping {GAP}s until the next miner submits...", flush=True)
            time.sleep(GAP)
    print("\n===== KOTH COMPLETE =====", flush=True)


if __name__ == "__main__":
    main()
