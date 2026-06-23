"""Version-key safety helpers.

The subnet ``weights_version`` must match the local package ``__version_key__`` before
validators score or submit weights. A mismatch means the validator is running code for a
different weight protocol and must fail closed.
"""

from __future__ import annotations

import core


def local_version_key() -> int:
    return int(core.__version_key__)


def assert_weights_version_matches(chain) -> int:
    expected = local_version_key()
    chain_version = chain.get_weights_version()
    if chain_version != expected:
        netuid = getattr(getattr(chain, "config", None), "netuid", "unknown")
        raise SystemExit(
            "version key mismatch: "
            f"local validator expects {expected}, but netuid {netuid} has "
            f"weights_version {chain_version}. Stopping before scoring or setting weights."
        )
    return chain_version
