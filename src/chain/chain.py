"""Thin Bittensor chain adapter.

Wraps the Bittensor SDK so the validator's core logic can be unit-tested without chain
access. ``set_weights`` is called exactly the same whether or not commit-reveal is
enabled on the subnet -- in bittensor >=10.x the SDK auto-routes through commit/reveal
when the subnet's hyperparameter turns it on, so there is no manual commit/reveal
choreography here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChainConfig:
    netuid: int
    network: str
    wallet_name: str
    hotkey_name: str
    wallet_path: str | None = None
    # Optional blockmachine.io API key (paid plan): makes it the preferred archive source
    # for historical queries (issue #151). Absent -> public archive node only.
    blockmachine_api_key: str | None = None


class BittensorChain:
    def __init__(self, config: ChainConfig):
        import bittensor as bt

        self.config = config
        self.bt = bt
        self.subtensor = bt.Subtensor(network=config.network)
        # Lazily-created archive connections (one per endpoint) for historical state beyond
        # the live node's pruning horizon (conviction-ledger backfill, issues #141/#151).
        self._archive_pool: dict[str, object] = {}
        wallet_kwargs = {"name": config.wallet_name, "hotkey": config.hotkey_name}
        if config.wallet_path:
            wallet_kwargs["path"] = config.wallet_path
        self.wallet = bt.Wallet(**wallet_kwargs)

    @property
    def hotkey(self) -> str:
        return self.wallet.hotkey.ss58_address

    def identity_name(self) -> str | None:
        """This validator's on-chain identity name (``btcli wallet set-identity``), or None
        if unset/unavailable.

        Reads the coldkey's public ss58 address only -- no wallet unlock needed. Best-effort:
        identity is a nice-to-have (currently just a wandb run label), so any chain hiccup
        here is swallowed rather than allowed to block validator startup.
        """

        try:
            identity = self.subtensor.query_identity(self.wallet.coldkeypub.ss58_address)
        except Exception:
            return None
        if identity is None:
            return None
        return identity.name or None

    def metagraph(self):
        return self.subtensor.metagraph(netuid=self.config.netuid)

    def emissions_by_hotkey(self, block: int) -> dict[str, float]:
        """Per-hotkey alpha emitted for the tempo ending at ``block`` (live node).

        ``metagraph.emission`` is denominated in alpha per tempo (verified live on netuid
        117: storage ``Emission`` in rao / 1e9). The live node only serves recent state;
        older blocks must go through ``archive_emissions_by_hotkey``.
        """

        mg = self.subtensor.metagraph(netuid=self.config.netuid, block=block)
        return {hotkey: float(e) for hotkey, e in zip(mg.hotkeys, mg.emission)}

    def _archive_endpoints(self) -> list[str]:
        """Ordered archive candidates for historical queries (issue #151): blockmachine
        (authenticated, ~1-3s per metagraph-at-block, paid plan) first when a key is
        configured, then the public archive node (~20-30s and frequent overloads).
        Historical chain state is objective, so the endpoint choice is a purely
        operator-local preference, never consensus-relevant.

        Auth pitfalls (measured; they cost real debugging time): ``bt.Subtensor``
        websockets cannot set an Authorization header, so the key rides blockmachine's
        documented query-string fallback ``?authorization=<key>``. ``?api_key=<key>`` is
        WRONG but fails silently -- it connects on an unauthenticated limited tier and only
        deep blocks fail ("State discarded"), masquerading as node pruning. The
        path-embedded form (``wss://.../<key>``) 404s.
        """

        from core.constants import ARCHIVE_CHAIN_ENDPOINT, BLOCKMACHINE_RPC_ENDPOINT

        endpoints = []
        if self.config.blockmachine_api_key:
            endpoints.append(
                f"{BLOCKMACHINE_RPC_ENDPOINT}?authorization={self.config.blockmachine_api_key}"
            )
        endpoints.append(ARCHIVE_CHAIN_ENDPOINT)
        return endpoints

    def _redact_key(self, text: str) -> str:
        """The API key must never reach a log line -- it can appear in websocket/URL
        exception text."""

        key = self.config.blockmachine_api_key
        return text.replace(key, "<redacted>") if key else text

    def archive_emissions_by_hotkey(self, block: int) -> dict[str, float]:
        """Same as ``emissions_by_hotkey`` but via an archive source -- the
        conviction-ledger backfill path for blocks past the live node's pruning horizon.

        Tries blockmachine first when a key is configured and falls back to the public
        archive for this query on ANY failure (same transient-safe posture as #120/#132):
        a bad/expired key degrades to the slower public node, never a blocked ledger or
        weight-setting.
        """

        from bittensor.utils.btlogging import logging as bt_logging

        from core.constants import BLOCKMACHINE_RPC_ENDPOINT

        last_exc: Exception | None = None
        for endpoint in self._archive_endpoints():
            try:
                subtensor = self._archive_pool.get(endpoint)
                if subtensor is None:
                    subtensor = self.bt.Subtensor(network=endpoint)
                    self._archive_pool[endpoint] = subtensor
                mg = subtensor.metagraph(netuid=self.config.netuid, block=block)
                return {hotkey: float(e) for hotkey, e in zip(mg.hotkeys, mg.emission)}
            except Exception as exc:  # noqa: BLE001 - fall through to the next source
                self._archive_pool.pop(endpoint, None)  # reconnect fresh next attempt
                is_blockmachine = endpoint.startswith(BLOCKMACHINE_RPC_ENDPOINT)
                hint = (
                    " -- the blockmachine key may be invalid/expired (the unauthenticated "
                    "tier discards old state, masquerading as pruning)"
                    if is_blockmachine and "state discarded" in str(exc).lower()
                    else ""
                )
                source = "blockmachine" if is_blockmachine else "public archive"
                bt_logging.warning(
                    f"archive query at block {block} failed via {source}{hint}: "
                    f"{self._redact_key(str(exc))}"
                )
                last_exc = exc
        raise last_exc

    def get_all_commitments(self) -> dict[str, str]:
        return self.subtensor.get_all_commitments(netuid=self.config.netuid)

    def get_weights_version(self) -> int:
        hyperparameters = self.subtensor.get_subnet_hyperparameters(netuid=self.config.netuid)
        if hyperparameters is None:
            raise RuntimeError(
                f"could not read subnet hyperparameters for netuid {self.config.netuid}"
            )
        value = getattr(hyperparameters, "weights_version", None)
        if value is None:
            raise RuntimeError("subnet hyperparameters did not include weights_version")
        return int(value)

    def tempo(self) -> int:
        value = self.subtensor.tempo(netuid=self.config.netuid)
        if not value:
            raise RuntimeError(f"could not read tempo for netuid {self.config.netuid}")
        return int(value)

    def commit_reveal_enabled(self) -> bool:
        return bool(self.subtensor.commit_reveal_enabled(netuid=self.config.netuid))

    def current_block(self) -> int:
        return int(self.subtensor.get_current_block())

    def block_hash(self, block: int) -> str:
        return str(self.subtensor.get_block_hash(block))

    def next_epoch_start_block(self) -> int:
        return int(self.subtensor.get_next_epoch_start_block(netuid=self.config.netuid))

    def get_my_commitment(self) -> str | None:
        raw_commitments = self.get_all_commitments()
        raw = raw_commitments.get(self.hotkey)
        if raw:
            return raw
        raw = self.subtensor.get_commitment_metadata(
            netuid=self.config.netuid,
            hotkey_ss58=self.hotkey,
        )
        if raw is None:
            return None
        return str(raw)

    def set_commitment(self, data: str):
        return self.subtensor.set_commitment(
            wallet=self.wallet,
            netuid=self.config.netuid,
            data=data,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )

    def set_weights(self, uids: list[int], weights: list[float], *, version_key: int):
        return self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uids,
            weights=weights,
            version_key=version_key,
            wait_for_inclusion=True,
        )

    def blocks_until_weights_allowed(self) -> int | None:
        """Blocks remaining until this validator's next set_weights is accepted, or None if
        that can't be determined (not registered on this subnet, or the SDK returned nothing).

        Below the subnet's weights_rate_limit, the SDK's own set_weights short-circuits before
        even attempting the extrinsic and returns a bare, contentless failure -- no error, no
        message, by construction (issue #79). Checking this ourselves lets the caller log
        *why* explicitly instead of a silent no-op-looking failure.
        """

        uid = self.subtensor.get_uid_for_hotkey_on_subnet(self.hotkey, self.config.netuid)
        if uid is None:
            return None
        since = self.subtensor.blocks_since_last_update(self.config.netuid, uid)
        limit = self.subtensor.weights_rate_limit(self.config.netuid)
        if since is None or limit is None:
            return None
        return max(0, limit - since)
