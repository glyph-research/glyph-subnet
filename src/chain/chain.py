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


class BittensorChain:
    def __init__(self, config: ChainConfig):
        import bittensor as bt

        self.config = config
        self.bt = bt
        self.subtensor = bt.Subtensor(network=config.network)
        wallet_kwargs = {"name": config.wallet_name, "hotkey": config.hotkey_name}
        if config.wallet_path:
            wallet_kwargs["path"] = config.wallet_path
        self.wallet = bt.Wallet(**wallet_kwargs)

    @property
    def hotkey(self) -> str:
        return self.wallet.hotkey.ss58_address

    def metagraph(self):
        return self.subtensor.metagraph(netuid=self.config.netuid)

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
