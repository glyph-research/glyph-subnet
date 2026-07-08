"""issue #79: chain.set_weights returns a bare, contentless failure when still inside the
subnet's weights-rate-limit window (the bittensor SDK's own _blocks_weight_limit check skips
the extrinsic attempt entirely) -- blocks_until_weights_allowed lets the caller detect and
log that explicitly instead of a silent "success=False error=None message=None"."""

from chain.chain import BittensorChain


class FakeSubtensor:
    def __init__(self, *, uid, since, limit):
        self._uid = uid
        self._since = since
        self._limit = limit

    def get_uid_for_hotkey_on_subnet(self, hotkey_ss58, netuid, block=None):
        return self._uid

    def blocks_since_last_update(self, netuid, uid, block=None):
        return self._since

    def weights_rate_limit(self, netuid, block=None):
        return self._limit


def _chain(subtensor: FakeSubtensor) -> BittensorChain:
    chain = object.__new__(BittensorChain)
    chain.config = type("Config", (), {"netuid": 117})()
    chain.subtensor = subtensor
    chain.wallet = type("Wallet", (), {"hotkey": type("Hotkey", (), {"ss58_address": "hk-ss58"})()})()
    return chain


def test_blocks_until_weights_allowed_computes_remaining():
    chain = _chain(FakeSubtensor(uid=5, since=100, limit=360))
    assert chain.blocks_until_weights_allowed() == 260


def test_blocks_until_weights_allowed_clamps_to_zero_when_past_due():
    chain = _chain(FakeSubtensor(uid=5, since=500, limit=360))
    assert chain.blocks_until_weights_allowed() == 0


def test_blocks_until_weights_allowed_none_when_not_registered():
    chain = _chain(FakeSubtensor(uid=None, since=100, limit=360))
    assert chain.blocks_until_weights_allowed() is None


def test_blocks_until_weights_allowed_none_when_sdk_returns_none():
    chain = _chain(FakeSubtensor(uid=5, since=None, limit=360))
    assert chain.blocks_until_weights_allowed() is None
