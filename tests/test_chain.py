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


# --- identity_name (issue #102 follow-up: wandb run-name identity source) --------------


class _Identity:
    def __init__(self, name):
        self.name = name


class FakeSubtensorIdentity:
    def __init__(self, result):
        self._result = result  # a _Identity, None, or an Exception instance to raise

    def query_identity(self, coldkey_ss58):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _chain_with_coldkey(subtensor, coldkey_ss58="ck-ss58") -> BittensorChain:
    chain = object.__new__(BittensorChain)
    chain.config = type("Config", (), {"netuid": 117})()
    chain.subtensor = subtensor
    chain.wallet = type(
        "Wallet", (),
        {
            "hotkey": type("Hotkey", (), {"ss58_address": "hk-ss58"})(),
            "coldkeypub": type("Coldkeypub", (), {"ss58_address": coldkey_ss58})(),
        },
    )()
    return chain


def test_identity_name_returns_name_when_set():
    chain = _chain_with_coldkey(FakeSubtensorIdentity(_Identity("my-validator")))
    assert chain.identity_name() == "my-validator"


def test_identity_name_none_when_no_identity_set():
    chain = _chain_with_coldkey(FakeSubtensorIdentity(None))
    assert chain.identity_name() is None


def test_identity_name_none_when_identity_name_is_empty():
    chain = _chain_with_coldkey(FakeSubtensorIdentity(_Identity("")))
    assert chain.identity_name() is None


def test_identity_name_none_on_chain_error():
    # Best-effort: a chain hiccup here must never raise/block startup.
    chain = _chain_with_coldkey(FakeSubtensorIdentity(RuntimeError("network down")))
    assert chain.identity_name() is None


def test_identity_name_queries_the_coldkey_public_address():
    calls = []

    class _RecordingSubtensor:
        def query_identity(self, coldkey_ss58):
            calls.append(coldkey_ss58)
            return None

    chain = _chain_with_coldkey(_RecordingSubtensor(), coldkey_ss58="ck-ss58-xyz")
    chain.identity_name()
    assert calls == ["ck-ss58-xyz"]
