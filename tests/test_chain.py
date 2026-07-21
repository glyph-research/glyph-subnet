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


# --- blockmachine archive source for the conviction backfill (issue #151) ---------------


def _archive_chain(*, key=None) -> BittensorChain:
    chain = object.__new__(BittensorChain)
    chain.config = type("Config", (), {"netuid": 117, "blockmachine_api_key": key})()
    chain._archive_pool = {}
    return chain


def test_archive_endpoints_prefer_blockmachine_only_when_a_key_is_configured():
    from core.constants import ARCHIVE_CHAIN_ENDPOINT

    assert _archive_chain()._archive_endpoints() == [ARCHIVE_CHAIN_ENDPOINT]
    with_key = _archive_chain(key="sekrit-key")._archive_endpoints()
    # ?authorization= is the only working websocket auth form: bt.Subtensor can't set an
    # Authorization header, ?api_key= silently lands on an unauthenticated limited tier,
    # and the path-embedded form 404s.
    assert with_key == ["wss://rpc.blockmachine.io?authorization=sekrit-key", ARCHIVE_CHAIN_ENDPOINT]


class _FakeMetagraph:
    hotkeys = ["hk-a", "hk-b"]
    emission = [100.0, 40.0]


class _ScriptedBt:
    """bt module stand-in: Subtensor(network=...) whose metagraph behavior is scripted per
    endpoint prefix."""

    def __init__(self, behavior):
        self._behavior = behavior  # endpoint-substring -> exception | metagraph
        self.constructed = []

    def Subtensor(self, network):  # noqa: N802 - mirrors the SDK surface
        self.constructed.append(network)
        behavior = next(v for k, v in self._behavior.items() if k in network)

        class _Sub:
            def metagraph(self, netuid, block=None):
                if isinstance(behavior, Exception):
                    raise behavior
                return behavior

        return _Sub()


def test_archive_query_falls_back_to_the_public_node_and_never_logs_the_key(caplog):
    from bittensor.utils.btlogging import logging as bt_logging

    bt_logging.set_info()
    chain = _archive_chain(key="sekrit-key")
    chain.bt = _ScriptedBt({
        "blockmachine": ConnectionError("handshake failed for wss://rpc.blockmachine.io?authorization=sekrit-key"),
        "archive.chain.opentensor.ai": _FakeMetagraph(),
    })

    result = chain.archive_emissions_by_hotkey(8_640_000)

    assert result == {"hk-a": 100.0, "hk-b": 40.0}  # ledger still advances
    assert "failed via blockmachine" in caplog.text
    assert "sekrit-key" not in caplog.text  # redacted even inside exception/URL text
    assert "<redacted>" in caplog.text


def test_state_discarded_from_blockmachine_hints_at_a_bad_key(caplog):
    from bittensor.utils.btlogging import logging as bt_logging

    bt_logging.set_info()
    chain = _archive_chain(key="sekrit-key")
    chain.bt = _ScriptedBt({
        "blockmachine": RuntimeError("State discarded for block ... use an archive node"),
        "archive.chain.opentensor.ai": _FakeMetagraph(),
    })

    result = chain.archive_emissions_by_hotkey(8_640_000)

    assert result == {"hk-a": 100.0, "hk-b": 40.0}
    assert "key may be invalid/expired" in caplog.text  # not misread as node pruning


def test_archive_query_raises_only_when_every_source_fails(caplog):
    from bittensor.utils.btlogging import logging as bt_logging
    import pytest

    bt_logging.set_info()
    chain = _archive_chain(key="sekrit-key")
    chain.bt = _ScriptedBt({
        "blockmachine": ConnectionError("down"),
        "archive.chain.opentensor.ai": ConnectionError("also down"),
    })

    with pytest.raises(ConnectionError):
        chain.archive_emissions_by_hotkey(8_640_000)
    # Failed connections are dropped from the pool so the next attempt reconnects fresh.
    assert chain._archive_pool == {}


def test_archive_connections_are_reused_across_queries():
    chain = _archive_chain()
    chain.bt = _ScriptedBt({"archive.chain.opentensor.ai": _FakeMetagraph()})

    chain.archive_emissions_by_hotkey(1)
    chain.archive_emissions_by_hotkey(2)

    assert len(chain.bt.constructed) == 1  # one connection, reused


def test_resolve_blockmachine_key_prefers_key_file_over_env(tmp_path, monkeypatch):
    from validator.service import resolve_blockmachine_key

    monkeypatch.setenv("BLOCKMACHINE_API_KEY", "env-key")
    args = type("Args", (), {"blockmachine_key_file": None})()
    assert resolve_blockmachine_key(args) == "env-key"

    key_file = tmp_path / "bm.key"
    key_file.write_text("file-key\n")
    args = type("Args", (), {"blockmachine_key_file": str(key_file)})()
    assert resolve_blockmachine_key(args) == "file-key"

    monkeypatch.delenv("BLOCKMACHINE_API_KEY", raising=False)
    args = type("Args", (), {"blockmachine_key_file": None})()
    assert resolve_blockmachine_key(args) is None


# --- Conviction v1.1: hotkey-side locked-alpha read (issue #156) -------------------------


def test_locked_alpha_by_hotkey_aggregates_per_hotkey_and_converts_rao_to_alpha():
    chain = object.__new__(BittensorChain)
    chain.config = type("Config", (), {"netuid": 117})()

    class _Subtensor:
        def get_hotkey_conviction(self, hotkey, netuid):
            assert netuid == 117
            # Runtime API returns the decayed lock mass in rao (verified live: exactly
            # the lock's stored locked_mass); hk-b has never locked -> 0.0.
            return {"hk-a": 27_681_451_239_434.0, "hk-b": 0.0}[hotkey]

    chain.subtensor = _Subtensor()
    assert chain.locked_alpha_by_hotkey(["hk-a", "hk-b"]) == {
        "hk-a": 27_681.451239434,
        "hk-b": 0.0,
    }
