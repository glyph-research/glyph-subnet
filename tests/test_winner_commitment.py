"""issue #103: publish the current champion on-chain (the validator's own, otherwise-unused
commitment slot) whenever the crown changes -- observability/bootstrapping only, never read
back into this validator's own scoring/promotion, which always independently re-benchmarks
the on-chain codec commitments (reigning champion included) every round regardless."""

import json

from bittensor.utils.btlogging import logging as bt_logging

from core.commitments import (
    CodecCommitment,
    WinnerCommitment,
    parse_winner_commitment,
    serialize_commitment,
    serialize_winner_commitment,
)
from core.constants import SCORING_VERSION
from core.state import CommitmentState, ValidatorState
from core.weights import WinnerEntry
from eval.corpus import StaticLocalProvider
from validation.precheck import PrecheckResult
from validator.service import _evaluate_round

_REV_A = "a" * 40


def test_winner_commitment_round_trips_through_serializer_parser():
    winner = WinnerCommitment(
        hotkey="hk-a", repo="a/codec", rev=_REV_A, ratio=0.31, commit_block=100, scoring_version=SCORING_VERSION,
    )
    raw = serialize_winner_commitment(winner)
    assert parse_winner_commitment(raw) == winner


def test_parse_winner_commitment_returns_none_for_other_commitment_forms():
    assert parse_winner_commitment(serialize_commitment(CodecCommitment(repo="a/b", rev=_REV_A))) is None
    assert parse_winner_commitment("garbage, not even close") is None
    assert parse_winner_commitment("") is None


def test_parse_winner_commitment_rejects_unsupported_future_version():
    payload = {"v": 999, "hotkey": "hk-a", "repo": "a/codec", "rev": _REV_A, "ratio": 0.3, "commit_block": 1,
               "scoring_version": 1}
    raw = "g1w|" + json.dumps(payload)
    assert parse_winner_commitment(raw) is None


class _FakeChain:
    def __init__(self, block: int, block_hash: str, raw_commitments: dict):
        self._block = block
        self._block_hash = block_hash
        self._raw_commitments = raw_commitments
        self.set_commitment_calls: list[str] = []
        self.set_commitment_response = type("Response", (), {"success": True})()

    def current_block(self) -> int:
        return self._block

    def block_hash(self, block: int) -> str:
        return self._block_hash

    def get_all_commitments(self) -> dict:
        return self._raw_commitments

    def metagraph(self):
        return type("Metagraph", (), {"hotkeys": ["hk-a", "incumbent-hk"], "uids": [7, 3]})()

    def set_commitment(self, data: str):
        self.set_commitment_calls.append(data)
        return self.set_commitment_response


def _args(**overrides):
    defaults = {
        "window_anchor": None,
        "max_artifact_bytes": 10_000,
        "corpus_dir": None,
        "eval_source": "fineweb",
        "eval_streams": 1,
        "eval_stream_bytes": 1024,
        "eval_benchmark_source": "",
        "eval_benchmark_streams": 0,
        "baseline_level": 3,
        "compress_budget_secs": 60.0,
        "floor_bps": 1.0,
        "win_margin": 0.05,
        "runner": "local",
    }
    return type("Args", (), {**defaults, **overrides})()


def _setup_round(monkeypatch, tmp_path, *, winner_history, new_winner_entry_or_none):
    block = 12345
    block_hash = "0xbeacon"
    raw_commitments = {"hk-a": serialize_commitment(CodecCommitment(repo="a/codec", rev=_REV_A))}
    chain = _FakeChain(block, block_hash, raw_commitments)

    def fake_precheck(repo, revision, *, max_artifact_bytes, download=True):
        return PrecheckResult(repo=repo, revision=revision, ok=True, artifact_hash="hash-a", artifact_bytes=10)

    monkeypatch.setattr("validator.service.precheck_codec", fake_precheck)
    monkeypatch.setattr("validator.service._make_runner", lambda args: object())

    def fake_run_round(state, runner, challengers, provider, specs, **kwargs):
        if new_winner_entry_or_none is not None:
            state.winner_history = [new_winner_entry_or_none]
        return {}

    monkeypatch.setattr("validator.service.run_round", fake_run_round)

    real_corpus_dir = tmp_path / "live"
    real_corpus_dir.mkdir()
    (real_corpus_dir / "chunk_00_fineweb.txt").write_bytes(b"x" * 4096)
    (real_corpus_dir / "provenance.json").write_text(
        json.dumps([{"source": "fineweb", "chunk_ids": ["chunk_00_fineweb.txt"]}])
    )
    monkeypatch.setattr(
        "validator.service.resolve_live_corpus",
        lambda seed, token=None: StaticLocalProvider(real_corpus_dir),
    )

    state = ValidatorState()
    if winner_history is not None:
        state.commitments["incumbent-hk:inc/codec@rev0"] = CommitmentState(
            hotkey="incumbent-hk", repo="inc/codec", revision="rev0", block=1, artifact_hash="inc-hash", valid=True,
        )
        state.winner_history = winner_history

    return chain, state, _args()


def test_evaluate_round_publishes_winner_commitment_when_crown_changes(monkeypatch, tmp_path):
    bt_logging.set_info()
    new_winner = WinnerEntry("hk-a", "a/codec", _REV_A, 0.25, 100)
    chain, state, args = _setup_round(monkeypatch, tmp_path, winner_history=None, new_winner_entry_or_none=new_winner)

    _evaluate_round(args, state, chain, "saltval")

    assert len(chain.set_commitment_calls) == 1
    published = parse_winner_commitment(chain.set_commitment_calls[0])
    assert published.hotkey == "hk-a"
    assert published.repo == "a/codec"
    assert published.rev == _REV_A
    assert published.ratio == 0.25
    assert published.commit_block == 100
    assert published.scoring_version == SCORING_VERSION


def test_evaluate_round_does_not_publish_when_champion_unchanged(monkeypatch, tmp_path):
    bt_logging.set_info()
    incumbent = WinnerEntry("incumbent-hk", "inc/codec", "rev0", 0.5, 1)
    chain, state, args = _setup_round(
        monkeypatch, tmp_path, winner_history=[incumbent], new_winner_entry_or_none=None
    )

    _evaluate_round(args, state, chain, "saltval")

    assert chain.set_commitment_calls == []


def test_evaluate_round_survives_a_publish_failure(monkeypatch, tmp_path, caplog):
    # A chain-write failure (exception, or a response with success=False) must never crash
    # or delay the round -- this is best-effort observability, not scoring-critical.
    bt_logging.set_info()
    new_winner = WinnerEntry("hk-a", "a/codec", _REV_A, 0.25, 100)
    chain, state, args = _setup_round(monkeypatch, tmp_path, winner_history=None, new_winner_entry_or_none=new_winner)

    def boom(data):
        raise RuntimeError("chain unavailable")

    chain.set_commitment = boom

    block, round_metrics, _raw_commitments, _metagraph = _evaluate_round(args, state, chain, "saltval")

    assert round_metrics["winner/crown_changed"] is True
    assert "failed to publish winner commitment" in caplog.text
