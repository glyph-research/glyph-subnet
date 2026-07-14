"""issue #71: a real round must (a) refuse --corpus-dir outright and (b) build its corpus
live via eval.live_corpus.resolve_live_corpus, keyed by this round's chain beacon -- never a
shared owner-published file.
"""

import json

import pytest
from bittensor.utils.btlogging import logging as bt_logging

from core.commitments import CodecCommitment, serialize_commitment
from core.state import CommitmentState, ValidatorState
from core.weights import WinnerEntry
from validation.precheck import PrecheckResult
from validator.service import _evaluate_round
from eval.corpus import StaticLocalProvider
from eval.streams import derive_seed


class FakeChain:
    def __init__(self, block: int, block_hash: str, raw_commitments: dict):
        self._block = block
        self._block_hash = block_hash
        self._raw_commitments = raw_commitments

    def current_block(self) -> int:
        return self._block

    def block_hash(self, block: int) -> str:
        return self._block_hash

    def get_all_commitments(self) -> dict:
        return self._raw_commitments


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


def test_evaluate_round_rejects_corpus_dir_on_a_real_round():
    # The guard must fire before ever touching the chain -- a fake chain with no methods
    # implemented proves this isn't reached only after some other real work.
    state = ValidatorState()
    with pytest.raises(SystemExit) as exc:
        _evaluate_round(_args(corpus_dir="./some/dir"), state, chain=object(), salt="salt")
    assert "--offline-demo" in str(exc.value)


def test_evaluate_round_logs_round_start_before_run_round(monkeypatch, tmp_path, caplog):
    # issue #81: an operator watching the log must see which incumbent/challengers are being
    # evaluated *before* the (potentially many-minute) evaluation runs, not only a post-hoc
    # summary once it's already done.
    bt_logging.set_info()
    block = 12345
    block_hash = "0xbeacon"
    salt = "saltval"
    raw_commitments = {"hk-a": serialize_commitment(CodecCommitment(repo="a/codec", rev="a" * 40))}
    chain = FakeChain(block, block_hash, raw_commitments)

    def fake_precheck(repo, revision, *, max_artifact_bytes, download=True):
        return PrecheckResult(repo=repo, revision=revision, ok=True, artifact_hash="hash-a", artifact_bytes=10)

    monkeypatch.setattr("validator.service.precheck_codec", fake_precheck)
    monkeypatch.setattr("validator.service._make_runner", lambda args: object())

    logged_before_run_round = {}

    def fake_run_round(state, runner, challengers, provider, specs, **kwargs):
        logged_before_run_round["text"] = caplog.text
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
    state.commitments["incumbent-hk:inc/codec@rev0"] = CommitmentState(
        hotkey="incumbent-hk", repo="inc/codec", revision="rev0", block=1, artifact_hash="inc-hash", valid=True
    )
    state.winner_history = [WinnerEntry("incumbent-hk", "inc/codec", "rev0", 0.5, 1)]
    _evaluate_round(_args(), state, chain, salt)

    assert "round: evaluating incumbent=incumbent-hk" in logged_before_run_round["text"]
    assert "hk-a" in logged_before_run_round["text"]


def test_evaluate_round_builds_corpus_via_resolve_live_corpus(monkeypatch, tmp_path):
    block = 12345
    block_hash = "0xbeacon"
    salt = "saltval"
    raw_commitments = {"hk-a": serialize_commitment(CodecCommitment(repo="a/codec", rev="a" * 40))}
    chain = FakeChain(block, block_hash, raw_commitments)

    def fake_precheck(repo, revision, *, max_artifact_bytes, download=True):
        return PrecheckResult(repo=repo, revision=revision, ok=True, artifact_hash="hash-a", artifact_bytes=10)

    monkeypatch.setattr("validator.service.precheck_codec", fake_precheck)
    monkeypatch.setattr("validator.service._make_runner", lambda args: object())

    captured_outcomes_call = {}

    def fake_run_round(state, runner, challengers, provider, specs, **kwargs):
        captured_outcomes_call["provider"] = provider
        captured_outcomes_call["challengers"] = challengers
        return {}

    monkeypatch.setattr("validator.service.run_round", fake_run_round)

    captured_resolve_call = {}
    real_corpus_dir = tmp_path / "live"
    real_corpus_dir.mkdir()
    (real_corpus_dir / "chunk_00_fineweb.txt").write_bytes(b"x" * 4096)
    (real_corpus_dir / "provenance.json").write_text(
        json.dumps([{"source": "fineweb", "chunk_ids": ["chunk_00_fineweb.txt"]}])
    )

    def fake_resolve_live_corpus(seed, token=None):
        captured_resolve_call["seed"] = seed
        captured_resolve_call["token"] = token
        return StaticLocalProvider(real_corpus_dir)

    monkeypatch.setattr("validator.service.resolve_live_corpus", fake_resolve_live_corpus)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    state = ValidatorState()
    _evaluate_round(_args(), state, chain, salt)

    expected_seed = derive_seed(block_hash, salt, block)
    assert captured_resolve_call["seed"] == str(expected_seed)
    assert captured_resolve_call["token"] is None  # unset HF_TOKEN -> anonymous, unaffected
    assert len(captured_outcomes_call["challengers"]) == 1
    assert captured_outcomes_call["challengers"][0].hotkey == "hk-a"


# --- HF_TOKEN threaded through to resolve_live_corpus (issue #108) ---------------------


def test_evaluate_round_passes_hf_token_env_var_to_resolve_live_corpus(monkeypatch, tmp_path):
    block = 12345
    block_hash = "0xbeacon"
    salt = "saltval"
    raw_commitments = {"hk-a": serialize_commitment(CodecCommitment(repo="a/codec", rev="a" * 40))}
    chain = FakeChain(block, block_hash, raw_commitments)

    def fake_precheck(repo, revision, *, max_artifact_bytes, download=True):
        return PrecheckResult(repo=repo, revision=revision, ok=True, artifact_hash="hash-a", artifact_bytes=10)

    monkeypatch.setattr("validator.service.precheck_codec", fake_precheck)
    monkeypatch.setattr("validator.service._make_runner", lambda args: object())
    monkeypatch.setattr("validator.service.run_round", lambda state, runner, challengers, provider, specs, **kw: {})

    real_corpus_dir = tmp_path / "live"
    real_corpus_dir.mkdir()
    (real_corpus_dir / "chunk_00_fineweb.txt").write_bytes(b"x" * 4096)
    (real_corpus_dir / "provenance.json").write_text(
        json.dumps([{"source": "fineweb", "chunk_ids": ["chunk_00_fineweb.txt"]}])
    )

    captured = {}

    def fake_resolve_live_corpus(seed, token=None):
        captured["token"] = token
        return StaticLocalProvider(real_corpus_dir)

    monkeypatch.setattr("validator.service.resolve_live_corpus", fake_resolve_live_corpus)
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken123")

    state = ValidatorState()
    _evaluate_round(_args(), state, chain, salt)

    assert captured["token"] == "hf_faketoken123"
