"""issue #41: wandb logging is pure observability -- a mocked/offline run must expose the
expected metric keys, enable console capture, and never let a wandb failure propagate."""

from __future__ import annotations

import argparse
import sys
from unittest.mock import MagicMock

import pytest

from core.wandb_logger import (
    WandbLogger,
    _NullWandbLogger,
    build_round_metrics,
    build_weights_metrics,
    make_wandb_logger,
)
from eval.evaluator import EvalOutcome
from eval.scoring import CodecScore, StreamResult


def _stream(source, ratio_num, ratio_den=1000, scored=True):
    return StreamResult(
        stream_id=f"{source}-0",
        raw_bytes=ratio_den,
        compressed_bytes=ratio_num,
        roundtrip_ok=True,
        compress_secs=1.0,
        decompress_secs=1.0,
        blob_hash="h",
        source=source,
        scored=scored,
    )


def _outcome(hotkey, ratio):
    results = [_stream("fineweb-edu", int(ratio * 1000)), _stream("pile", int(ratio * 1000))]
    score = CodecScore(valid=True, ratio=ratio, throughput_bps_min=20_000, reasons=[])
    return EvalOutcome(hotkey=hotkey, score=score, results=results)


@pytest.fixture
def fake_wandb(monkeypatch):
    """Install a mock ``wandb`` module so WandbLogger's lazy ``import wandb`` picks it up."""

    fake = MagicMock()
    fake.init.return_value = MagicMock(get_url=MagicMock(return_value="https://wandb.ai/x/y/runs/z"))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def test_null_logger_never_touches_wandb(fake_wandb):
    logger = _NullWandbLogger()
    assert logger.enabled is False
    logger.log({"a": 1})
    logger.finish()
    fake_wandb.init.assert_not_called()
    fake_wandb.log.assert_not_called()


def test_wandb_off_flag_yields_null_logger(fake_wandb):
    args = argparse.Namespace(wandb_off=True)
    logger = make_wandb_logger(args)
    assert isinstance(logger, _NullWandbLogger)
    fake_wandb.init.assert_not_called()


def test_start_run_enables_console_capture_and_anonymous_fallback(fake_wandb, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    WandbLogger(enabled=True, project="glyph-subnet", offline=False)
    fake_wandb.init.assert_called_once()
    kwargs = fake_wandb.init.call_args.kwargs
    assert kwargs["anonymous"] == "allow"
    assert kwargs["mode"] == "online"
    fake_wandb.Settings.assert_called_once_with(console="wrap")


def test_start_run_uses_real_key_when_present(fake_wandb, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "sekrit")
    WandbLogger(enabled=True)
    kwargs = fake_wandb.init.call_args.kwargs
    assert kwargs["anonymous"] is None


def test_offline_mode_passed_through(fake_wandb):
    WandbLogger(enabled=True, offline=True)
    assert fake_wandb.init.call_args.kwargs["mode"] == "offline"


def test_make_wandb_logger_defaults_to_glyph_research_org_text_compression(fake_wandb):
    # issue #102: an args object that never set wandb_project/wandb_entity at all (e.g.
    # constructed programmatically rather than via build_parser) must still land the run in
    # the glyph-research-org/text-compression team project, not an arbitrary/personal one.
    args = argparse.Namespace(wandb_off=False)
    logger = make_wandb_logger(args)
    assert logger._project == "text-compression"
    assert logger._entity == "glyph-research-org"


# --- run name defaults to on-chain identity (issue #102 follow-up) ----------------


def test_make_wandb_logger_uses_the_resolved_identity_name(fake_wandb):
    # Multiple validators sharing the glyph-research-org/text-compression project must be
    # distinguishable at a glance instead of wandb's random auto-generated name. The caller
    # (validator.service) resolves the actual on-chain-identity-or-hotkey fallback and passes
    # it straight through.
    args = argparse.Namespace(wandb_off=False)
    logger = make_wandb_logger(args, identity_name="5F...somehotkey")
    assert logger._name == "5F...somehotkey"


def test_wandb_name_flag_overrides_identity_name(fake_wandb):
    args = argparse.Namespace(wandb_off=False, wandb_name="my-custom-run")
    logger = make_wandb_logger(args, identity_name="5F...somehotkey")
    assert logger._name == "my-custom-run"


def test_make_wandb_logger_leaves_name_unset_without_identity_name(fake_wandb):
    # No explicit override and no resolved identity (e.g. programmatically-built args, or
    # chain lookup failed) -> name stays None, wandb picks its own.
    args = argparse.Namespace(wandb_off=False)
    logger = make_wandb_logger(args)
    assert logger._name is None


def test_start_run_passes_name_to_wandb_init(fake_wandb):
    WandbLogger(enabled=True, name="w-h")
    assert fake_wandb.init.call_args.kwargs["name"] == "w-h"


def test_build_round_metrics_has_expected_keys_for_simulated_round():
    outcomes = {"hkA": _outcome("hkA", 0.42), "hkB": _outcome("hkB", 0.5)}
    metrics = build_round_metrics(
        block=123,
        baseline_ratio=0.6,
        num_challengers=2,
        outcomes=outcomes,
        excluded_hotkeys_count=1,
        commit_phase_seen_count=3,
        winner_hotkey="hkA",
        winner_ratio=0.42,
        crown_changed=True,
    )
    for key in [
        "round/block",
        "round/baseline_ratio",
        "round/num_challengers",
        "round/excluded_hotkeys_count",
        "round/commit_phase_seen_count",
        "winner/hotkey",
        "winner/ratio",
        "winner/crown_changed",
    ]:
        assert key in metrics
    for hotkey in outcomes:
        prefix = f"challenger/{hotkey}"
        assert f"{prefix}/ratio" in metrics
        assert f"{prefix}/valid" in metrics
        assert f"{prefix}/roundtrip_ok" in metrics
        assert f"{prefix}/throughput_bps_min" in metrics
        assert f"{prefix}/beats_baseline" in metrics
        assert f"{prefix}/fineweb_edu_ratio" in metrics
        assert f"{prefix}/pile_ratio" in metrics
    assert metrics["winner/hotkey"] == "hkA"
    assert metrics["winner/crown_changed"] is True


def test_build_weights_metrics_has_expected_keys():
    metrics = build_weights_metrics(
        block=10, tempo=360, is_burn_tempo=False, uids=[0, 1, 2], weights=[0.0, 1.0, 0.0]
    )
    assert metrics == {
        "weights/block": 10,
        "weights/tempo": 360,
        "weights/is_burn_tempo": False,
        "weights/nonzero_count": 1,
        "weights/nonzero": "[(1, 1.0)]",
    }


def test_log_forwards_metrics_to_wandb(fake_wandb):
    logger = WandbLogger(enabled=True, restart_interval_hours=0)
    metrics = build_round_metrics(
        block=1, baseline_ratio=0.6, num_challengers=0, outcomes={}, excluded_hotkeys_count=0,
        commit_phase_seen_count=0, winner_hotkey=None, winner_ratio=None, crown_changed=False,
    )
    logger.log(metrics)
    fake_wandb.log.assert_called_once_with(metrics)


def test_init_failure_disables_logging_without_raising(fake_wandb):
    fake_wandb.init.side_effect = RuntimeError("network down")
    logger = WandbLogger(enabled=True)  # must not raise
    assert logger.enabled is False
    logger.log({"a": 1})  # still must not raise, and stays a no-op
    fake_wandb.log.assert_not_called()


def test_log_failure_disables_logging_without_raising(fake_wandb):
    logger = WandbLogger(enabled=True)
    fake_wandb.log.side_effect = RuntimeError("connection reset")
    logger.log({"a": 1})  # must not raise
    assert logger.enabled is False
    logger.log({"b": 2})  # now a no-op; must not raise or call wandb.log again
    fake_wandb.log.assert_called_once()


def test_finish_is_noop_when_disabled_or_no_run(fake_wandb):
    logger = WandbLogger(enabled=False)
    logger.finish()  # no-op, no exception
    fake_wandb.init.assert_not_called()
