import json
from pathlib import Path

import pytest
from bittensor.utils.btlogging import logging as bt_logging

from eval.corpus import StaticLocalProvider
from eval.evaluator import evaluate_artifact, paired_eval
from eval.runner import ArtifactRef, InsufficientGpuMemoryError, LocalSubprocessRunner, ResourceCaps
from core.constants import SCORING_VERSION
from core.state import ScoreState, ValidatorState, load_state, save_state
from eval.streams import sample_source_streams
from core.weights import WinnerEntry

REPO = Path(__file__).resolve().parents[1]
REFERENCE_CODEC = REPO / "reference_codec"
CORPUS = REPO / "samples" / "corpus"

FLOOR = 1.0  # bytes/sec, effectively disabled for tiny streams in CI
BUDGET = 60.0


def _broken_codec(directory: Path):
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entrypoints": {
                    "compress": ["python3", "c.py", "--input", "{input}", "--output", "{output}"],
                    "decompress": ["python3", "d.py", "--input", "{input}", "--output", "{output}"],
                },
            }
        )
    )
    (directory / "c.py").write_text(
        "import argparse;p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output')\n"
        "a=p.parse_args();open(a.output,'wb').write(open(a.input,'rb').read())\n"
    )
    (directory / "d.py").write_text(
        "import argparse;p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output')\n"
        "a=p.parse_args();open(a.output,'wb').write(b'wrong')\n"
    )


def test_paired_eval_reference_valid_broken_invalid(tmp_path):
    _broken_codec(tmp_path)
    provider = StaticLocalProvider(CORPUS)
    specs = sample_source_streams(42, 0, provider.total_bytes, stream_bytes=4096, streams=3)
    artifacts = [
        ("hk_ref", ArtifactRef("glyph/ref", "local", local_path=str(REFERENCE_CODEC))),
        ("hk_broken", ArtifactRef("t/broken", "local", local_path=str(tmp_path))),
    ]
    outcomes = paired_eval(
        LocalSubprocessRunner(), artifacts, provider, specs,
        caps=ResourceCaps(), floor_bps=FLOOR, budget_secs=BUDGET,
    )
    assert outcomes["hk_ref"].score.valid is True
    assert outcomes["hk_ref"].score.ratio < 1.0
    assert outcomes["hk_broken"].score.valid is False
    # burn-seed material is available from the valid codec's per-stream outputs
    assert len(outcomes["hk_ref"].burn_outputs()) == 3


# --- GPU-memory host fault must abort, not invalidate/exclude the codec (issue #105) ------


class _OutOfGpuMemoryRunner:
    def run_stream(self, artifact, stream, *, caps):
        raise InsufficientGpuMemoryError("insufficient free GPU memory: need ~22.0 GiB, 5.0 GiB free")


def test_evaluate_artifact_propagates_gpu_memory_fault_instead_of_invalidating(tmp_path):
    provider = StaticLocalProvider(CORPUS)
    specs = sample_source_streams(42, 0, provider.total_bytes, stream_bytes=4096, streams=1)
    artifact = ArtifactRef("hk/codec", "local", local_path=str(tmp_path))

    # Must raise (round aborts, this codec is never scored or one-shot-excluded) -- not
    # return an EvalOutcome with an invalid result, which is what happens for a genuine
    # entrypoint crash (see test_paired_eval_reference_valid_broken_invalid above).
    with pytest.raises(InsufficientGpuMemoryError):
        evaluate_artifact(
            _OutOfGpuMemoryRunner(), "hk_gpu_starved", artifact, provider, specs,
            caps=ResourceCaps(), floor_bps=FLOOR, budget_secs=BUDGET,
        )


def test_evaluate_artifact_logs_per_stream_progress_and_result(tmp_path, caplog):
    # issue #86: the log went completely silent for the full compress+decompress duration of
    # every stream -- up to ~450s each -- with nothing to distinguish "still working" from
    # "hung". Each stream must log before it starts and again (with ratio/roundtrip/timing)
    # once it finishes, or (with the failure reason) if the entrypoint crashes.
    bt_logging.set_info()
    _broken_codec(tmp_path)
    provider = StaticLocalProvider(CORPUS)
    specs = sample_source_streams(42, 0, provider.total_bytes, stream_bytes=4096, streams=3)
    artifacts = [
        ("hk_ref", ArtifactRef("glyph/ref", "local", local_path=str(REFERENCE_CODEC))),
        ("hk_broken", ArtifactRef("t/broken", "local", local_path=str(tmp_path))),
    ]
    paired_eval(
        LocalSubprocessRunner(), artifacts, provider, specs,
        caps=ResourceCaps(), floor_bps=FLOOR, budget_secs=BUDGET,
    )

    out = caplog.text
    assert "evaluating hk_ref: stream source-0 (1/3)..." in out
    assert "evaluating hk_ref: stream source-2 (3/3)..." in out
    assert "evaluating hk_ref: stream source-0 done -- ratio=" in out
    assert "roundtrip_ok=True" in out
    assert "compress_secs=" in out and "decompress_secs=" in out
    # the broken codec fails bit-exactness on its first stream, not a crash -- run_stream
    # returns a result rather than raising, so it logs the normal "done" line with
    # roundtrip_ok=False, not the RunnerError "failed:" branch.
    assert "evaluating hk_broken: stream source-0 (1/3)..." in out
    assert "roundtrip_ok=False" in out


def test_state_round_trip(tmp_path):
    state = ValidatorState()
    state.winner_history = [WinnerEntry("hkA", "a/c", "rev123456", ratio=0.42, commit_block=100)]
    state.scores["hkA:a/c@rev123456"] = ScoreState(
        hotkey="hkA", repo="a/c", revision="rev123456", ratio=0.42,
        roundtrip_ok=True, throughput_bps=50000.0, valid=True, commit_block=100,
        scoring_version=SCORING_VERSION,
    )
    state.last_round_outputs = [("s0", 123, "abc"), ("s1", 456, "def")]
    state.excluded_hotkeys = {"hk_loser"}
    path = tmp_path / "state" / "validator_state.json"
    save_state(path, state)

    reloaded = load_state(path)
    assert reloaded.winner_history[0].ratio == 0.42
    assert reloaded.winner_history[0].commit_block == 100
    assert reloaded.last_round_outputs[0] == ("s0", 123, "abc")
    assert reloaded.excluded_hotkeys == {"hk_loser"}


# --- codec reference links logged at eval start (issue #126) ----------------------------


def test_evaluate_artifact_logs_the_repo_link_once_before_the_first_stream(caplog):
    bt_logging.set_info()
    provider = StaticLocalProvider(CORPUS)
    specs = sample_source_streams(42, 0, provider.total_bytes, stream_bytes=4096, streams=2)
    artifact = ArtifactRef("glyph/ref", "local", local_path=str(REFERENCE_CODEC))

    evaluate_artifact(
        LocalSubprocessRunner(), "hk_ref", artifact, provider, specs,
        caps=ResourceCaps(), floor_bps=FLOOR, budget_secs=BUDGET,
    )

    out = caplog.text
    assert out.count("https://huggingface.co/glyph/ref/tree/local") == 1
    # identifying line comes before any per-stream output
    assert out.index("https://huggingface.co/glyph/ref") < out.index("stream source-0 (1/2)")
    assert "hub.docker.com" not in out  # reference codec pins no custom image


def test_evaluate_artifact_logs_the_docker_hub_link_for_a_hub_hosted_custom_image(tmp_path, caplog):
    bt_logging.set_info()
    _broken_codec(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["image"] = "someuser/somecodec@sha256:" + "a" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    provider = StaticLocalProvider(CORPUS)
    specs = sample_source_streams(42, 0, provider.total_bytes, stream_bytes=4096, streams=1)

    evaluate_artifact(
        LocalSubprocessRunner(), "hk_img", ArtifactRef("t/broken", "local", local_path=str(tmp_path)),
        provider, specs, caps=ResourceCaps(), floor_bps=FLOOR, budget_secs=BUDGET,
    )

    out = caplog.text
    assert "image=someuser/somecodec@sha256:" in out
    assert "https://hub.docker.com/r/someuser/somecodec" in out


def test_evaluate_artifact_logs_no_docker_hub_link_for_a_registry_hosted_image(tmp_path, caplog):
    # A ghcr.io/... (or any explicit-registry) reference is not a Docker Hub image; a
    # hub.docker.com link would point at the wrong place. The raw pinned ref still shows.
    bt_logging.set_info()
    _broken_codec(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["image"] = "ghcr.io/user/mycodec@sha256:" + "b" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    provider = StaticLocalProvider(CORPUS)
    specs = sample_source_streams(42, 0, provider.total_bytes, stream_bytes=4096, streams=1)

    evaluate_artifact(
        LocalSubprocessRunner(), "hk_img", ArtifactRef("t/broken", "local", local_path=str(tmp_path)),
        provider, specs, caps=ResourceCaps(), floor_bps=FLOOR, budget_secs=BUDGET,
    )

    out = caplog.text
    assert "image=ghcr.io/user/mycodec@sha256:" in out
    assert "hub.docker.com" not in out


def test_evaluate_artifact_link_logging_never_raises_without_a_manifest(tmp_path, caplog):
    # Display-only (issue #126): no local_path at all, or a local dir without a readable
    # manifest.json, must never raise -- the repo link still logs, evaluation is unaffected.
    bt_logging.set_info()
    provider = StaticLocalProvider(CORPUS)

    evaluate_artifact(  # no local_path on the ref (e.g. a remote-fetching runner)
        LocalSubprocessRunner(), "hk_remote", ArtifactRef("a/b", "deadbeef"),
        provider, [], caps=ResourceCaps(), floor_bps=FLOOR, budget_secs=BUDGET,
    )
    evaluate_artifact(  # local_path present but no manifest.json in it
        LocalSubprocessRunner(), "hk_nomanifest", ArtifactRef("c/d", "cafebabe", local_path=str(tmp_path)),
        provider, [], caps=ResourceCaps(), floor_bps=FLOOR, budget_secs=BUDGET,
    )

    out = caplog.text
    assert "https://huggingface.co/a/b/tree/deadbeef" in out
    assert "https://huggingface.co/c/d/tree/cafebabe" in out
