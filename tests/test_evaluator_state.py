import json
from pathlib import Path

from eval.corpus import StaticLocalProvider
from eval.evaluator import paired_eval
from eval.runner import ArtifactRef, LocalSubprocessRunner, ResourceCaps
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


def test_state_round_trip(tmp_path):
    state = ValidatorState()
    state.winner_history = [WinnerEntry("hkA", "a/c", "rev123456", ratio=0.42, commit_block=100)]
    state.scores["hkA:a/c@rev123456"] = ScoreState(
        hotkey="hkA", repo="a/c", revision="rev123456", ratio=0.42,
        roundtrip_ok=True, throughput_bps=50000.0, valid=True, commit_block=100,
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
