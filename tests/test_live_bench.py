"""issue #139: live-data benchmark stream -- benchmark-only like enwik9, prefetched in the
between-rounds window, never allowed to block/delay/fail a round, retained up to 1 GiB."""

import threading
import time

from core.wandb_logger import build_round_metrics
from eval.corpus import StaticLocalProvider
from eval.evaluator import EvalOutcome
from eval.live_bench import (
    LIVE_SOURCE,
    LivePrefetcher,
    LiveSnapshotStore,
    SnapshotAppendedProvider,
    fetch_live_text,
    live_benchmark_spec,
)
from eval.scoring import CodecScore, StreamResult, score_codec
from validator.service import _append_live_benchmark_stream


# --- fetch: recentchanges extracts, deduped, complete-or-nothing ------------------------


def _page(title, size):
    return {"title": title, "extract": "x" * size}


def test_fetch_live_text_accumulates_until_min_bytes_and_dedups_titles():
    calls = []

    def fake_api(params):
        calls.append(params)
        # The same article edited twice in the window must contribute once.
        return {
            "query": {"pages": {
                "1": _page(f"Article {len(calls)}", 600),
                "2": _page("Repeat", 600),
            }},
            "continue": {"grccontinue": f"c{len(calls)}"},
        }

    data = fetch_live_text(2000, deadline_secs=30, api_get=fake_api, politeness_secs=0)

    assert data is not None
    assert len(data) >= 2000
    # 1 unique + the deduped repeat on the first call, then 1 unique per later call.
    assert data.count(b"\n\n") == len(calls) + 1


def test_fetch_live_text_returns_none_when_the_deadline_passes_first():
    def stingy_api(params):
        return {"query": {"pages": {"1": _page("Tiny", 250)}}, "continue": {}}

    assert fetch_live_text(10**7, deadline_secs=0.2, api_get=stingy_api, politeness_secs=0) is None


def test_fetch_live_text_skips_short_extracts_and_survives_api_errors():
    calls = []

    def flaky_api(params):
        calls.append(params)
        if len(calls) == 1:
            raise ConnectionError("api down")
        return {
            "query": {"pages": {
                "1": _page("Stub", 50),  # < 200 chars: skipped
                "2": _page(f"Real {len(calls)}", 800),
            }},
            "continue": {},
        }

    data = fetch_live_text(1500, deadline_secs=30, api_get=flaky_api, politeness_secs=0)

    assert data is not None and b"x" * 800 in data


# --- snapshot store: complete-only visibility + 1GiB oldest-first retention -------------


def test_store_latest_sees_only_complete_snapshots(tmp_path):
    store = LiveSnapshotStore(tmp_path)
    assert store.latest() is None

    (tmp_path / "300.txt.part").write_bytes(b"incomplete")  # an in-flight write
    assert store.latest() is None

    store.save(100, b"older")
    store.save(200, b"newer")
    path, block = store.latest()
    assert block == 200
    assert path.read_bytes() == b"newer"


def test_store_prunes_oldest_first_down_to_the_retention_cap(tmp_path):
    store = LiveSnapshotStore(tmp_path, retention_bytes=25)
    store.save(1, b"0123456789")
    store.save(2, b"0123456789")
    store.save(3, b"0123456789")  # 30 bytes total -> oldest goes

    remaining = sorted(int(p.stem) for p in tmp_path.glob("*.txt"))
    assert remaining == [2, 3]

    store.save(4, b"x" * 100)  # single oversized snapshot: newest is always kept
    remaining = sorted(int(p.stem) for p in tmp_path.glob("*.txt"))
    assert remaining == [4]


# --- prefetcher: non-blocking, abandoned-not-awaited, complete-or-nothing ---------------


def test_prefetcher_writes_a_complete_snapshot_in_the_background(tmp_path):
    store = LiveSnapshotStore(tmp_path)
    prefetcher = LivePrefetcher(store, min_bytes=4, fetch=lambda n: b"livedata")

    prefetcher.start(123)
    prefetcher._thread.join(timeout=5)

    path, block = store.latest()
    assert block == 123
    assert path.read_bytes() == b"livedata"


def test_prefetcher_writes_nothing_on_a_short_fetch(tmp_path):
    store = LiveSnapshotStore(tmp_path)
    prefetcher = LivePrefetcher(store, min_bytes=4, fetch=lambda n: None)

    prefetcher.start(123)
    prefetcher._thread.join(timeout=5)

    assert store.latest() is None
    assert list(tmp_path.iterdir()) == []  # no partial files either


def test_prefetcher_start_never_blocks_and_a_hung_fetch_is_abandoned(tmp_path):
    release = threading.Event()

    def hung_fetch(n):
        release.wait(timeout=10)
        return None

    store = LiveSnapshotStore(tmp_path)
    prefetcher = LivePrefetcher(store, min_bytes=4, fetch=hung_fetch)

    started = time.monotonic()
    prefetcher.start(1)
    first_thread = prefetcher._thread
    prefetcher.start(2)  # previous fetch still hung: skipped, not awaited
    assert time.monotonic() - started < 1.0  # both calls returned immediately
    assert prefetcher._thread is first_thread

    release.set()
    first_thread.join(timeout=5)


# --- round wiring: latest snapshot becomes one benchmark-only live-0 stream -------------


def _args(state_dir):
    return type("Args", (), {"state_dir": state_dir})()


def test_round_uses_the_latest_complete_snapshot_as_a_benchmark_stream(tmp_path, caplog):
    from bittensor.utils.btlogging import logging as bt_logging

    bt_logging.set_info()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "chunk_00.txt").write_bytes(b"c" * 100)
    base = StaticLocalProvider(corpus)
    store = LiveSnapshotStore(tmp_path / "state" / "live_data")
    store.save(50, b"stale snapshot")
    store.save(90, b"fresh live bytes")

    provider, specs = _append_live_benchmark_stream(_args(str(tmp_path / "state")), base, [])

    assert len(specs) == 1
    spec = specs[0]
    assert (spec.stream_id, spec.source, spec.scored) == ("live-0", LIVE_SOURCE, False)
    assert provider.materialize(spec) == b"fresh live bytes"  # latest, not the stale one
    assert provider.stream_source(spec) is None  # inline bytes; no remote range
    assert provider.read_range(0, 100) == b"c" * 100  # base corpus untouched
    assert "using snapshot from block 90" in caplog.text


def test_round_proceeds_without_a_live_stream_when_no_snapshot_exists(tmp_path, caplog):
    from bittensor.utils.btlogging import logging as bt_logging

    bt_logging.set_info()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "chunk_00.txt").write_bytes(b"c" * 100)
    base = StaticLocalProvider(corpus)

    provider, specs = _append_live_benchmark_stream(_args(str(tmp_path / "state")), base, [])

    assert provider is base and specs == []
    assert "skipping the live stream this round" in caplog.text

    # And with no state dir at all (offline/test args): silently unchanged.
    provider, specs = _append_live_benchmark_stream(_args(None), base, [])
    assert provider is base and specs == []


def test_snapshot_appended_provider_routes_ranges_and_source_lookup(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "chunk_00.txt").write_bytes(b"base-corpus-bytes")
    base = StaticLocalProvider(corpus)
    provider = SnapshotAppendedProvider(base, b"LIVE")

    assert provider.total_bytes == base.total_bytes + 4
    assert provider.read_range(base.total_bytes, 4) == b"LIVE"
    assert provider.read_range(base.total_bytes - 5, 9) == b"bytesLIVE"  # boundary crossing
    assert provider.source_range(LIVE_SOURCE) == (base.total_bytes, 4)

    spec = live_benchmark_spec(base, b"LIVE")
    assert spec.offset == base.total_bytes and spec.length == 4


# --- never scored: no effect on scored_ratio/promotion; own wandb field -----------------


def _result(stream_id, source, scored, ratio_num=500, raw=1000):
    return StreamResult(
        stream_id=stream_id, raw_bytes=raw, compressed_bytes=ratio_num, roundtrip_ok=True,
        compress_secs=1.0, decompress_secs=1.0, blob_hash="h", source=source, scored=scored,
    )


def test_live_stream_never_contributes_to_scored_ratio(caplog):
    with_live = score_codec(
        [
            _result("fineweb-edu-0", "fineweb-edu", True, ratio_num=400),
            _result("live-0", LIVE_SOURCE, False, ratio_num=999, raw=1000),
        ],
        floor_bps=1.0, budget_secs=60.0,
    )
    without_live = score_codec(
        [_result("fineweb-edu-0", "fineweb-edu", True, ratio_num=400)],
        floor_bps=1.0, budget_secs=60.0,
    )
    assert with_live.ratio == without_live.ratio  # pinned exactly like enwik9


def test_round_metrics_report_live_and_enwik9_ratios_separately():
    outcome = EvalOutcome(
        hotkey="hkA",
        score=CodecScore(valid=True, ratio=0.4, throughput_bps_min=20_000, reasons=[]),
        results=[
            _result("fineweb-edu-0", "fineweb-edu", True, ratio_num=400),
            _result("enwik9-0", "enwik9", False, ratio_num=300),
            _result("live-0", LIVE_SOURCE, False, ratio_num=800),
        ],
    )
    metrics = build_round_metrics(
        block=1, baseline_ratio=0.6, num_challengers=1, outcomes={"hkA": outcome},
        excluded_hotkeys_count=0, commit_phase_seen_count=0, winner_hotkey=None,
        winner_ratio=None, crown_changed=False,
    )
    assert metrics["challenger/hkA/enwik9_ratio"] == 0.3
    assert metrics["challenger/hkA/live_ratio"] == 0.8  # the generalization gap, visible
