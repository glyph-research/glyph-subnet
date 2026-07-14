from eval.scoring import (
    StreamResult,
    aggregate_ratio,
    score_codec,
    source_ratio_breakdown,
    scored_ratio,
    zstd_baseline_ratio,
)
import pytest

FLOOR = 10 * 1024  # 10 KiB/s
BUDGET = 100.0


def result(sid, raw, comp, ok=True, c_secs=1.0, d_secs=1.0, source=None, scored=True):
    return StreamResult(
        stream_id=sid,
        raw_bytes=raw,
        compressed_bytes=comp,
        roundtrip_ok=ok,
        compress_secs=c_secs,
        decompress_secs=d_secs,
        blob_hash="h",
        source=source,
        scored=scored,
    )


def test_aggregate_ratio():
    rs = [result("a", 1000, 400), result("b", 1000, 600)]
    assert aggregate_ratio(rs) == 0.5


def test_all_pass_is_valid():
    rs = [result("a", 1_000_000, 400_000), result("b", 1_000_000, 500_000)]
    score = score_codec(rs, floor_bps=FLOOR, budget_secs=BUDGET)
    assert score.valid is True
    assert score.ratio == 0.45


def test_scored_ratio_is_the_flat_mean_of_stream_ratios():
    # issue #112: every scored stream counts equally, regardless of source. On a symmetric
    # 2+2 fixture the flat mean coincides with the old mean-of-source-means (0.55) -- the
    # asymmetric regression test below is what actually pins the rule change.
    rs = [
        result("fineweb-edu-0", 10_000, 2_000, source="fineweb-edu"),
        result("fineweb-edu-1", 10_000, 4_000, source="fineweb-edu"),
        result("pile-0", 1_000, 900, source="pile"),
        result("pile-1", 1_000, 700, source="pile"),
    ]
    assert source_ratio_breakdown(rs) == pytest.approx({"fineweb-edu": 0.3, "pile": 0.8})
    assert scored_ratio(rs) == pytest.approx(0.55)
    assert score_codec(rs, floor_bps=FLOOR, budget_secs=BUDGET).ratio == pytest.approx(0.55)


def test_scored_ratio_asymmetric_mix_is_not_source_weighted():
    # issue #112's regression guard: with the 2x fineweb-edu / 1x pile mix, the old
    # mean-of-per-source-means would give (mean(0.2, 0.4) + 0.9) / 2 = 0.6 -- pile's single
    # stream counting as much as fineweb-edu's two combined. The flat per-stream mean is
    # (0.2 + 0.4 + 0.9) / 3 = 0.5. The two must differ on this fixture, or the aggregation
    # has silently reverted to source-weighted averaging.
    rs = [
        result("fineweb-edu-0", 10_000, 2_000, source="fineweb-edu"),
        result("fineweb-edu-1", 10_000, 4_000, source="fineweb-edu"),
        result("pile-0", 1_000, 900, source="pile"),
    ]
    breakdown = source_ratio_breakdown(rs)
    old_mean_of_means = sum(breakdown.values()) / len(breakdown)
    assert old_mean_of_means == pytest.approx(0.6)
    assert scored_ratio(rs) == pytest.approx(0.5)
    assert scored_ratio(rs) != pytest.approx(old_mean_of_means)
    # source_ratio_breakdown itself stays intact for wandb per-source visibility.
    assert breakdown == pytest.approx({"fineweb-edu": 0.3, "pile": 0.9})


def test_benchmark_only_streams_do_not_affect_score_or_validity():
    rs = [
        result("fineweb-0", 1_000_000, 300_000, source="fineweb"),
        result("pile-0", 1_000_000, 500_000, source="pile"),
        result(
            "enwik9-0",
            1_000_000,
            990_000,
            ok=False,
            d_secs=10_000.0,
            source="enwik9",
            scored=False,
        ),
    ]
    score = score_codec(rs, floor_bps=FLOOR, budget_secs=BUDGET)
    assert score.valid is True
    assert score.ratio == 0.4


def test_any_roundtrip_failure_invalidates():
    rs = [result("a", 1_000_000, 400_000), result("b", 1_000_000, 500_000, ok=False)]
    score = score_codec(rs, floor_bps=FLOOR, budget_secs=BUDGET)
    assert score.valid is False
    assert "round-trip failed" in score.reasons[0]


def test_below_throughput_floor_invalidates():
    # 1 MiB decompressed in 1000s -> ~1 KiB/s, well below the 10 KiB/s floor.
    rs = [result("a", 1_048_576, 400_000, d_secs=1000.0)]
    score = score_codec(rs, floor_bps=FLOOR, budget_secs=BUDGET)
    assert score.valid is False
    assert any("throughput" in r for r in score.reasons)


def test_compress_over_budget_invalidates():
    rs = [result("a", 1_000_000, 400_000, c_secs=BUDGET + 1)]
    score = score_codec(rs, floor_bps=FLOOR, budget_secs=BUDGET)
    assert score.valid is False
    assert any("over budget" in r for r in score.reasons)


def test_no_streams_invalid():
    score = score_codec([], floor_bps=FLOOR, budget_secs=BUDGET)
    assert score.valid is False


def test_zstd_baseline_compresses_text():
    streams = [b"the quick brown fox " * 5000, b"lorem ipsum dolor sit amet " * 5000]
    ratio = zstd_baseline_ratio(streams)
    assert 0.0 < ratio < 1.0
