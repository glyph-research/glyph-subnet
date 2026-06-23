from eval.scoring import (
    StreamResult,
    aggregate_ratio,
    score_codec,
    zstd_baseline_ratio,
)

FLOOR = 10 * 1024  # 10 KiB/s
BUDGET = 100.0


def result(sid, raw, comp, ok=True, c_secs=1.0, d_secs=1.0):
    return StreamResult(
        stream_id=sid,
        raw_bytes=raw,
        compressed_bytes=comp,
        roundtrip_ok=ok,
        compress_secs=c_secs,
        decompress_secs=d_secs,
        blob_hash="h",
    )


def test_aggregate_ratio():
    rs = [result("a", 1000, 400), result("b", 1000, 600)]
    assert aggregate_ratio(rs) == 0.5


def test_all_pass_is_valid():
    rs = [result("a", 1_000_000, 400_000), result("b", 1_000_000, 500_000)]
    score = score_codec(rs, floor_bps=FLOOR, budget_secs=BUDGET)
    assert score.valid is True
    assert score.ratio == 0.45


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
