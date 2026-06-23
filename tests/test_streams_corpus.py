from pathlib import Path

from eval.corpus import StaticLocalProvider
from eval.streams import derive_seed, sample_streams

CORPUS = Path(__file__).resolve().parents[1] / "samples" / "corpus"


# --- seed + sampling determinism -------------------------------------------------

def test_derive_seed_is_stable():
    a = derive_seed("0xblockhash", "saltsalt", 7)
    b = derive_seed("0xblockhash", "saltsalt", 7)
    assert a == b
    assert a != derive_seed("0xblockhash", "saltsalt", 8)


def test_sample_streams_deterministic_and_bounded():
    total = 40000
    seed = derive_seed("0xabc", "mysalt", 1)
    specs_a = sample_streams(seed, total, stream_bytes=4096, streams=8)
    specs_b = sample_streams(seed, total, stream_bytes=4096, streams=8)
    assert specs_a == specs_b
    assert len(specs_a) == 8
    for spec in specs_a:
        assert spec.length == 4096
        assert 0 <= spec.offset <= total - spec.length


def test_different_seed_changes_selection():
    total = 40000
    s1 = sample_streams(derive_seed("h", "salt", 1), total, stream_bytes=4096, streams=8)
    s2 = sample_streams(derive_seed("h", "salt", 2), total, stream_bytes=4096, streams=8)
    assert [x.offset for x in s1] != [x.offset for x in s2]


def test_stream_larger_than_corpus_clamps():
    specs = sample_streams(123, 1000, stream_bytes=8192, streams=4)
    assert all(s.length == 1000 and s.offset == 0 for s in specs)


# --- static local provider -------------------------------------------------------

def test_provider_total_and_manifest_stable():
    provider = StaticLocalProvider(CORPUS)
    assert provider.total_bytes == sum(p.stat().st_size for p in CORPUS.iterdir())
    assert provider.manifest().manifest_hash() == StaticLocalProvider(CORPUS).manifest().manifest_hash()


def test_read_range_matches_concatenation():
    provider = StaticLocalProvider(CORPUS)
    full = provider.read_range(0, provider.total_bytes)
    # spans the file boundary between the two sorted files
    mid = provider.read_range(15000, 10000)
    assert mid == full[15000:25000]
    assert len(mid) == 10000


def test_materialize_equals_read_range():
    provider = StaticLocalProvider(CORPUS)
    seed = derive_seed("blk", "salt", 3)
    for spec in sample_streams(seed, provider.total_bytes, stream_bytes=2048, streams=8):
        assert provider.materialize(spec) == provider.read_range(spec.offset, spec.length)
        assert len(provider.materialize(spec)) == spec.length
