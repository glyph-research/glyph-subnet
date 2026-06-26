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


# --- per-source eval: source_range + sample_source_streams (issue #10) -----------

import json

from eval.streams import sample_source_streams


def test_sample_source_streams_confined_and_fresh():
    start, span = 1000, 12000
    seed = derive_seed("0xblk", "salt", 3)
    a = sample_source_streams(seed, start, span, stream_bytes=4096, streams=2)
    b = sample_source_streams(seed, start, span, stream_bytes=4096, streams=2)
    assert a == b and len(a) == 2
    for spec in a:
        assert spec.length == 4096
        assert start <= spec.offset and spec.offset + spec.length <= start + span
    # fresh per round (different seed -> different windows)
    other = sample_source_streams(derive_seed("0xblk", "salt", 4), start, span, stream_bytes=4096, streams=2)
    assert [s.offset for s in a] != [s.offset for s in other]


def test_sample_source_streams_clamps_to_span():
    specs = sample_source_streams(7, 500, 1000, stream_bytes=8192, streams=2)
    assert all(s.length == 1000 and s.offset == 500 for s in specs)


def _write_corpus(tmp_path, sizes_sources):
    # sizes_sources: list of (filename, size, source); writes files + provenance.json
    prov = {}
    for name, size, source in sizes_sources:
        (tmp_path / name).write_bytes(b"x" * size)
        prov.setdefault(source, []).append(name)
    (tmp_path / "provenance.json").write_text(
        json.dumps([{"source": s, "chunk_ids": ids} for s, ids in prov.items()])
    )


def test_source_range_from_provenance(tmp_path):
    _write_corpus(tmp_path, [
        ("chunk_00_fineweb.txt", 100, "fineweb"),
        ("chunk_01_fineweb.txt", 200, "fineweb"),
        ("chunk_02_enwik9.txt", 50, "enwik9"),
    ])
    p = StaticLocalProvider(tmp_path)
    assert p.source_range("fineweb") == (0, 300)   # two contiguous fineweb chunks
    assert p.source_range("enwik9") == (300, 50)
    assert p.source_range("pile") is None           # absent source
    # the resolved range materializes to exactly the source's bytes
    start, span = p.source_range("fineweb")
    assert len(p.read_range(start, span)) == 300


def test_source_range_none_without_provenance(tmp_path):
    (tmp_path / "data.txt").write_bytes(b"y" * 64)
    assert StaticLocalProvider(tmp_path).source_range("fineweb") is None


def test_source_range_non_contiguous_is_none(tmp_path):
    # fineweb chunks split by a pile chunk in sorted order -> not a single span
    _write_corpus(tmp_path, [
        ("chunk_00_fineweb.txt", 100, "fineweb"),
        ("chunk_01_pile.txt", 100, "pile"),
        ("chunk_02_fineweb.txt", 100, "fineweb"),
    ])
    assert StaticLocalProvider(tmp_path).source_range("fineweb") is None
