import json

import pytest

from eval.corpus import StaticLocalProvider
from eval.live_corpus import (
    Source,
    fetch_source_chunks,
    resolve_live_corpus,
    write_mixed_corpus,
    _extract_text,
    _skip_for_seed,
)

# Two small sources standing in for the FineWeb/Pile/enwik9 mix. Records are injected, so
# these tests never touch the network or the real `datasets` library.
_SOURCES = [
    Source("alpha", "fake/alpha", "cfgA", "train", "text", 2),
    Source("beta", "fake/beta", None, "train", "text", 1),
]


def _records(prefix: str, n: int) -> list[dict]:
    return [{"text": f"{prefix}-{i:04d} " + "word " * 80} for i in range(n)]


def _records_by_source(skip_cap: int) -> dict:
    # Enough records after the seed-derived skip to fill every source's chunks.
    return {
        "alpha": _records("alpha", skip_cap + 200),
        "beta": _records("beta", skip_cap + 200),
    }


def test_skip_for_seed_is_deterministic_and_seed_dependent():
    a = _skip_for_seed("fineweb", "0xbeacon", 100_000)
    assert a == _skip_for_seed("fineweb", "0xbeacon", 100_000)
    # Different seed -> different slice; different source -> different slice.
    assert a != _skip_for_seed("fineweb", "0xother", 100_000)
    assert a != _skip_for_seed("pile", "0xbeacon", 100_000)
    assert 0 <= a < 100_000
    assert _skip_for_seed("fineweb", "0xbeacon", 0) == 0


def test_extract_text_prefers_field_then_falls_back():
    assert _extract_text({"text": "hello"}, "text") == "hello"
    # Missing configured field -> first long string value.
    assert _extract_text({"body": "x" * 300}, "text") == "x" * 300
    assert _extract_text({"n": 5}, "text") == ""


def test_fetch_source_chunks_skips_then_fills():
    source = _SOURCES[0]
    chunk_bytes = 4096
    skip_cap = 8
    skip = _skip_for_seed(source.name, "seed1", skip_cap)
    recs = _records("alpha", skip_cap + 200)
    chunks, prov = fetch_source_chunks(
        source, chunk_bytes, seed="seed1", skip_cap=skip_cap, records=recs
    )
    assert len(chunks) == source.chunks
    assert all(len(c) == chunk_bytes for c in chunks)
    assert prov["records_skipped"] == skip
    assert prov["source"] == "alpha"
    assert prov["bytes"] == chunk_bytes * source.chunks
    # The first kept document must come from AFTER the skipped records.
    assert f"alpha-{skip:04d}".encode() in chunks[0]


def test_fetch_source_chunks_raises_when_exhausted():
    source = _SOURCES[0]
    with pytest.raises(RuntimeError):
        fetch_source_chunks(source, 4096, seed="s", skip_cap=2, records=_records("alpha", 3))


def test_write_mixed_corpus_orders_and_records_provenance(tmp_path):
    chunk_bytes = 4096
    skip_cap = 8
    prov = write_mixed_corpus(
        tmp_path,
        _SOURCES,
        chunk_bytes,
        seed="seed1",
        skip_cap=skip_cap,
        records_by_source=_records_by_source(skip_cap),
    )
    files = sorted(p.name for p in tmp_path.glob("chunk_*"))
    # 2 alpha + 1 beta, named/ordered so sorted concatenation is alpha... then beta.
    assert files == ["chunk_00_alpha.txt", "chunk_01_alpha.txt", "chunk_02_beta.txt"]
    assert [e["source"] for e in prov] == ["alpha", "beta"]
    assert (tmp_path / "provenance.json").exists()

    provider = StaticLocalProvider(tmp_path)
    assert provider.total_bytes == chunk_bytes * 3


def test_metadata_files_excluded_from_corpus(tmp_path):
    chunk_bytes = 4096
    skip_cap = 8
    write_mixed_corpus(
        tmp_path,
        _SOURCES,
        chunk_bytes,
        seed="seed1",
        skip_cap=skip_cap,
        records_by_source=_records_by_source(skip_cap),
    )
    provider = StaticLocalProvider(tmp_path)
    manifest = provider.manifest()
    (tmp_path / "manifest.json").write_text(json.dumps({"manifest_hash": manifest.manifest_hash()}))
    # Adding manifest.json must not change the sampled corpus size or hash.
    reloaded = StaticLocalProvider(tmp_path)
    assert reloaded.total_bytes == provider.total_bytes
    assert reloaded.manifest().manifest_hash() == manifest.manifest_hash()
    assert all(c.id.startswith("chunk_") for c in reloaded.manifest().chunks)


# --- resolve_live_corpus: issue #71's core acceptance criteria ---------------------------
#
# Two independent validators calling resolve_live_corpus with the same beacon-derived seed
# must land on byte-identical corpora with no shared file and no coordination between them.
# Each call here uses its own cache_root (simulating two separate validator hosts) and its
# own copy of the injected records (simulating two separate, independent HF stream reads).


def _patch_iter_dataset(monkeypatch, skip_cap):
    """Make _iter_dataset return a fresh copy of fake records every call (never network).

    Each call gets its OWN fresh iterator -- exactly like two independent validators each
    opening their own streaming connection to the real HF dataset -- so a same-seed test
    isn't just replaying one shared iterator.
    """

    import eval.live_corpus as live_corpus

    records = _records_by_source(skip_cap)

    def fake_iter_dataset(source, token):
        return iter(records[source.name])

    monkeypatch.setattr(live_corpus, "_iter_dataset", fake_iter_dataset)


def test_resolve_live_corpus_same_seed_is_byte_identical_across_independent_instances(tmp_path, monkeypatch):
    _patch_iter_dataset(monkeypatch, skip_cap=8)

    validator_a = resolve_live_corpus(
        "beacon-round-42", sources=_SOURCES, chunk_bytes=4096, skip_cap=8, cache_root=tmp_path / "validator-a"
    )
    validator_b = resolve_live_corpus(
        "beacon-round-42", sources=_SOURCES, chunk_bytes=4096, skip_cap=8, cache_root=tmp_path / "validator-b"
    )

    assert validator_a.manifest().manifest_hash() == validator_b.manifest().manifest_hash()
    assert validator_a.read_range(0, validator_a.total_bytes) == validator_b.read_range(0, validator_b.total_bytes)


def test_resolve_live_corpus_different_seed_differs(tmp_path, monkeypatch):
    _patch_iter_dataset(monkeypatch, skip_cap=8)

    a = resolve_live_corpus(
        "beacon-round-1", sources=_SOURCES, chunk_bytes=4096, skip_cap=8, cache_root=tmp_path / "a"
    )
    b = resolve_live_corpus(
        "beacon-round-2", sources=_SOURCES, chunk_bytes=4096, skip_cap=8, cache_root=tmp_path / "b"
    )

    assert a.manifest().manifest_hash() != b.manifest().manifest_hash()


def test_slice_is_not_the_dataset_prefix():
    # A nonzero seed must skip a nonzero number of records for at least one source -- the
    # corpus must not be the fixed, memorisable prefix of the dataset.
    skip_cap = 1000
    skip_alpha = _skip_for_seed("alpha", "beacon-xyz", skip_cap)
    skip_beta = _skip_for_seed("beta", "beacon-xyz", skip_cap)
    assert skip_alpha > 0 or skip_beta > 0


def test_resolve_live_corpus_reuses_cache_without_rebuilding(tmp_path, monkeypatch):
    _patch_iter_dataset(monkeypatch, skip_cap=8)
    cache_root = tmp_path / "cache"
    seed = "beacon-round-7"

    provider_1 = resolve_live_corpus(
        seed, sources=_SOURCES, chunk_bytes=4096, skip_cap=8, cache_root=cache_root
    )

    import eval.live_corpus as live_corpus

    def failing_iter_dataset(source, token):
        raise AssertionError("must not re-stream from HF for an already-cached seed")

    monkeypatch.setattr(live_corpus, "_iter_dataset", failing_iter_dataset)
    provider_2 = resolve_live_corpus(
        seed, sources=_SOURCES, chunk_bytes=4096, skip_cap=8, cache_root=cache_root
    )

    assert provider_1.manifest().manifest_hash() == provider_2.manifest().manifest_hash()
