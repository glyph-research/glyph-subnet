import json

import pytest

from eval.corpus import OracleProvider, StaticLocalProvider
from oracle.oracle import (
    Source,
    fetch_source_chunks,
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


def test_oracle_provider_hash_verification(tmp_path):
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
    good_hash = StaticLocalProvider(tmp_path).manifest().manifest_hash()

    provider = OracleProvider(tmp_path, expected_manifest_hash=good_hash)
    assert provider.total_bytes > 0
    with pytest.raises(ValueError):
        OracleProvider(tmp_path, expected_manifest_hash="deadbeef")


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
