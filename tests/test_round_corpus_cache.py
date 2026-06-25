"""Offline tests for RoundCorpusProvider caching + prefetch (#22, increment 3).

A fake filesystem over local Parquet fixtures lets us assert the file list and opened
ParquetFiles are cached (so the footer parse is paid once), prefetch warms them, and the
cached path stays byte-for-byte identical to the uncached read.
"""

import pyarrow as pa
import pyarrow.parquet as pq

from core.round_corpus import ChunkLocator, SourceSpec, round_seed
from eval.round_corpus_provider import (
    RoundCorpusProvider,
    fetch_chunk_from_files,
)

ROW_GROUP_SIZE = 50
DATASET = "fake/ds"
REVISION = "rev1"
BASE = f"datasets/{DATASET}@{REVISION}"


class _FakeFS:
    def __init__(self, base_to_paths):
        self.base_to_paths = base_to_paths
        self.find_count = 0
        self.open_count: dict[str, int] = {}

    def find(self, base):
        self.find_count += 1
        return list(self.base_to_paths.get(base, []))

    def open(self, path):
        self.open_count[path] = self.open_count.get(path, 0) + 1
        return open(path, "rb")


def _make_shard(path, tag, n_rows=400):
    texts = [f"{tag}-{i:05d} " + "lorem ipsum dolor sit amet " * 4 for i in range(n_rows)]
    pq.write_table(pa.table({"text": texts}), path, row_group_size=ROW_GROUP_SIZE)


def _fixture(tmp_path):
    p0, p1 = str(tmp_path / "s0.parquet"), str(tmp_path / "s1.parquet")
    _make_shard(p0, "s0")
    _make_shard(p1, "s1")
    return _FakeFS({BASE: [p0, p1]}), [p0, p1]


def _loc(source, chunk):
    seed = round_seed("0xbeacon", source, chunk)
    return ChunkLocator(
        source=source, dataset=DATASET, revision=REVISION, config=None,
        chunk_index=chunk, seed_hex=seed.hex(),
    )


def test_file_list_cached_across_chunks(tmp_path):
    fs, _ = _fixture(tmp_path)
    provider = RoundCorpusProvider(fs)
    for chunk in range(3):
        provider.fetch_chunk(_loc("fineweb", chunk), chunk_bytes=800)
    assert fs.find_count == 1  # listed once for the (dataset, revision), not per chunk


def test_parquet_file_opened_once_per_path(tmp_path):
    fs, _ = _fixture(tmp_path)
    provider = RoundCorpusProvider(fs)
    loc = _loc("fineweb", 0)
    provider.fetch_chunk(loc, chunk_bytes=800)
    provider.fetch_chunk(loc, chunk_bytes=800)
    path = provider._selected_path(loc)
    assert fs.open_count[path] == 1  # footer parsed once, reused on the second fetch


def test_cached_matches_uncached(tmp_path):
    fs, files = _fixture(tmp_path)
    provider = RoundCorpusProvider(fs)
    for chunk in range(4):
        loc = _loc("pile", chunk)
        cached = provider.fetch_chunk(loc, chunk_bytes=900)
        uncached = fetch_chunk_from_files(
            bytes.fromhex(loc.seed_hex), files, lambda p: open(p, "rb"), chunk_bytes=900
        )
        assert cached == uncached


def test_prefetch_warms_cache_so_round_does_no_new_opens(tmp_path):
    fs, _ = _fixture(tmp_path)
    provider = RoundCorpusProvider(fs)
    spec = (SourceSpec("ds", DATASET, REVISION, None, 2),)
    provider.prefetch("0xbeacon", spec=spec)
    opens_after_prefetch = dict(fs.open_count)
    chunks, secs = provider.fetch_round("0xbeacon", spec=spec, chunk_bytes=800)
    assert len(chunks) == 2 and secs >= 0
    assert fs.open_count == opens_after_prefetch  # no new file opens during the round


def test_lru_eviction_reopens_evicted(tmp_path):
    fs, files = _fixture(tmp_path)
    provider = RoundCorpusProvider(fs, max_open_files=1)
    # Find two locators that select different files, then revisit the first.
    locs = [_loc("s", i) for i in range(12)]
    paths = {provider._selected_path(loc): loc for loc in locs}
    assert len(paths) >= 2, "need locators hitting >=2 distinct files"
    (p_a, loc_a), (p_b, loc_b) = list(paths.items())[:2]
    provider.fetch_chunk(loc_a, chunk_bytes=600)
    provider.fetch_chunk(loc_b, chunk_bytes=600)  # evicts p_a (cap 1)
    provider.fetch_chunk(loc_a, chunk_bytes=600)  # must reopen p_a
    assert fs.open_count[p_a] == 2
