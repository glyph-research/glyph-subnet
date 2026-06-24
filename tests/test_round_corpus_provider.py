"""Offline tests for the round-corpus provider: range-read == full-read, plus determinism.

Uses a local Parquet fixture (no network), which is exactly the equivalence Taras asked to
verify before merge -- that reading selected row-groups via range assembles byte-identical
data to a full read.
"""

import math

import pyarrow as pa
import pyarrow.parquet as pq

from core.round_corpus import position_keys, round_seed
from eval.round_corpus_provider import ROW_JOIN, fetch_chunk_from_files


def _open_rb(path):
    return open(path, "rb")

ROW_GROUP_SIZE = 50


def _make_fixture(path, n_rows=600):
    texts = [f"row-{i:05d} " + "lorem ipsum dolor sit amet " * 4 for i in range(n_rows)]
    pq.write_table(pa.table({"text": texts}), path, row_group_size=ROW_GROUP_SIZE)
    return texts


def _reference(seed: bytes, texts: list[str], chunk_bytes: int) -> bytes:
    """Recompute the expected chunk from a FULL read of the rows (no range), mirroring the
    provider's selection so any divergence means range-read != full-read."""

    file_key, rg_key, offset_key = position_keys(seed)
    assert file_key % 1 == 0  # single fixture file -> file index 0
    num_rg = math.ceil(len(texts) / ROW_GROUP_SIZE)

    def rg_bytes(i: int) -> bytes:
        buf = bytearray()
        for value in texts[i * ROW_GROUP_SIZE : (i + 1) * ROW_GROUP_SIZE]:
            if value:
                buf += value.encode("utf-8") + ROW_JOIN
        return bytes(buf)

    start = rg_key % num_rg
    buf = bytearray()
    read = 0
    while len(buf) < chunk_bytes and read < num_rg:
        buf += rg_bytes((start + read) % num_rg)
        read += 1
    span = len(buf) - chunk_bytes
    offset = offset_key % (span + 1) if span > 0 else 0
    return bytes(buf[offset : offset + chunk_bytes])


def test_range_read_matches_full_read(tmp_path):
    path = str(tmp_path / "shard.parquet")
    texts = _make_fixture(path)
    chunk_bytes = 1500
    for beacon in ["0xaaa", "0xbbb", "0xccc", "0xddd"]:
        seed = round_seed(beacon, "fineweb", 0)
        got = fetch_chunk_from_files(seed, [path], _open_rb, chunk_bytes=chunk_bytes)
        assert len(got) == chunk_bytes
        assert got == _reference(seed, texts, chunk_bytes)


def test_same_seed_is_deterministic(tmp_path):
    path = str(tmp_path / "shard.parquet")
    _make_fixture(path)
    seed = round_seed("0xbeacon", "pile", 1)
    a = fetch_chunk_from_files(seed, [path], _open_rb, chunk_bytes=1200)
    b = fetch_chunk_from_files(seed, [path], _open_rb, chunk_bytes=1200)
    assert a == b


def test_different_beacons_generally_differ(tmp_path):
    path = str(tmp_path / "shard.parquet")
    _make_fixture(path)
    out = {
        fetch_chunk_from_files(round_seed(f"0xblk{i}", "fineweb", 0), [path], _open_rb, chunk_bytes=1000)
        for i in range(12)
    }
    assert len(out) > 1  # fresh selection across beacons, not a constant


def test_text_column_fallback(tmp_path):
    # A source whose text column is not literally "text" still resolves (first column).
    path = str(tmp_path / "alt.parquet")
    pq.write_table(
        pa.table({"content": [f"doc {i} " + "abc " * 30 for i in range(300)]}),
        path,
        row_group_size=ROW_GROUP_SIZE,
    )
    seed = round_seed("0xb", "enwik9", 0)
    got = fetch_chunk_from_files(seed, [path], _open_rb, chunk_bytes=900, text_column="text")
    assert len(got) == 900
