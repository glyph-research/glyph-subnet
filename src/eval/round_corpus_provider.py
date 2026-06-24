"""Resolve beacon-derived ChunkLocators to bytes via Parquet row-group range reads (#22, 2/n).

Each ``ChunkLocator`` (from ``core.round_corpus``) carries a beacon-only seed. This module
turns that seed into a concrete (file, row-group, offset) against the source's *pinned* Parquet
snapshot and reads only the needed row-group(s) via HTTP range -- never the whole multi-GB
shard. Because selection is beacon-only and the revision is pinned, every validator assembles
byte-identical chunk bytes for a given beacon.

The selection/assembly core (``fetch_chunk_from_files``) is filesystem-agnostic and unit-tested
on a local Parquet fixture; ``fetch_chunk_bytes`` wires it to ``huggingface_hub.HfFileSystem``.
"""

from __future__ import annotations

from collections.abc import Callable

from core.round_corpus import CHUNK_BYTES, ChunkLocator, position_keys

# Joined between rows so the assembled chunk is one contiguous text blob.
ROW_JOIN = b"\n\n"


def _row_group_text_bytes(parquet_file, row_group: int, column: str) -> bytes:
    table = parquet_file.read_row_group(row_group, columns=[column])
    buf = bytearray()
    for value in table.column(0).to_pylist():
        if value:
            buf += value.encode("utf-8") + ROW_JOIN
    return bytes(buf)


def fetch_chunk_from_files(
    seed: bytes,
    files: list[str],
    open_file: Callable[[str], object],
    *,
    chunk_bytes: int = CHUNK_BYTES,
    text_column: str = "text",
) -> bytes:
    """Assemble exactly ``chunk_bytes`` of bytes for ``seed`` from ``files`` (sorted, pinned).

    Picks a file and a starting row-group from the seed, reads forward (wrapping) until at least
    ``chunk_bytes`` are assembled, then returns a ``chunk_bytes`` window at a seed-derived offset.
    Reads at row-group granularity, so only the touched groups are fetched.
    """

    import pyarrow.parquet as pq

    if not files:
        raise RuntimeError("no parquet files for source")
    file_key, rg_key, offset_key = position_keys(seed)
    path = files[file_key % len(files)]
    parquet_file = pq.ParquetFile(open_file(path))
    num_row_groups = parquet_file.metadata.num_row_groups
    names = parquet_file.schema_arrow.names
    column = text_column if text_column in names else names[0]

    start = rg_key % num_row_groups
    buf = bytearray()
    read = 0
    while len(buf) < chunk_bytes and read < num_row_groups:
        buf += _row_group_text_bytes(parquet_file, (start + read) % num_row_groups, column)
        read += 1
    if len(buf) < chunk_bytes:
        raise RuntimeError(f"source exhausted: assembled {len(buf)} < {chunk_bytes}")

    span = len(buf) - chunk_bytes
    offset = offset_key % (span + 1) if span > 0 else 0
    return bytes(buf[offset : offset + chunk_bytes])


def list_parquet_files(fs, dataset: str, revision: str) -> list[str]:
    """Sorted list of the dataset's Parquet shard paths at the pinned ``revision``."""

    base = f"datasets/{dataset}@{revision}"
    files = [p for p in fs.find(base) if p.endswith(".parquet")]
    return sorted(files)


def fetch_chunk_bytes(
    locator: ChunkLocator,
    fs=None,
    *,
    chunk_bytes: int = CHUNK_BYTES,
    text_column: str = "text",
) -> bytes:
    """Fetch a locator's bytes from HuggingFace via pinned-revision Parquet range reads."""

    from huggingface_hub import HfFileSystem

    fs = fs or HfFileSystem()
    files = list_parquet_files(fs, locator.dataset, locator.revision)
    return fetch_chunk_from_files(
        bytes.fromhex(locator.seed_hex),
        files,
        fs.open,
        chunk_bytes=chunk_bytes,
        text_column=text_column,
    )
