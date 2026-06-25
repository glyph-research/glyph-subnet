"""Resolve beacon-derived ChunkLocators to bytes via Parquet row-group range reads (#22).

Each ``ChunkLocator`` (from ``core.round_corpus``) carries a beacon-only seed. This module
turns that seed into a concrete (file, row-group, offset) against the source's *pinned* Parquet
snapshot and reads only the needed row-group(s) via HTTP range -- never the whole multi-GB
shard. Because selection is beacon-only and the revision is pinned, every validator assembles
byte-identical chunk bytes for a given beacon.

The selection/assembly core (``fetch_chunk_from_files``) is filesystem-agnostic and unit-tested
on a local Parquet fixture; ``fetch_chunk_bytes`` wires it to ``huggingface_hub.HfFileSystem``.
``RoundCorpusProvider`` adds file-list + opened-ParquetFile caching and next-beacon prefetch so a
whole round's fetch stays within the round cadence budget -- the dominant cost is the per-file
Parquet footer parse on open, which the cache pays once instead of per chunk/round (#22, 3/n).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable

from core.round_corpus import CHUNK_BYTES, SOURCE_SPEC, ChunkLocator, derive_round_corpus, position_keys

# Joined between rows so the assembled chunk is one contiguous text blob.
ROW_JOIN = b"\n\n"


def _row_group_text_bytes(parquet_file, row_group: int, column: str) -> bytes:
    table = parquet_file.read_row_group(row_group, columns=[column])
    buf = bytearray()
    for value in table.column(0).to_pylist():
        if value:
            buf += value.encode("utf-8") + ROW_JOIN
    return bytes(buf)


def _assemble_chunk(
    seed: bytes,
    files: list[str],
    get_parquet: Callable[[str], object],
    *,
    chunk_bytes: int,
    text_column: str,
) -> bytes:
    """Pick file + start row-group from ``seed`` and assemble a ``chunk_bytes`` window.

    ``get_parquet(path)`` returns a ``pyarrow`` ``ParquetFile`` (cached or freshly opened), which
    is the seam the caching provider hooks so the footer parse is paid once per file.
    """

    if not files:
        raise RuntimeError("no parquet files for source")
    file_key, rg_key, offset_key = position_keys(seed)
    path = files[file_key % len(files)]
    parquet_file = get_parquet(path)
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

    return _assemble_chunk(
        seed,
        files,
        lambda path: pq.ParquetFile(open_file(path)),
        chunk_bytes=chunk_bytes,
        text_column=text_column,
    )


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


class RoundCorpusProvider:
    """Per-round corpus fetch with caching + next-beacon prefetch (#22, 3/n).

    Reuses one filesystem connection and caches (a) the parquet file list per pinned
    ``(dataset, revision)`` and (b) opened ``ParquetFile`` objects (LRU) -- so the expensive
    per-file footer parse is paid once instead of on every chunk and every round. ``prefetch``
    warms those caches for the *next* beacon while the current round runs, keeping the
    blocking per-round fetch within the round cadence budget.
    """

    def __init__(self, fs=None, *, max_open_files: int = 16, text_column: str = "text") -> None:
        self._fs = fs
        self._max_open = max_open_files
        self._text_column = text_column
        self._file_lists: dict[tuple[str, str], list[str]] = {}
        self._open: OrderedDict[str, object] = OrderedDict()

    def _filesystem(self):
        if self._fs is None:
            from huggingface_hub import HfFileSystem

            self._fs = HfFileSystem()
        return self._fs

    def _files(self, dataset: str, revision: str) -> list[str]:
        key = (dataset, revision)
        cached = self._file_lists.get(key)
        if cached is None:
            cached = list_parquet_files(self._filesystem(), dataset, revision)
            self._file_lists[key] = cached
        return cached

    def _parquet(self, path: str):
        pf = self._open.get(path)
        if pf is not None:
            self._open.move_to_end(path)
            return pf
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(self._filesystem().open(path))
        self._open[path] = pf
        while len(self._open) > self._max_open:
            self._open.popitem(last=False)  # evict least-recently-used
        return pf

    def _selected_path(self, locator: ChunkLocator) -> str:
        files = self._files(locator.dataset, locator.revision)
        if not files:
            raise RuntimeError(f"no parquet files for {locator.dataset}@{locator.revision}")
        file_key, _, _ = position_keys(bytes.fromhex(locator.seed_hex))
        return files[file_key % len(files)]

    def fetch_chunk(self, locator: ChunkLocator, *, chunk_bytes: int = CHUNK_BYTES) -> bytes:
        return _assemble_chunk(
            bytes.fromhex(locator.seed_hex),
            self._files(locator.dataset, locator.revision),
            self._parquet,
            chunk_bytes=chunk_bytes,
            text_column=self._text_column,
        )

    def fetch_round(
        self, beacon: str, *, spec=SOURCE_SPEC, chunk_bytes: int = CHUNK_BYTES
    ) -> tuple[list[bytes], float]:
        """All of ``beacon``'s chunks (in locator order) plus the wall-clock seconds taken."""

        started = time.time()
        chunks = [
            self.fetch_chunk(loc, chunk_bytes=chunk_bytes)
            for loc in derive_round_corpus(beacon, spec)
        ]
        return chunks, time.time() - started

    def prefetch(self, beacon: str, *, spec=SOURCE_SPEC) -> int:
        """Warm the file-list + ParquetFile (footer) caches for ``beacon``. Returns files opened.

        Call this for the *next* beacon while the current round evaluates so the next round's
        fetch skips the dominant footer-parse cost. Best-effort: a failure to warm one file does
        not raise (the real fetch will surface it).
        """

        opened = 0
        for loc in derive_round_corpus(beacon, spec):
            try:
                path = self._selected_path(loc)
                if path not in self._open:
                    self._parquet(path)
                    opened += 1
            except Exception:
                continue
        return opened
