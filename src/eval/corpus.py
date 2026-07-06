"""Corpus providers.

The corpus is a single logical byte sequence that streams are sampled from. A
``CorpusProvider`` exposes its manifest (so the sample is reproducible and the data is
freshness-auditable) and materializes byte ranges.

- ``StaticLocalProvider``: concatenates files in a directory; used for tests and M0
  dry-runs.
- ``OracleProvider``: resolves the owner-published fresh-data corpus by its on-chain
  manifest hash (implemented with the data oracle, glyph-oracle).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from core.hashing import sha256_file
from eval.streams import RangeSource, StreamSpec


@dataclass(frozen=True)
class ChunkRef:
    id: str
    size: int
    hash: str


@dataclass(frozen=True)
class CorpusManifest:
    version: int
    total_bytes: int
    chunks: list[ChunkRef]

    def manifest_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(f"v{self.version}:{self.total_bytes}".encode())
        for chunk in self.chunks:
            digest.update(f"|{chunk.id}:{chunk.size}:{chunk.hash}".encode())
        return digest.hexdigest()


class CorpusProvider(Protocol):
    @property
    def total_bytes(self) -> int: ...

    def manifest(self) -> CorpusManifest: ...

    def read_range(self, offset: int, length: int) -> bytes: ...

    def materialize(self, spec: StreamSpec) -> bytes: ...

    def stream_source(self, spec: StreamSpec) -> RangeSource | None: ...


class StaticLocalProvider:
    """Treat the files under ``directory`` (sorted) as one concatenated corpus.

    Reserved metadata files (the corpus manifest and provenance) are never part of the
    sampled content, so the oracle can write them alongside the chunk files.
    """

    RESERVED = {"manifest.json", "provenance.json"}

    def __init__(self, directory: str | Path, *, base_url: str | None = None):
        # base_url: public location serving the *same* corpus as one contiguous blob (chunk
        # order == sorted manifest order), so the Chutes runner range-fetches it instead of the
        # validator inlining bytes. None (tests/M0) -> callers fall back to inlining.
        self.base_url = base_url
        self.directory = Path(directory)
        files = [
            p
            for p in self.directory.rglob("*")
            if p.is_file() and p.name not in self.RESERVED
        ]
        files.sort(key=lambda p: p.relative_to(self.directory).as_posix())
        self._index: list[tuple[int, int, Path]] = []
        offset = 0
        for path in files:
            size = path.stat().st_size
            self._index.append((offset, size, path))
            offset += size
        self._total = offset

    @property
    def total_bytes(self) -> int:
        return self._total

    def manifest(self) -> CorpusManifest:
        chunks = [
            ChunkRef(id=path.relative_to(self.directory).as_posix(), size=size, hash=sha256_file(path))
            for _, size, path in self._index
        ]
        return CorpusManifest(version=1, total_bytes=self._total, chunks=chunks)

    def read_range(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0:
            raise ValueError("offset and length must be non-negative")
        end = min(offset + length, self._total)
        out = bytearray()
        for start, size, path in self._index:
            file_end = start + size
            if file_end <= offset:
                continue
            if start >= end:
                break
            read_start = max(offset, start) - start
            read_end = min(end, file_end) - start
            with path.open("rb") as handle:
                handle.seek(read_start)
                out += handle.read(read_end - read_start)
        return bytes(out)

    def materialize(self, spec: StreamSpec) -> bytes:
        return self.read_range(spec.offset, spec.length)

    def stream_source(self, spec: StreamSpec) -> RangeSource | None:
        """The remote URL+range for ``spec``, or None when no ``base_url`` is configured."""

        if not self.base_url:
            return None
        return RangeSource(url=self.base_url, offset=spec.offset, length=spec.length)

    def source_range(self, source: str) -> tuple[int, int] | None:
        """Global ``(start, total_bytes)`` of one provenance source's chunks, or None.

        Reads ``provenance.json`` (a list of ``{"source", "chunk_ids": [...]}`` entries written
        beside the corpus) and maps the named source's chunk files to their byte span in the
        concatenated corpus. Returns None when there is no provenance, the source is absent, or
        its chunks are not contiguous (the per-source eval then falls back to whole-corpus
        sampling). Used by the FineWeb-only eval (issue #10).
        """

        import json

        prov_path = self.directory / "provenance.json"
        if not prov_path.is_file():
            return None
        try:
            entries = json.loads(prov_path.read_text())
        except (ValueError, OSError):
            return None
        by_id = {path.relative_to(self.directory).as_posix(): (start, size) for start, size, path in self._index}
        for entry in entries:
            if entry.get("source") != source:
                continue
            spans = [by_id[c] for c in (entry.get("chunk_ids") or []) if c in by_id]
            if not spans:
                return None
            spans.sort()
            start = spans[0][0]
            total = 0
            cursor = start
            for s, size in spans:
                if s != cursor:  # non-contiguous -> can't treat as a single source span
                    return None
                cursor += size
                total += size
            return start, total
        return None


class OracleProvider(StaticLocalProvider):
    """A corpus produced by the data oracle, verified against its published manifest hash.

    In production the owner-run oracle scrapes fresh, attested-timestamp text daily,
    publishes the corpus, and commits the manifest hash on-chain. Validators resolve the
    corpus locally and assert it matches the committed hash.
    """

    def __init__(
        self,
        directory: str | Path,
        expected_manifest_hash: str | None = None,
        *,
        base_url: str | None = None,
    ):
        super().__init__(directory, base_url=base_url)
        if expected_manifest_hash:
            actual = self.manifest().manifest_hash()
            if actual != expected_manifest_hash:
                raise ValueError(
                    f"corpus manifest hash {actual} does not match on-chain commitment "
                    f"{expected_manifest_hash}"
                )
