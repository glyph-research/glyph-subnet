"""Corpus providers (DESIGN §5).

The corpus is a single logical byte sequence that streams are sampled from. A
``CorpusProvider`` exposes its manifest (so the sample is reproducible and the data is
freshness-auditable) and materializes byte ranges.

- ``StaticLocalProvider``: concatenates files in a directory; used for tests and M0
  dry-runs.
- ``OracleProvider``: resolves the owner-published fresh-data corpus by its on-chain
  manifest hash (implemented with the data oracle, DESIGN §5 / glyph-oracle).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from eval.streams import StreamSpec


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StaticLocalProvider:
    """Treat the files under ``directory`` (sorted) as one concatenated corpus.

    Reserved metadata files (the corpus manifest and provenance) are never part of the
    sampled content, so the oracle can write them alongside the chunk files.
    """

    RESERVED = {"manifest.json", "provenance.json"}

    def __init__(self, directory: str | Path):
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
            ChunkRef(id=path.relative_to(self.directory).as_posix(), size=size, hash=_sha256_file(path))
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


class OracleProvider(StaticLocalProvider):
    """A corpus produced by the data oracle, verified against its published manifest hash.

    In production the owner-run oracle scrapes fresh, attested-timestamp text daily,
    publishes the corpus, and commits the manifest hash on-chain. Validators resolve the
    corpus locally and assert it matches the committed hash (DESIGN §5).
    """

    def __init__(self, directory: str | Path, expected_manifest_hash: str | None = None):
        super().__init__(directory)
        if expected_manifest_hash:
            actual = self.manifest().manifest_hash()
            if actual != expected_manifest_hash:
                raise ValueError(
                    f"corpus manifest hash {actual} does not match on-chain commitment "
                    f"{expected_manifest_hash}"
                )
