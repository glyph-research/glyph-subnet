"""Deterministic, beacon-seeded corpus streamed live from HuggingFace (issue #71).

Previously the owner ran a separate process (``glyph-oracle``) to build a mixed corpus
file once, publish it, and commit its hash on-chain; every validator then pointed
``--corpus-dir``/``--corpus-url`` at that same shared file. That is an extra always-on
operational dependency and a single point of failure -- if the oracle process stalls, the
corpus goes stale for the whole network.

Instead, each validator builds its own copy of the corpus directly, independently, from the
same beacon-seeded skip offset -- no shared file, no owner process. Determinism across
independent validators comes from ``_skip_for_seed`` and ``fetch_source_chunks`` being pure
functions of ``(seed, dataset)``: given the same seed and the same pinned dataset revision,
two validators land on byte-identical chunks without coordinating.

The anti-memorisation guarantee does not rest on freshness (a mixed-source corpus rebuilt
from a fixed dataset revision is no less resistant to memorisation than one rebuilt daily --
see ``MIXED_SOURCES``' history). It rests on:

1. **Scale + an unpredictable slice.** FineWeb / The Pile are tens of TB -- far beyond what a
   codec can embed under the artifact size cap. We do NOT take the fixed prefix of each
   dataset (a miner could just memorise the first few MiB); we skip a seed-derived number of
   records first, so *which* slice lands in the corpus is not knowable in advance. The seed is
   the post-commitment chain beacon, so the slice is fixed for a given round yet unpredictable
   at commit time.
2. **Beacon-seeded stream sampling** (eval/streams.py) over the resolved corpus, so which
   windows are actually scored is unpredictable too.
3. **Source mixing**, which bounds the influence of any single poisoned source to a
   negligible fraction of the corpus.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from eval.corpus import StaticLocalProvider

# Minimum bytes for a record to be worth keeping (drops boilerplate/stubs).
_MIN_DOC_BYTES = 200

# Default per-round corpus shape: 8 x 2 MiB (3x FineWeb / 3x Pile / 2x enwik9), matching the
# launch mix (issue #10).
DEFAULT_CHUNK_BYTES = 2 * 2**20
DEFAULT_SKIP_CAP = 100_000


@dataclass(frozen=True)
class Source:
    """One mixed-corpus source and how many chunk files it contributes."""

    name: str
    dataset: str
    config: str | None
    split: str
    text_field: str
    chunks: int


# The launch mix: 3x FineWeb / 3x Pile / 2x enwik9 (issue #10). Order here is the order the
# chunk files are concatenated into the corpus.
MIXED_SOURCES: list[Source] = [
    Source("fineweb", "HuggingFaceFW/fineweb", "sample-10BT", "train", "text", 3),
    Source("pile", "monology/pile-uncopyrighted", None, "train", "text", 3),
    Source("enwik9", "haukur/enwik9", None, "train", "text", 2),
]


def _extract_text(record: dict, text_field: str) -> str:
    """Pull the text out of a dataset record, tolerating schema differences.

    Prefer the configured field; otherwise fall back to the first string-valued field. The
    exact field name varies across datasets, so we stay defensive rather than assume.
    """

    value = record.get(text_field)
    if isinstance(value, str):
        return value
    for candidate in record.values():
        if isinstance(candidate, str) and len(candidate) >= _MIN_DOC_BYTES:
            return candidate
    return ""


def _skip_for_seed(source_name: str, seed: str, dataset_records_skip_cap: int) -> int:
    """Deterministic, seed-derived number of records to skip before sampling a source.

    Taking the fixed prefix of a public dataset would let a miner memorise exactly that
    slice, so the start offset is derived from the beacon ``seed``. It is bounded by
    ``dataset_records_skip_cap`` so no validator has to stream forever.
    """

    digest = hashlib.sha256(f"{seed}:{source_name}".encode()).digest()
    if dataset_records_skip_cap <= 0:
        return 0
    return int.from_bytes(digest[:8], "big") % dataset_records_skip_cap


def _iter_dataset(source: Source, token: str | None) -> Iterator[dict]:
    from datasets import load_dataset

    dataset = load_dataset(
        source.dataset,
        source.config,
        split=source.split,
        streaming=True,
        token=token,
    )
    return iter(dataset)


def fetch_source_chunks(
    source: Source,
    chunk_bytes: int,
    *,
    seed: str,
    token: str | None = None,
    skip_cap: int = DEFAULT_SKIP_CAP,
    records: Iterable[dict] | None = None,
) -> tuple[list[bytes], dict]:
    """Stream ``source`` into ``source.chunks`` chunks of ``chunk_bytes`` bytes each.

    Skips a seed-derived number of records first (so the slice is not the fixed prefix), then
    concatenates documents until the chunks are filled. ``records`` is injectable for tests;
    in production it comes from a streaming ``datasets`` iterator. Returns the chunk byte blobs
    plus a provenance entry recording exactly what was drawn.
    """

    target = chunk_bytes * source.chunks
    skip = _skip_for_seed(source.name, seed, skip_cap)
    iterator = iter(records) if records is not None else _iter_dataset(source, token)

    buffer = bytearray()
    docs = 0
    skipped = 0
    for record in iterator:
        if skipped < skip:
            skipped += 1
            continue
        text = _extract_text(record, source.text_field)
        if len(text.encode("utf-8")) < _MIN_DOC_BYTES:
            continue
        buffer += (text.rstrip() + "\n\n").encode("utf-8")
        docs += 1
        if len(buffer) >= target:
            break

    if len(buffer) < target:
        raise RuntimeError(
            f"source {source.name!r} yielded {len(buffer)} bytes, need {target} "
            f"({source.chunks} x {chunk_bytes}); dataset exhausted or skip_cap too high"
        )

    chunks = [bytes(buffer[i * chunk_bytes : (i + 1) * chunk_bytes]) for i in range(source.chunks)]
    provenance = {
        "source": source.name,
        "dataset": source.dataset,
        "config": source.config,
        "split": source.split,
        "chunks": source.chunks,
        "records_skipped": skip,
        "records_used": docs,
        "bytes": target,
    }
    return chunks, provenance


def write_mixed_corpus(
    out_dir: Path,
    sources: list[Source],
    chunk_bytes: int,
    *,
    seed: str,
    token: str | None = None,
    skip_cap: int = DEFAULT_SKIP_CAP,
    records_by_source: dict[str, Iterable[dict]] | None = None,
) -> list[dict]:
    """Build the mixed corpus on disk and return the provenance records.

    Chunk files are named ``chunk_<NN>_<source>.txt`` so the sorted concatenation order
    (FineWeb, then Pile, then enwik9) is stable and self-describing.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("chunk_*"):
        stale.unlink()

    provenance: list[dict] = []
    index = 0
    for source in sources:
        injected = records_by_source.get(source.name) if records_by_source else None
        chunks, entry = fetch_source_chunks(
            source, chunk_bytes, seed=seed, token=token, skip_cap=skip_cap, records=injected
        )
        chunk_ids = []
        for chunk in chunks:
            name = f"chunk_{index:02d}_{source.name}.txt"
            (out_dir / name).write_bytes(chunk)
            chunk_ids.append(name)
            index += 1
        entry["chunk_ids"] = chunk_ids
        provenance.append(entry)

    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return provenance


def local_corpus_cache_dir(seed: str, *, cache_root: Path | None = None) -> Path:
    """Stable, per-seed directory for one round's materialized live corpus.

    Keyed off a hash of the seed alone (not a naive sanitized string): the seed is a
    beacon-derived value with no path-traversal risk, but hashing keeps the directory name
    bounded regardless of seed length/formatting.
    """

    digest = hashlib.sha256(str(seed).encode("utf-8")).hexdigest()
    root = cache_root or Path(tempfile.gettempdir()) / "glyph-live-corpus"
    return root / digest


def resolve_live_corpus(
    seed: str,
    *,
    sources: list[Source] | None = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    skip_cap: int = DEFAULT_SKIP_CAP,
    token: str | None = None,
    cache_root: Path | None = None,
) -> StaticLocalProvider:
    """Deterministically stream this round's corpus slice straight from HuggingFace.

    Two independent calls with the same ``seed`` land on byte-identical chunks -- there is no
    shared file and no owner-run process to keep alive (issue #71). Reuses the on-disk cache
    when this exact seed was already built (e.g. this round already ran once), rather than
    re-streaming from HuggingFace on every call.
    """

    out_dir = local_corpus_cache_dir(seed, cache_root=cache_root)
    if not (out_dir / "provenance.json").is_file():
        write_mixed_corpus(
            out_dir, sources or MIXED_SOURCES, chunk_bytes, seed=str(seed), token=token, skip_cap=skip_cap
        )
    return StaticLocalProvider(out_dir)
