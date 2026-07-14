"""Deterministic, beacon-seeded corpus streamed live from HuggingFace (issue #71).

Previously the owner ran a separate process (``glyph-oracle``) to build a mixed corpus
file once, publish it, and commit its hash on-chain; every validator then pointed
``--corpus-dir``/``--corpus-url`` at that same shared file. That is an extra always-on
operational dependency and a single point of failure -- if the oracle process stalls, the
corpus goes stale for the whole network.

Instead, each validator builds its own copy of the corpus directly, independently, from the
same beacon-seeded shard + skip offset -- no shared file, no owner process. Determinism
across independent validators comes from ``_shard_for_seed``/``_skip_for_seed``/
``fetch_source_chunks`` being pure functions of ``(seed, dataset)``: given the same seed and
the same dataset file layout, two validators land on byte-identical chunks without
coordinating.

The anti-memorisation guarantee does not rest on freshness (a mixed-source corpus rebuilt
from a fixed dataset revision is no less resistant to memorisation than one rebuilt daily --
see ``MIXED_SOURCES``' history). It rests on:

1. **Scale + an unpredictable slice.** fineweb-edu / The Pile are TBs -- far beyond what a
   codec can embed under the artifact size cap. We do NOT take the fixed prefix of each
   dataset (a miner could just memorise the first few MiB). A seed-derived *shard* of the
   dataset is selected first, THEN a seed-derived number of records is skipped within that
   one shard (issue #112) -- the reachable universe is the whole dataset, while the
   per-round streaming cost stays a bounded walk into one file. The seed is the
   post-commitment chain beacon, so the slice is fixed for a given round yet unpredictable
   at commit time.

   History (issue #112): the bounded skip originally applied from record 0 of the whole
   dataset, so the entire reachable attack surface was the first ``skip_cap`` records per
   source -- a few hundred MB, trivially embeddable, and confirmed exploited on mainnet by a
   dictionary-lookup codec. Shard-randomisation makes the skip bound a latency decision
   again instead of the security boundary. It still doesn't make a partial-cache hit
   *impossible* on any single round -- it works because the incumbent is freshly
   re-evaluated on independently-redrawn shards every round it holds the crown, so a
   partial cache must hit on every scored source simultaneously, every round, forever.
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

# Per-round corpus shape (issue #112): 5 x 4 MiB (2x fineweb-edu / 1x pile / 2x enwik9).
# Chunk size matches EVAL_STREAM_BYTES so each scored source's span supports its scored
# windows: fineweb-edu's 8 MiB span gives its two 4 MiB windows real start-offset freedom,
# pile's 4 MiB span is exactly its one 4 MiB window (its per-round unpredictability comes
# from the shard+skip randomisation of the slice itself).
DEFAULT_CHUNK_BYTES = 4 * 2**20
# Bounds how far the seed-derived record skip can reach WITHIN the selected shard -- purely
# a latency cap so no validator has to walk a whole shard every round. NOT a security
# boundary: which shard is walked is itself seed-derived out of the full dataset (issue
# #112; this cap being the entire reachable universe was exactly the exploited flaw).
DEFAULT_SKIP_CAP = 100_000


@dataclass(frozen=True)
class Source:
    """One mixed-corpus source and how many chunk files it contributes.

    ``excluded_shards`` retires dataset shards that are permanently compromised for scoring
    (issue #112): pile's shard 0 covers records [0, 100_000) -- the entire reachable universe
    of the original prefix-bounded sampler, confirmed embedded in an exploiting codec's
    lookup dictionary -- so it is never selectable again.
    """

    name: str
    dataset: str
    config: str | None
    split: str
    text_field: str
    chunks: int
    excluded_shards: tuple[int, ...] = ()


# The scored mix (issue #112): 2x fineweb-edu / 1x pile, plus 2x enwik9 (benchmark-only
# display, not scored -- see EVAL_BENCHMARK_SOURCE). fineweb-edu's `default` config is the
# full corpus spanning all CommonCrawl dumps (3.5+ TB), not the sample-10BT/100BT/350BT
# convenience subsets -- brand new to scoring, so no burned-range exclusion needed. Order
# here is the order the chunk files are concatenated into the corpus.
MIXED_SOURCES: list[Source] = [
    Source("fineweb-edu", "HuggingFaceFW/fineweb-edu", "default", "train", "text", 2),
    Source("pile", "monology/pile-uncopyrighted", None, "train", "text", 1, excluded_shards=(0,)),
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
    """Deterministic, seed-derived number of records to skip within the selected shard.

    Taking the fixed prefix of a shard would let a miner memorise every shard's prefix, so
    the start offset within the shard is derived from the beacon ``seed`` too. It is bounded
    by ``dataset_records_skip_cap`` so no validator has to stream forever -- a pure latency
    cap, not the security boundary (see ``_shard_for_seed`` / issue #112).
    """

    digest = hashlib.sha256(f"{seed}:{source_name}".encode()).digest()
    if dataset_records_skip_cap <= 0:
        return 0
    return int.from_bytes(digest[:8], "big") % dataset_records_skip_cap


def _shard_for_seed(source: Source, seed: str, num_shards: int) -> int:
    """Deterministic, seed-derived shard index for one source, from the selectable pool.

    Selecting the shard first is what makes the whole dataset the reachable sampling
    universe (issue #112): the bounded record skip then applies within that one shard, so
    per-round streaming cost stays a bounded walk into one file while the slice a miner
    would have to pre-embed grows from ~skip_cap records to the entire dataset. Shards in
    ``source.excluded_shards`` (permanently compromised ranges) are never selectable. Pure
    function of ``(seed, source)`` -- two validators pick the same shard without
    coordinating, as long as they see the same dataset file layout.
    """

    pool = [index for index in range(num_shards) if index not in source.excluded_shards]
    if not pool:
        raise RuntimeError(
            f"source {source.name!r} has no selectable shards "
            f"({num_shards} total, {len(source.excluded_shards)} excluded)"
        )
    digest = hashlib.sha256(f"{seed}:{source.name}:shard".encode()).digest()
    return pool[int.from_bytes(digest[:8], "big") % len(pool)]


def _iter_dataset(source: Source, token: str | None, *, seed: str) -> tuple[Iterator[dict], dict]:
    """Open one seed-selected shard of ``source`` as a streaming record iterator.

    Returns ``(iterator, shard_meta)`` where ``shard_meta`` records which shard was drawn
    out of how many, for provenance. ``num_shards`` comes from the dataset's file layout, so
    it is identical for every validator reading the same dataset state -- same property the
    record contents themselves already rely on.
    """

    from datasets import load_dataset

    dataset = load_dataset(
        source.dataset,
        source.config,
        split=source.split,
        streaming=True,
        token=token,
    )
    num_shards = int(dataset.n_shards)
    shard_index = _shard_for_seed(source, seed, num_shards)
    # num_shards == n_shards -> each group is exactly one underlying file (contiguous split).
    dataset = dataset.shard(num_shards=num_shards, index=shard_index)
    return iter(dataset), {"shard_index": shard_index, "num_shards": num_shards}


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

    Selects a seed-derived shard of the dataset (issue #112), skips a seed-derived number of
    records within it (so the slice is not any shard's fixed prefix), then concatenates
    documents until the chunks are filled. ``records`` is injectable for tests (bypassing
    shard selection -- the injected iterable IS the shard); in production it comes from a
    streaming ``datasets`` iterator over the selected shard. Returns the chunk byte blobs
    plus a provenance entry recording exactly what was drawn.
    """

    target = chunk_bytes * source.chunks
    skip = _skip_for_seed(source.name, seed, skip_cap)
    if records is not None:
        iterator, shard_meta = iter(records), {"shard_index": None, "num_shards": None}
    else:
        iterator, shard_meta = _iter_dataset(source, token, seed=seed)

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
            f"({source.chunks} x {chunk_bytes}); shard exhausted or skip_cap too high"
        )

    chunks = [bytes(buffer[i * chunk_bytes : (i + 1) * chunk_bytes]) for i in range(source.chunks)]
    provenance = {
        "source": source.name,
        "dataset": source.dataset,
        "config": source.config,
        "split": source.split,
        "chunks": source.chunks,
        "shard_index": shard_meta["shard_index"],
        "num_shards": shard_meta["num_shards"],
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
