"""Glyph data oracle (DESIGN §5): build the evaluation corpus from large, mixed sources.

The owner runs this to (re)build the corpus. Earlier versions scraped the live Wikipedia
recent-changes stream for *freshness* (data that provably postdates miner commitments).
That source is editable, so a miner could plant text it holds a pre-built dictionary for,
wait for the oracle to scrape it, then specialise its codec -- corpus poisoning at the
source (exploit vector #10).

The corpus is now drawn from large, hard-to-influence public datasets, mixed across
independent sources:

    3 chunks  FineWeb   (HuggingFaceFW/fineweb, config sample-10BT)
    3 chunks  The Pile  (monology/pile-uncopyrighted)
    2 chunks  enwik9    (haukur/enwik9)

The anti-memorisation guarantee no longer rests on freshness. It rests on:

1. **Scale + an unpredictable slice.** FineWeb / The Pile are tens of TB -- far beyond
   what a codec can embed under the artifact size cap. Crucially we do NOT take the fixed
   prefix of each dataset (a miner could just memorise the first few MiB); we skip a
   seed-derived number of records first, so *which* slice lands in the corpus is not
   knowable in advance. The seed is the post-commitment chain beacon (``--seed``), so the
   slice is fixed for a given build yet unpredictable at commit time.
2. **Beacon-seeded stream sampling** (eval/streams.py) over the published corpus, so which
   windows are actually scored is unpredictable too.
3. **Source mixing**, which bounds the influence of any single poisoned source to a
   negligible fraction of the corpus.

It writes corpus chunk files plus a manifest, and prints the manifest hash to commit
on-chain. Validators resolve the corpus and verify it against that hash via
``OracleProvider``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from eval.corpus import StaticLocalProvider

USER_AGENT = "glyph-oracle/0.2 (+https://github.com/glyph-research/glyph-subnet)"

# Minimum bytes for a record to be worth keeping (drops boilerplate/stubs).
_MIN_DOC_BYTES = 200


@dataclass(frozen=True)
class Source:
    """One mixed-corpus source and how many chunk files it contributes."""

    name: str
    dataset: str
    config: str | None
    split: str
    text_field: str
    chunks: int


# The launch mix: 3x FineWeb / 3x Pile / 2x enwik9 (issue #10). Order here is the order
# the chunk files are concatenated into the corpus.
MIXED_SOURCES: list[Source] = [
    Source("fineweb", "HuggingFaceFW/fineweb", "sample-10BT", "train", "text", 3),
    Source("pile", "monology/pile-uncopyrighted", None, "train", "text", 3),
    Source("enwik9", "haukur/enwik9", None, "train", "text", 2),
]


def _extract_text(record: dict, text_field: str) -> str:
    """Pull the text out of a dataset record, tolerating schema differences.

    Prefer the configured field; otherwise fall back to the first string-valued field.
    The exact field name varies across datasets, so we stay defensive rather than assume.
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
    slice, so the start offset is derived from the post-commitment beacon ``seed``. It is
    bounded by ``dataset_records_skip_cap`` so the oracle does not have to stream forever.
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
    skip_cap: int = 100_000,
    records: Iterable[dict] | None = None,
) -> tuple[list[bytes], dict]:
    """Stream ``source`` into ``source.chunks`` chunks of ``chunk_bytes`` bytes each.

    Skips a seed-derived number of records first (so the slice is not the fixed prefix),
    then concatenates documents until the chunks are filled. ``records`` is injectable for
    tests; in production it comes from a streaming ``datasets`` iterator. Returns the chunk
    byte blobs plus a provenance entry recording exactly what was drawn.
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
    skip_cap: int = 100_000,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Glyph corpus from large mixed sources")
    parser.add_argument("--out-dir", default="./corpus")
    parser.add_argument(
        "--seed",
        required=True,
        help="Post-commitment chain beacon (e.g. a recent block hash). Seeds the per-source "
        "skip offset so the drawn slice is unpredictable at commit time.",
    )
    parser.add_argument(
        "--chunk-bytes", type=int, default=2 * 2**20, help="Bytes per chunk file (default 2 MiB)"
    )
    parser.add_argument(
        "--skip-cap",
        type=int,
        default=100_000,
        help="Upper bound on seed-derived records skipped before sampling each source",
    )
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for gated datasets")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)

    write_mixed_corpus(
        out_dir,
        MIXED_SOURCES,
        args.chunk_bytes,
        seed=args.seed,
        token=args.hf_token,
        skip_cap=args.skip_cap,
    )

    provider = StaticLocalProvider(out_dir)
    manifest = provider.manifest()
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": manifest.version,
                "total_bytes": manifest.total_bytes,
                "manifest_hash": manifest.manifest_hash(),
                "seed": args.seed,
                "chunks": [{"id": c.id, "size": c.size, "hash": c.hash} for c in manifest.chunks],
            },
            indent=2,
        )
    )
    total_chunks = sum(s.chunks for s in MIXED_SOURCES)
    print(f"sources={len(MIXED_SOURCES)} chunks={total_chunks} total_bytes={manifest.total_bytes:,}")
    print(f"corpus written to {out_dir}")
    print(f"manifest_hash={manifest.manifest_hash()}  (commit this on-chain)")


if __name__ == "__main__":
    main()
