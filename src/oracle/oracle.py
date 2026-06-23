"""Glyph data oracle (DESIGN §5): scrape fresh, attested-timestamp text into a corpus.

The owner runs this daily. It pulls recent, server-timestamped English text (so the data
provably postdates miner commitments), writes it as corpus chunk files plus a manifest,
and prints the manifest hash to commit on-chain. Validators then resolve the corpus and
verify it against that hash via ``OracleProvider``.

Only sources with *platform-attested* timestamps are used -- author-claimed dates are
forgeable. The default source is the Wikipedia recent-changes stream (public, no auth).
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

from eval.corpus import StaticLocalProvider

USER_AGENT = "glyph-oracle/0.1 (+https://github.com/glyph-research/glyph-subnet)"
WIKI_API = "https://en.wikipedia.org/w/api.php"


def _api_get(params: dict) -> dict:
    url = f"{WIKI_API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_wikipedia_recent(target_bytes: int, max_pages: int = 500) -> list[tuple[str, str, str]]:
    """Return [(title, server_timestamp, plain_text)] from recent main-namespace edits."""

    changes = _api_get(
        {
            "action": "query",
            "list": "recentchanges",
            "rcnamespace": 0,
            "rctype": "edit",
            "rcprop": "title|timestamp",
            "rclimit": min(max_pages, 500),
            "format": "json",
        }
    )
    seen: set[str] = set()
    out: list[tuple[str, str, str]] = []
    total = 0
    for change in changes.get("query", {}).get("recentchanges", []):
        title = change.get("title")
        timestamp = change.get("timestamp", "")  # server-attested
        if not title or title in seen:
            continue
        seen.add(title)
        try:
            extract = _api_get(
                {
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "titles": title,
                    "format": "json",
                }
            )
            pages = extract.get("query", {}).get("pages", {})
            text = next(iter(pages.values()), {}).get("extract", "")
        except Exception:
            continue
        if len(text) < 200:
            continue
        out.append((title, timestamp, text))
        total += len(text.encode("utf-8"))
        if total >= target_bytes:
            break
    return out


def write_corpus(out_dir: Path, documents: list[tuple[str, str, str]], chunk_bytes: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("chunk_*.txt"):
        stale.unlink()
    buffer = bytearray()
    index = 0
    provenance = []
    for title, timestamp, text in documents:
        provenance.append({"title": title, "timestamp": timestamp, "bytes": len(text.encode())})
        buffer += (text.rstrip() + "\n\n").encode("utf-8")
        while len(buffer) >= chunk_bytes:
            (out_dir / f"chunk_{index:05d}.txt").write_bytes(buffer[:chunk_bytes])
            del buffer[:chunk_bytes]
            index += 1
    if buffer:
        (out_dir / f"chunk_{index:05d}.txt").write_bytes(bytes(buffer))
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a fresh Glyph corpus from attested sources")
    parser.add_argument("--out-dir", default="./corpus")
    parser.add_argument("--source", choices=["wikipedia"], default="wikipedia")
    parser.add_argument("--target-bytes", type=int, default=8 * 2**20)
    parser.add_argument("--chunk-bytes", type=int, default=1 * 2**20)
    parser.add_argument("--max-pages", type=int, default=500)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)

    if args.source == "wikipedia":
        documents = fetch_wikipedia_recent(args.target_bytes, max_pages=args.max_pages)
    else:  # pragma: no cover
        raise SystemExit(f"unsupported source: {args.source}")

    if not documents:
        raise SystemExit("no documents fetched; check connectivity")

    write_corpus(out_dir, documents, args.chunk_bytes)
    manifest = StaticLocalProvider(out_dir).manifest()
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": manifest.version,
                "total_bytes": manifest.total_bytes,
                "manifest_hash": manifest.manifest_hash(),
                "chunks": [{"id": c.id, "size": c.size, "hash": c.hash} for c in manifest.chunks],
            },
            indent=2,
        )
    )
    print(f"documents={len(documents)} total_bytes={manifest.total_bytes:,}")
    print(f"corpus written to {out_dir}")
    print(f"manifest_hash={manifest.manifest_hash()}  (commit this on-chain)")


if __name__ == "__main__":
    main()
