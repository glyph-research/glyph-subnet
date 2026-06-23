"""Scrape fresh, server-timestamped Wikipedia text into a corpus (testnet benchmark).

Efficient batched fetch via the MediaWiki recentchanges generator + extracts. Writes
20 MiB chunk files plus a provenance log. Usage:

    python scripts/scrape_fresh_corpus.py <target_bytes> <out_dir>
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
UA = "glyph-oracle/0.1 (testnet compression benchmark; contact poleandjerry@gmail.com)"
CHUNK = 20 * 2**20


def get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "maxlag": 5})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(12):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                wait = int(e.headers.get("Retry-After", 0) or 0) or min(60, 5 * (attempt + 1))
                print(f"  rate-limited ({e.code}); sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
        except Exception:
            if attempt >= 11:
                raise
            time.sleep(3 * (attempt + 1))
    return {}


def main() -> None:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 85 * 2**20
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "fresh_corpus")
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("chunk_*.txt"):
        stale.unlink()

    # generator=random returns 20 random *current* (latest-version) articles per call --
    # real, diverse Wikipedia text, no continuation/dedup death-spiral.
    params = {
        "action": "query", "format": "json", "generator": "random",
        "grnnamespace": 0, "grnlimit": "20",
        "prop": "extracts", "explaintext": 1, "exlimit": "max",
    }
    seen: set[str] = set()
    provenance = []
    buf = bytearray()
    chunk_idx = 0
    total = 0
    calls = 0

    while total < target:
        data = get(params)
        calls += 1
        pages = data.get("query", {}).get("pages", {})
        for p in pages.values():
            title = p.get("title")
            text = p.get("extract", "")
            if not title or title in seen or len(text) < 200:
                continue
            seen.add(title)
            blob = (text.rstrip() + "\n\n").encode("utf-8")
            buf += blob
            total += len(blob)
            provenance.append({"title": title, "bytes": len(blob)})
            while len(buf) >= CHUNK:
                (out / f"chunk_{chunk_idx:05d}.txt").write_bytes(bytes(buf[:CHUNK]))
                del buf[:CHUNK]
                chunk_idx += 1
        if calls % 25 == 0:
            print(f"  {total/2**20:.1f} MiB / {target/2**20:.0f} MiB ({len(seen)} articles, {calls} calls)", flush=True)
        time.sleep(0.4)

    if buf:
        (out / f"chunk_{chunk_idx:05d}.txt").write_bytes(bytes(buf))
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"DONE: {total/2**20:.1f} MiB, {len(seen)} fresh articles -> {out}", flush=True)


if __name__ == "__main__":
    main()
