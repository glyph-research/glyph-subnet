"""Live-data benchmark stream (issue #139): fresh text no model could have memorized.

Resurrects the pre-#19 live Wikipedia sourcing (``scripts/scrape_fresh_corpus.py`` in git
history) as a **benchmark-only** stream, like enwik9: shown per challenger in logs/wandb,
never scored, so per-validator differences in fetched bytes are irrelevant to consensus.
The corpus-poisoning concern that removed live sourcing from *scoring* (exploit vector #10)
does not apply to an unscored display stream, and the eval sandbox runs codecs with
networking disabled, so a codec cannot fetch the same live data at eval time.

Pipeline: after each round, during the idle wait, a daemon thread fetches >=4 MiB of
recently-changed Wikipedia article text (markup-free extracts, deduped by title). The next
round appends the latest *complete* snapshot as one ``live-0`` stream. A fetch failure,
short fetch, or hung thread never blocks, delays, or fails a round -- the round falls back
to the previous complete snapshot, or skips the live stream entirely (same
transient-vs-permanent principle as #120/#132/#135). Used snapshots are retained under the
state dir up to ``LIVE_RETENTION_BYTES`` for later re-examination, oldest deleted first.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from bittensor.utils.btlogging import logging as bt_logging

from eval.corpus import CorpusManifest
from eval.streams import RangeSource, StreamSpec

LIVE_SOURCE = "live"
# The fetch must reliably land >=4 MiB inside the ~100-block (~20 min) between-rounds
# window; the deadline leaves headroom so a slow fetch is abandoned before the next round.
LIVE_MIN_SNAPSHOT_BYTES = 4 * 2**20
LIVE_FETCH_DEADLINE_SECS = 15 * 60.0
LIVE_RETENTION_BYTES = 1 * 2**30

_API = "https://en.wikipedia.org/w/api.php"
_UA = "glyph-validator/1.0 (live compression benchmark; contact poleandjerry@gmail.com)"


def _api_get(params: dict) -> dict:
    url = _API + "?" + urllib.parse.urlencode({**params, "maxlag": 5})
    request = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_live_text(
    min_bytes: int = LIVE_MIN_SNAPSHOT_BYTES,
    *,
    deadline_secs: float = LIVE_FETCH_DEADLINE_SECS,
    api_get=_api_get,
    politeness_secs: float = 0.4,
) -> bytes | None:
    """Fetch >=``min_bytes`` of recently-changed Wikipedia article text, or ``None``.

    MediaWiki ``generator=recentchanges`` + plain-text extracts (the batched pattern the
    pre-#19 scraper proved out with ``generator=random``): full current text of
    recently-edited/created mainspace articles, bot edits and redirects skipped, deduped by
    title so an article edited twice in the window contributes once. Returns ``None`` when
    the deadline passes first -- a short snapshot is never written, only complete ones.
    """

    deadline = time.monotonic() + deadline_secs
    params = {
        "action": "query", "format": "json",
        "generator": "recentchanges", "grcnamespace": 0,
        "grctype": "edit|new", "grcshow": "!bot|!redirect", "grclimit": 50,
        "prop": "extracts", "explaintext": 1, "exlimit": "max",
    }
    seen: set[str] = set()
    buf = bytearray()
    cont: dict = {}
    while len(buf) < min_bytes and time.monotonic() < deadline:
        try:
            data = api_get({**params, **cont})
        except Exception as exc:  # noqa: BLE001 - transient API trouble: retry until deadline
            bt_logging.debug(f"live benchmark: fetch call failed, retrying: {exc}")
            time.sleep(min(5.0, max(politeness_secs, 1.0)))
            continue
        for page in (data.get("query", {}).get("pages", {}) or {}).values():
            title = page.get("title")
            text = page.get("extract", "")
            if not title or title in seen or len(text) < 200:
                continue
            seen.add(title)
            buf += (text.rstrip() + "\n\n").encode("utf-8")
        # Page through the recent-changes window; when exhausted, restart from the top --
        # new changes accumulate continuously, and `seen` keeps repeats out.
        cont = data.get("continue") or {}
        if politeness_secs:
            time.sleep(politeness_secs)
    if len(buf) < min_bytes:
        return None
    return bytes(buf)


class LiveSnapshotStore:
    """Complete live snapshots under ``directory``, named ``<block>.txt``.

    Writes go to a ``.part`` file first and are renamed into place, so ``latest()`` can
    only ever observe complete snapshots. Retention is capped at ``retention_bytes`` total,
    oldest (lowest block) deleted first; the newest snapshot is always kept.
    """

    def __init__(self, directory: str | Path, *, retention_bytes: int = LIVE_RETENTION_BYTES):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._retention_bytes = retention_bytes

    def _snapshots(self) -> list[tuple[int, Path]]:
        out = []
        for path in self.directory.glob("*.txt"):
            if path.stem.isdigit():
                out.append((int(path.stem), path))
        return sorted(out)

    def save(self, block: int, data: bytes) -> Path:
        final = self.directory / f"{block}.txt"
        part = self.directory / f"{block}.txt.part"
        part.write_bytes(data)
        part.rename(final)
        self._prune()
        return final

    def latest(self) -> tuple[Path, int] | None:
        snapshots = self._snapshots()
        if not snapshots:
            return None
        block, path = snapshots[-1]
        return path, block

    def _prune(self) -> None:
        snapshots = self._snapshots()
        total = sum(path.stat().st_size for _, path in snapshots)
        while total > self._retention_bytes and len(snapshots) > 1:
            _, oldest = snapshots.pop(0)
            total -= oldest.stat().st_size
            oldest.unlink()


class LivePrefetcher:
    """Background fetch of the next round's live snapshot -- strictly best-effort.

    ``start()`` returns immediately; the fetch runs on a daemon thread and writes a
    complete snapshot (or nothing). A still-running fetch is left alone -- abandoned, never
    awaited -- so a hung fetch can neither delay the round loop nor stack up threads.
    """

    def __init__(self, store: LiveSnapshotStore, *, min_bytes: int = LIVE_MIN_SNAPSHOT_BYTES, fetch=fetch_live_text):
        self._store = store
        self._min_bytes = min_bytes
        self._fetch = fetch
        self._thread: threading.Thread | None = None

    def start(self, block: int) -> None:
        if self._thread is not None and self._thread.is_alive():
            bt_logging.warning(
                "live benchmark: previous prefetch still running; not starting another"
            )
            return
        self._thread = threading.Thread(
            target=self._run, args=(block,), name=f"live-prefetch-{block}", daemon=True
        )
        self._thread.start()

    def _run(self, block: int) -> None:
        try:
            data = self._fetch(self._min_bytes)
            if data is None:
                bt_logging.warning(
                    "live benchmark: prefetch came up short; the next round falls back to "
                    "the previous complete snapshot"
                )
                return
            path = self._store.save(block, data)
            bt_logging.info(f"live benchmark: snapshot ready: {path.name} ({len(data):,} bytes)")
        except Exception as exc:  # noqa: BLE001 - must never propagate into the round loop
            bt_logging.warning(f"live benchmark: prefetch failed: {exc}")


class SnapshotAppendedProvider:
    """A corpus provider with a live snapshot appended after the base corpus bytes.

    Streams that lie inside the base range are served (and range-sourced) by the base
    provider untouched; the trailing ``live`` region is served inline from memory --
    ``stream_source`` returns None there, so every runner falls back to inlined bytes.
    """

    def __init__(self, base, snapshot: bytes):
        self._base = base
        self._snapshot = snapshot

    @property
    def total_bytes(self) -> int:
        return self._base.total_bytes + len(self._snapshot)

    def manifest(self) -> CorpusManifest:
        return self._base.manifest()

    def read_range(self, offset: int, length: int) -> bytes:
        base_total = self._base.total_bytes
        end = offset + length
        out = bytearray()
        if offset < base_total:
            out += self._base.read_range(offset, min(end, base_total) - offset)
        if end > base_total:
            live_start = max(offset - base_total, 0)
            out += self._snapshot[live_start : end - base_total]
        return bytes(out)

    def materialize(self, spec: StreamSpec) -> bytes:
        return self.read_range(spec.offset, spec.length)

    def stream_source(self, spec: StreamSpec) -> RangeSource | None:
        if spec.offset >= self._base.total_bytes:
            return None
        return self._base.stream_source(spec)

    def source_range(self, source: str) -> tuple[int, int] | None:
        if source == LIVE_SOURCE:
            return (self._base.total_bytes, len(self._snapshot))
        if hasattr(self._base, "source_range"):
            return self._base.source_range(source)
        return None


def live_benchmark_spec(base_provider, snapshot: bytes) -> StreamSpec:
    """The single benchmark-only ``live-0`` stream spanning the appended snapshot."""

    return StreamSpec(
        stream_id="live-0",
        offset=base_provider.total_bytes,
        length=len(snapshot),
        source=LIVE_SOURCE,
        scored=False,
    )
