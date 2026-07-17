"""Best-effort Weights & Biases observability for the validator (issue #41).

Pure side effect: this module must never read back into or alter scoring, promotion,
weights, or burn -- it only reports what the validator already decided. Every wandb call is
wrapped so a wandb/network/auth failure can never crash or delay a round; with logging
disabled (``--wandb.off``) the validator behaves byte-identically to a build without this
module at all (nothing here is imported/executed on that path beyond the no-op logger).

Default posture: ON. No ``WANDB_API_KEY`` configured -> falls back to an anonymous run
(shareable link, no login required to view) rather than blocking or crashing; ``--wandb.offline``
switches to fully local (no network) logging instead.
"""

from __future__ import annotations

import os
import time
import traceback


class WandbLogger:
    """No-op unless enabled; every public method is best-effort and never raises."""

    # Consecutive-failure budget before logging fully disables itself (issue #127): one
    # transient network blip or wandb-internal hiccup (observed live: HandleAbandonedError
    # from a run torn down during a pm2 restart) should cost that single call, not blind
    # observability until the next scheduled restart (default 24h). Each failed call drops
    # the run handle so the next call retries with a fresh _start_run(); only a streak of
    # failures -- every one of which already got that fresh-run retry -- is treated as
    # "wandb is fundamentally broken, stop trying".
    _MAX_CONSECUTIVE_FAILURES = 3

    def __init__(
        self,
        *,
        enabled: bool = True,
        project: str = "text-compression",
        entity: str | None = "glyph-research-org",
        name: str | None = None,
        offline: bool = False,
        notes: str | None = None,
        restart_interval_hours: float = 24.0,
    ):
        self.enabled = enabled
        self._project = project
        self._entity = entity
        self._name = name
        self._offline = offline
        self._notes = notes
        self._restart_interval_secs = max(restart_interval_hours, 0.0) * 3600.0
        self._run = None
        self._run_started_at = 0.0
        self._consecutive_failures = 0
        if self.enabled:
            self._safe(self._start_run)

    # --- internals -----------------------------------------------------------------------

    def _safe(self, fn, *args, **kwargs):
        """Run ``fn`` best-effort. An exception never propagates -- a wandb outage degrades
        to no-op observability, not a broken validator round. A single failure only skips
        that call (the dead run handle is dropped so the next call starts fresh); logging
        fully disables only after ``_MAX_CONSECUTIVE_FAILURES`` failures in a row."""

        if not self.enabled:
            return None
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - wandb must never crash/delay a round
            self._consecutive_failures += 1
            # The run may be torn down/abandoned; drop it so the next call attempts a
            # fresh _start_run() instead of re-poking a dead handle.
            self._run = None
            if self._consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
                print(
                    f"[wandb] disabling logging after {self._consecutive_failures} "
                    f"consecutive errors: {exc!r}"
                )
                traceback.print_exc()
                self.enabled = False
            else:
                print(
                    f"[wandb] call failed ({self._consecutive_failures}/"
                    f"{self._MAX_CONSECUTIVE_FAILURES} consecutive), retrying with a "
                    f"fresh run on the next call: {exc!r}"
                )
            return None
        self._consecutive_failures = 0
        return result

    def _start_run(self) -> None:
        import wandb

        mode = "offline" if self._offline else "online"
        # No API key configured -> anonymous mode: wandb still creates a real, shareable run
        # (viewable without login) instead of failing or silently going dark.
        anonymous = None if os.environ.get("WANDB_API_KEY") else "allow"
        self._run = wandb.init(
            project=self._project,
            entity=self._entity,
            name=self._name,
            mode=mode,
            notes=self._notes,
            anonymous=anonymous,
            # Explicit even though "wrap" is wandb's own default: guarantees stdout/stderr
            # (the validator's existing print(...) lines) land in the run's Logs tab.
            settings=wandb.Settings(console="wrap"),
        )
        self._run_started_at = time.time()
        url = self._run.get_url() if self._run is not None else None
        print(f"[wandb] run started: {url or '(offline run, no public URL)'}")

    def _maybe_restart(self) -> None:
        if self._restart_interval_secs <= 0:
            return
        if time.time() - self._run_started_at < self._restart_interval_secs:
            return
        if self._run is not None:
            self._run.finish()
        self._start_run()

    # --- public API ------------------------------------------------------------------------

    def log(self, metrics: dict) -> None:
        """Log a flat dict of metrics for the current step. No-op if disabled/failed."""

        self._safe(self._log_impl, metrics)

    def _log_impl(self, metrics: dict) -> None:
        import wandb

        if self._run is None:
            # No live run: either the previous call failed (handle dropped, issue #127) or
            # the initial _start_run in __init__ failed -- retry from scratch here.
            self._start_run()
        else:
            self._maybe_restart()
        if self._run is not None:
            wandb.log(metrics)

    def finish(self) -> None:
        if not self.enabled or self._run is None:
            return
        self._safe(self._run.finish)


class _NullWandbLogger(WandbLogger):
    """Explicit no-op instance for --wandb.off, so call sites stay identical either way."""

    def __init__(self) -> None:
        super().__init__(enabled=False)


def make_wandb_logger(args, *, identity_name: str | None = None) -> WandbLogger:
    """``identity_name`` is the caller's already-resolved fallback run name (e.g. the
    validator's on-chain identity name, or its hotkey ss58 if no identity is set) -- so
    multiple validators logging into the same shared glyph-research-org/text-compression
    project are distinguishable at a glance instead of wandb's random auto-generated name.
    An explicit ``--wandb.name`` always wins over it; with neither, name stays None and
    wandb picks its own.
    """

    if getattr(args, "wandb_off", False):
        return _NullWandbLogger()
    name = getattr(args, "wandb_name", None) or identity_name
    return WandbLogger(
        enabled=True,
        project=getattr(args, "wandb_project", "text-compression") or "text-compression",
        entity=getattr(args, "wandb_entity", "glyph-research-org"),
        name=name,
        offline=getattr(args, "wandb_offline", False),
        notes=getattr(args, "wandb_notes", None),
        restart_interval_hours=getattr(args, "wandb_restart_interval", 24.0),
    )


def build_round_metrics(
    *,
    block: int,
    baseline_ratio: float,
    num_challengers: int,
    outcomes: dict,
    excluded_hotkeys_count: int,
    commit_phase_seen_count: int,
    winner_hotkey: str | None,
    winner_ratio: float | None,
    crown_changed: bool,
    hotkey_to_uid: dict[str, int] | None = None,
) -> dict:
    """Flatten one eval round's outcomes into a wandb-loggable metrics dict.

    Pure formatting -- reads already-computed ``EvalOutcome``s, never recomputes or
    influences scoring. ``hotkey_to_uid`` (the metagraph's mapping, issue #126) labels each
    hotkey-keyed entry with its UID so following along doesn't require manually
    cross-referencing the metagraph; ``-1`` means unknown (no mapping supplied, or the
    hotkey isn't currently registered).
    """

    from eval.scoring import source_ratio_breakdown, stream_ratio

    hotkey_to_uid = hotkey_to_uid or {}
    metrics: dict = {
        "round/block": block,
        "round/baseline_ratio": baseline_ratio,
        "round/num_challengers": num_challengers,
        "round/excluded_hotkeys_count": excluded_hotkeys_count,
        "round/commit_phase_seen_count": commit_phase_seen_count,
        "winner/hotkey": winner_hotkey or "",
        "winner/uid": hotkey_to_uid.get(winner_hotkey, -1) if winner_hotkey else -1,
        "winner/ratio": winner_ratio if winner_ratio is not None else float("nan"),
        "winner/crown_changed": crown_changed,
    }
    for hotkey, outcome in outcomes.items():
        score = outcome.score
        prefix = f"challenger/{hotkey}"
        breakdown = source_ratio_breakdown(outcome.results)
        # Benchmark-only streams reported per source (issue #139): a single averaged
        # "enwik9_ratio" bucket would silently blend in the live stream's very different
        # numbers -- the whole point of the live stream is seeing that gap.
        benchmark_by_source: dict[str, list[float]] = {}
        for r in outcome.results:
            if not r.scored:
                benchmark_by_source.setdefault(r.source or "benchmark", []).append(stream_ratio(r))
        metrics[f"{prefix}/uid"] = hotkey_to_uid.get(hotkey, -1)
        metrics[f"{prefix}/ratio"] = score.ratio
        metrics[f"{prefix}/valid"] = score.valid
        # The real infra/runner failure (e.g. docker pull denied), when the codec never ran
        # at all -- distinguishes "host couldn't run it" from "codec produced wrong output"
        # in the dashboard just like in the log summary (issue #127).
        metrics[f"{prefix}/runner_error"] = outcome.error or ""
        metrics[f"{prefix}/roundtrip_ok"] = all(r.roundtrip_ok for r in outcome.results)
        metrics[f"{prefix}/throughput_bps_min"] = score.throughput_bps_min
        metrics[f"{prefix}/beats_baseline"] = bool(score.valid and score.ratio < baseline_ratio)
        # Log every scored source generically so corpus renames (e.g. the issue #112
        # fineweb -> fineweb-edu switch, which silently dropped fineweb_ratio) can't
        # desync this logger from live_corpus source labels again.
        for source, source_ratio in breakdown.items():
            key = source.replace("-", "_")
            metrics[f"{prefix}/{key}_ratio"] = source_ratio
        for source, ratios in benchmark_by_source.items():
            key = source.replace("-", "_")
            metrics[f"{prefix}/{key}_ratio"] = sum(ratios) / len(ratios)
    return metrics


def build_weights_metrics(
    *,
    block: int,
    tempo: int,
    is_burn_tempo: bool,
    uids: list[int],
    weights: list[float],
    conviction: dict | None = None,
) -> dict:
    """Flatten a weight-setting decision into a wandb-loggable metrics dict (log only --
    computed and applied identically whether or not this is ever called).

    ``conviction`` is the per-winner Miner Conviction report (issue #141): earned/staked/
    required_lock/compliant per winner hotkey, so any gating is explainable after the fact.
    """

    nonzero = [(uid, round(w, 4)) for uid, w in zip(uids, weights) if w > 0]
    metrics = {
        "weights/block": block,
        "weights/tempo": tempo,
        "weights/is_burn_tempo": is_burn_tempo,
        "weights/nonzero_count": len(nonzero),
        "weights/nonzero": str(nonzero),
    }
    for hotkey, entry in (conviction or {}).items():
        prefix = f"conviction/{hotkey}"
        metrics[f"{prefix}/earned"] = entry["earned"]
        metrics[f"{prefix}/staked"] = entry["staked"]
        metrics[f"{prefix}/required_lock"] = entry["required_lock"]
        metrics[f"{prefix}/compliant"] = entry["compliant"]
    return metrics
