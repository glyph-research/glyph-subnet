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
        if self.enabled:
            self._safe(self._start_run)

    # --- internals -----------------------------------------------------------------------

    def _safe(self, fn, *args, **kwargs):
        """Run ``fn`` best-effort. Any exception disables further logging but never propagates
        -- a wandb outage degrades to no-op observability, not a broken validator round."""

        if not self.enabled:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - wandb must never crash/delay a round
            print(f"[wandb] disabling logging after error: {exc!r}")
            traceback.print_exc()
            self.enabled = False
            return None

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


def make_wandb_logger(args) -> WandbLogger:
    if getattr(args, "wandb_off", False):
        return _NullWandbLogger()
    # Default the run name to this validator's own wallet identity, so multiple validators
    # logging into the same shared glyph-research-org/text-compression project are
    # distinguishable at a glance instead of wandb's random auto-generated name. Explicit
    # --wandb.name always wins; with neither set (e.g. args built without wallet/hotkey
    # names) name stays None and wandb picks its own.
    name = getattr(args, "wandb_name", None)
    if not name:
        wallet_name = getattr(args, "wallet_name", None)
        hotkey_name = getattr(args, "hotkey_name", None)
        if wallet_name and hotkey_name:
            name = f"{wallet_name}-{hotkey_name}"
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
) -> dict:
    """Flatten one eval round's outcomes into a wandb-loggable metrics dict.

    Pure formatting -- reads already-computed ``EvalOutcome``s, never recomputes or
    influences scoring.
    """

    from eval.scoring import source_ratio_breakdown, stream_ratio

    metrics: dict = {
        "round/block": block,
        "round/baseline_ratio": baseline_ratio,
        "round/num_challengers": num_challengers,
        "round/excluded_hotkeys_count": excluded_hotkeys_count,
        "round/commit_phase_seen_count": commit_phase_seen_count,
        "winner/hotkey": winner_hotkey or "",
        "winner/ratio": winner_ratio if winner_ratio is not None else float("nan"),
        "winner/crown_changed": crown_changed,
    }
    for hotkey, outcome in outcomes.items():
        score = outcome.score
        prefix = f"challenger/{hotkey}"
        breakdown = source_ratio_breakdown(outcome.results)
        benchmark_ratios = [stream_ratio(r) for r in outcome.results if not r.scored]
        metrics[f"{prefix}/ratio"] = score.ratio
        metrics[f"{prefix}/valid"] = score.valid
        metrics[f"{prefix}/roundtrip_ok"] = all(r.roundtrip_ok for r in outcome.results)
        metrics[f"{prefix}/throughput_bps_min"] = score.throughput_bps_min
        metrics[f"{prefix}/beats_baseline"] = bool(score.valid and score.ratio < baseline_ratio)
        if "fineweb" in breakdown:
            metrics[f"{prefix}/fineweb_ratio"] = breakdown["fineweb"]
        if "pile" in breakdown:
            metrics[f"{prefix}/pile_ratio"] = breakdown["pile"]
        if benchmark_ratios:
            metrics[f"{prefix}/enwik9_ratio"] = sum(benchmark_ratios) / len(benchmark_ratios)
    return metrics


def build_weights_metrics(
    *,
    block: int,
    tempo: int,
    is_burn_tempo: bool,
    uids: list[int],
    weights: list[float],
) -> dict:
    """Flatten a weight-setting decision into a wandb-loggable metrics dict (log only --
    computed and applied identically whether or not this is ever called)."""

    nonzero = [(uid, round(w, 4)) for uid, w in zip(uids, weights) if w > 0]
    return {
        "weights/block": block,
        "weights/tempo": tempo,
        "weights/is_burn_tempo": is_burn_tempo,
        "weights/nonzero_count": len(nonzero),
        "weights/nonzero": str(nonzero),
    }
