"""King-of-the-hill reign worker.

Runs one challenge round: the incumbent and all challengers are evaluated on identical
beacon-seeded streams (paired), the crown changes only on a strict epsilon ratio beat,
losers are excluded forever (one shot), and the standing winner's per-stream outputs
become the temporal-burn seed. Used as a library by the validator orchestrator and
runnable as its own PM2 process.
"""

from __future__ import annotations

from bittensor.utils.btlogging import logging as bt_logging

from eval.evaluator import paired_eval
from eval.runner import ArtifactRef, LocalSubprocessRunner, ResourceCaps
from core.artifact import local_snapshot_dir
from core.constants import SCORING_VERSION
from core.state import CommitmentState, ScoreState, ValidatorState
from core.weights import WinnerEntry, promote_winner, rank_key, should_promote


def find_winner_commitment(state: ValidatorState, winner: WinnerEntry) -> CommitmentState | None:
    for commitment in state.commitments.values():
        if (
            commitment.hotkey == winner.hotkey
            and commitment.repo == winner.repo
            and commitment.revision == winner.revision
            and commitment.valid
        ):
            return commitment
    return None


def artifact_ref(commitment: CommitmentState, runner) -> ArtifactRef:
    """Build an ArtifactRef for the configured runner.

    The Chutes runner downloads ``repo@rev`` inside the chute. A runner that executes
    locally (``LocalSubprocessRunner``, ``DockerRunner``) needs the artifact on disk, so it
    snapshot-downloads it here -- flagged via ``needs_local_artifact`` rather than an
    isinstance chain so new local-execution runners don't need a change here.
    """

    if isinstance(runner, LocalSubprocessRunner) or getattr(runner, "needs_local_artifact", False):
        local = getattr(commitment, "local_path", None)
        if not local:
            from huggingface_hub import snapshot_download

            # local_dir materializes real files rather than the cache's default
            # symlinks-into-blobs/ layout, which dangle once this directory is bind-mounted
            # into a container alone, without the separate blobs/ dir they point into (#66).
            local = snapshot_download(
                repo_id=commitment.repo,
                revision=commitment.revision,
                local_dir=local_snapshot_dir(commitment.repo, commitment.revision),
            )
        return ArtifactRef(
            repo=commitment.repo, rev=commitment.revision, sha256=commitment.artifact_hash, local_path=local
        )
    return ArtifactRef(repo=commitment.repo, rev=commitment.revision, sha256=commitment.artifact_hash)


def run_round(
    state: ValidatorState,
    runner,
    challengers: list[CommitmentState],
    provider,
    stream_specs,
    *,
    caps: ResourceCaps,
    floor_bps: float,
    budget_secs: float,
    margin: float,
    block: int | None,
    eligible_hotkeys: set[str],
    baseline_ratio: float | None = None,
) -> dict:
    """Evaluate incumbent + challengers on identical streams and update the crown.

    Returns the raw ``{hotkey: EvalOutcome}`` this round produced (e.g. for wandb's
    per-source-breakdown reporting, issue #41) -- purely a read of what was already
    computed; nothing here changes because it's returned.
    """

    artifacts: list[tuple[str, ArtifactRef]] = []
    incumbent = state.winner_history[0] if state.winner_history else None
    incumbent_commitment = find_winner_commitment(state, incumbent) if incumbent else None
    if incumbent_commitment is not None:
        artifacts.append((incumbent_commitment.hotkey, artifact_ref(incumbent_commitment, runner)))
    for challenger in challengers:
        artifacts.append((challenger.hotkey, artifact_ref(challenger, runner)))

    outcomes = paired_eval(
        runner, artifacts, provider, stream_specs, caps=caps, floor_bps=floor_bps, budget_secs=budget_secs
    )

    # Re-score every codec on this round's fresh streams (also refreshes the incumbent's ratio).
    for hotkey, outcome in outcomes.items():
        if outcome.score.valid:
            bt_logging.info(f"candidate {hotkey}: ratio={outcome.score.ratio:.4f} valid")
        else:
            bt_logging.warning(f"candidate {hotkey}: invalid ({outcome.score.reasons})")
        commitment = next((c for c in state.commitments.values() if c.hotkey == hotkey and c.valid), None)
        if commitment is None:
            continue
        state.scores[commitment.key] = ScoreState(
            hotkey=hotkey,
            repo=commitment.repo,
            revision=commitment.revision,
            ratio=outcome.score.ratio,
            roundtrip_ok=all(r.roundtrip_ok for r in outcome.results if r.scored)
            if any(r.scored for r in outcome.results)
            else False,
            throughput_bps=outcome.score.throughput_bps_min,
            valid=outcome.score.valid,
            commit_block=commitment.block or 0,
            evaluated_at_block=block,
            scoring_version=SCORING_VERSION,
        )

    current_ratio = None
    if incumbent_commitment is not None and incumbent_commitment.hotkey in outcomes:
        inc_outcome = outcomes[incumbent_commitment.hotkey]
        if inc_outcome.score.valid:
            current_ratio = inc_outcome.score.ratio
            state.winner_history[0] = state.scores[incumbent_commitment.key].as_winner()
        else:
            # Incumbent failed its own re-evaluation -> actually vacate the crown (issue #67).
            # rolling_weights_for_hotkeys only looks at winner_history's presence, never
            # re-checks state.scores[...].valid, so leaving this entry in place would let a
            # codec that just broke keep earning weight indefinitely whenever no challenger
            # happens to appear in a given round. Popping shifts the hot standby (index 1, if
            # any) up to index 0, so it's naturally treated as the incumbent next round --
            # "promotes later" needs no special-casing here, current_ratio already stays None
            # so a valid challenger THIS round still vacant-crown-promotes immediately.
            bt_logging.warning(f"incumbent {incumbent_commitment.hotkey} failed re-eval: {inc_outcome.score.reasons}")
            state.winner_history.pop(0)

    ranked = sorted(
        (c for c in challengers if outcomes.get(c.hotkey) and outcomes[c.hotkey].score.valid),
        key=lambda c: rank_key(state.scores[c.key].as_winner()),
    )
    winner_outputs = None
    for challenger in challengers:
        outcome = outcomes.get(challenger.hotkey)
        if outcome and outcome.score.valid:
            continue  # ranked promotion handles valid ones below
        # invalid challenger -> one-shot exclusion
        state.excluded_hotkeys.add(challenger.hotkey)

    for challenger in ranked:
        challenger_ratio = state.scores[challenger.key].ratio
        beats_baseline = baseline_ratio is None or challenger_ratio < baseline_ratio
        if should_promote(challenger_ratio, current_ratio, margin) and beats_baseline:
            state.winner_history = promote_winner(
                state.winner_history,
                state.scores[challenger.key].as_winner(),
                eligible_hotkeys=eligible_hotkeys,
            )
            current_ratio = challenger_ratio
            winner_outputs = outcomes[challenger.hotkey].burn_outputs()
            bt_logging.info(f"new winner: {challenger.hotkey} ratio={challenger_ratio:.4f}")
        else:
            # Challenged and lost -> one shot, excluded from future rounds.
            state.excluded_hotkeys.add(challenger.hotkey)

    # Burn seed = the standing winner's per-stream outputs this round.
    if winner_outputs is None and incumbent_commitment is not None:
        inc = outcomes.get(incumbent_commitment.hotkey)
        if inc and inc.score.valid:
            winner_outputs = inc.burn_outputs()
    if winner_outputs is not None:
        state.last_round_outputs = list(winner_outputs)

    return outcomes


def main() -> None:
    """Standalone reign worker: run a single challenge round against chain + corpus."""

    from core.dotenv import load_dotenv
    from core.log_config import configure_logging
    from validator.service import run_reign_only, build_parser

    load_dotenv()
    args = build_parser().parse_args()
    configure_logging(args)
    run_reign_only(args)


if __name__ == "__main__":
    main()
