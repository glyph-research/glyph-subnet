# Reign and temporal burn

## King-of-the-hill (reign_worker)

Each round, the incumbent and all new challengers are evaluated on the **same**
beacon-seeded streams (paired comparison kills sampling variance). Rules:

- Score = compressed ÷ raw bytes (lower is better), gated on bit-exact round-trip on every
  stream, a 10 KiB/s decompress floor, and the compress budget.
- A challenger takes the crown only by beating the incumbent's ratio by **ε = 5%**
  (relative; launch setting `win_margin = 0.05`). Ties go to the **earliest commit block**,
  so copying a public codec is worthless.
- A challenger that does not win is **excluded forever** (one shot). The registration burn
  is therefore the spam-proof submission fee.
- The previous winner is a hot standby: if the incumbent's artifact vanishes or fails
  re-evaluation, it is promoted back.

Weights for the two live slots: **current winner 70% / previous winner 30%**.

### On-chain winner commitment (observability only, issue #103)

Whenever the crown changes, the validator publishes a small record — `{hotkey, repo, rev,
ratio, commit_block, scoring_version}` — on its **own** hotkey's commitment slot (the same
`chain.set_commitment` mechanism miners use, otherwise unused by any validator). This is a
cheap, auditable trail of crown changes for tooling/dashboards and a bootstrapping
cross-check signal — **it is never read back into that same validator's own scoring or
promotion**. A validator always independently re-benchmarks the on-chain codec commitments
(reigning champion included) every round regardless of what's published here; a fresh
machine keeps doing exactly that, treating the current champion as just another challenger.
A publish failure is logged and otherwise ignored — it never crashes or delays a round.

## Temporal burn (weight_setter)

**Currently enabled** (`core.constants.BURN_ENABLED = True`). Disabled at launch (issue #43),
re-enabled (issue #88) after a live weight-copier was observed on netuid 117 -- a UID
mirroring the validator's exact weight vector each round without running the real
evaluation, exactly the attack this schedule exists to punish.

A 25% daily burn is applied *in time*, not as a static carve-out:

- Tempos are grouped into windows of 4. Exactly one tempo per window is a **burn tempo**:
  weights go 100% to UID 0.
- The burn position is `H(S ‖ window) mod 4`, where the seed `S` is derived from the most
  recent challenge round's per-stream outputs (sizes + blob hashes). Only validators that
  actually ran (or replayed) the evaluation can compute `S`.
- Combined with native **commit-reveal** weights, a copy-cat validator that never evaluates
  cannot reproduce the schedule: it guesses the burn position wrong 3 windows out of 4 (or
  copies stale reveals), diverging maximally on the tempos it gets wrong — which Yuma
  bonding punishes hard.

Effective time-averaged split: **52.5% winner / 22.5% previous / 25% burned**. The burn also
reduces continuous alpha sell pressure during long reigns.

To disable (or re-enable after disabling): flip `BURN_ENABLED` in `src/core/constants.py` and
ship that source change to every validator together (this is consensus-critical and must
never be a per-operator runtime override -- see the constant's own comment and the module
header note in `core/constants.py`). With it disabled, no tempo is ever a burn tempo,
`weight_setter.decide_weights` always short-circuits `burn` to `False`, all emission flows to
the normal rolling-winner distribution (70% / 30%) every tempo, and a weight-copying
validator is no longer penalised by burn-tempo divergence specifically (commit-reveal and the
earliest-commit winner tie-break still apply).
