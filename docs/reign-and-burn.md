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

A 10% daily burn is applied *in time*, not as a static carve-out (issue #168 reduced the
cadence from 25% / windows of 4):

- Tempos are grouped into windows of 10. Exactly one tempo per window is a **burn tempo**:
  weights go 100% to UID 0.
- The burn position is `H(S ‖ window) mod 10`, where the seed `S` is derived from the most
  recent challenge round's per-stream outputs (sizes + blob hashes). Only validators that
  actually ran (or replayed) the evaluation can compute `S`.
- Combined with native **commit-reveal** weights, a copy-cat validator that never evaluates
  cannot reproduce the schedule: it guesses the burn position wrong 9 windows out of 10 (or
  copies stale reveals), diverging maximally on the tempos it gets wrong — which Yuma
  bonding punishes hard. Detection is less frequent than at windows of 4, but every window
  still contains a guaranteed-wrong tempo at an unpredictable position, and each hit costs
  the copier the same.

Effective time-averaged split: **63% winner / 27% previous / 10% burned**. The burn also
reduces continuous alpha sell pressure during long reigns.

To disable (or re-enable after disabling): flip `BURN_ENABLED` in `src/core/constants.py` and
ship that source change to every validator together (this is consensus-critical and must
never be a per-operator runtime override -- see the constant's own comment and the module
header note in `core/constants.py`). With it disabled, no tempo is ever a burn tempo,
`weight_setter.decide_weights` always short-circuits `burn` to `False`, all emission flows to
the normal rolling-winner distribution (70% / 30%) every tempo, and a weight-copying
validator is no longer penalised by burn-tempo divergence specifically (commit-reveal and the
earliest-commit winner tie-break still apply).

### Owner emergency burn override (issue #113)

A code-level `BURN_ENABLED` flip needs a deploy across every validator to take effect --
too slow for an active incident. The subnet owner (whichever hotkey currently occupies
`BURN_UID` on the live metagraph) can instead publish `{v, force_burn}` on its own
commitment slot (prefix `g1b|`, distinct from every miner/winner commitment form). Every
validator checks it each tempo: `force_burn=true` forces 100% burn unconditionally,
overriding the normal schedule and `BURN_ENABLED` alike, effective network-wide with no code
deploy. A missing, malformed, or `force_burn=false` commitment falls through unchanged to
today's schedule -- additive only, this can force a burn tempo but never suppress one.
Reading it costs nothing extra: `get_all_commitments()` is already fetched every round for
codec precheck / weight-setting.

Note the owner's `BURN_UID`-occupant hotkey also holds #103's per-validator winner-commitment
slot if it happens to run a validator itself -- a single commitment slot can't hold both
payloads at once. Use a **separate, dedicated hotkey purely for governance signaling** (this
override and any future owner-only signal), holding no codec commitment and running no
validator, so its slot is reserved for this.
