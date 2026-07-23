# Reign and temporal burn

## King-of-the-hill (reign_worker)

Each round, the incumbent and all new challengers are evaluated on the **same**
beacon-seeded streams (paired comparison kills sampling variance). Rules:

- Score = compressed ÷ raw bytes (lower is better), gated on bit-exact round-trip on every
  stream, a 10 KiB/s decompress floor, and the compress budget.
- A challenger takes the crown only by beating the incumbent's ratio by **ε = 1%**
  (relative; `win_margin = 0.01`, lowered from 5% in issue #177 now that emission is
  proportional to improvement — a marginal win earns a marginal share instead of being
  refused). Ties go to the **earliest commit block**, so copying a public codec is
  worthless.
- A challenger that does not win is **excluded forever** (one shot). The registration burn
  is therefore the spam-proof submission fee.
- The previous winner is a hot standby: if the incumbent's artifact vanishes or fails
  re-evaluation, it is promoted back.

### Emission follows improvement (issue #177)

There are no fixed 70/30 slots any more. Each winner earns a share of the pot for **how
much it moved the frontier**:

    share = 25% (top payee only) + 15% x (percent improvement over the winner it dethroned)

Improvement is measured and recorded **at promotion time** — challenger vs the incumbent it
beat, on identical streams, the only moment the comparison is meaningful — and is never
recomputed later. A winner that took a vacant crown improved on nothing and records `0`,
earning the base while it is top of the ladder and nothing once it is deeper.

Shares are assigned newest-first, each capped by whatever is left of the pot, so:

- a **≥5% jump** computes to ≥100% and takes the entire pot — prior winners earn nothing
  that tempo. There is deliberately no per-winner ceiling.
- a run of **small wins** spreads the pot: at the 1% minimum the top payee earns 40% and
  each one below 15%, so five winners are paid before it runs out.
- if the shares **under-subscribe** the pot they are scaled up to fill it, so emission
  always reaches winners; the pot burns only when nobody qualifies at all.

Because the margin is only 1%, a miner holding a large improvement could release it in ~1%
slices across several hotkeys. Measured, that is **exactly neutral for the slicer** — the
share formula is linear in improvement, so `25 + 15x3` pays the same whether the 3% arrives
as one entry or three — and it is weakly *unfavourable to everyone else*, since the extra
rungs push older winners down and can displace one off the bottom of the pot. What limits
it is that the **base is paid once, to the top payee**, so it cannot be collected per
slice (the original concern); plus the registration cost of each extra hotkey and each
slice having to satisfy its own conviction requirement. Every promotion logs its
improvement, so the distribution over time makes slicing visible — worth watching in the
data rather than pre-emptively penalising honest small wins.

Payees are the **conviction-compliant** winners in the retained history (issue #170):
weight-setting walks the history newest-first, skipping any entry that is no longer
eligible (deregistered/excluded) or below its required conviction, and allocates to the
ones it finds until the pot is gone — or burns if none qualifies. Retention keeps
`WINNER_HISTORY_DEPTH = 20` entries so there is a deep fallback ladder (issue #175),
making a burn close to a last resort; the crown itself is always `history[0]` regardless
of compliance, so scoring, dethroning, and the commit-order gauntlet are untouched by all
of this. A dethroned winner that keeps its conviction locked stays in the queue and earns
again whenever a more recent winner falls out of compliance. Note the accepted consequence
of a deep ladder: an entry whose cumulative earnings are below the `1000 α` free allowance
requires *zero* conviction and so can be paid with nothing locked — flowing beats burning,
and it can collect at most that allowance before its own requirement gates it.

Compliance is evaluated lazily down the ladder: weight-setting stops reading once the pot
is exhausted, so the per-tempo chain cost tracks how many winners actually get paid (one
read when a big improvement takes everything, five at the 1% minimum) rather than the
retained depth. Each query is isolated — one failed read falls back to the staked rule for
that hotkey alone.

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

Effective time averages: **90% of emission to winners, 10% burned**, with the winner
portion split by improvement (see above) rather than by fixed slots. The burn also reduces
continuous alpha sell pressure during long reigns.

To disable (or re-enable after disabling): flip `BURN_ENABLED` in `src/core/constants.py` and
ship that source change to every validator together (this is consensus-critical and must
never be a per-operator runtime override -- see the constant's own comment and the module
header note in `core/constants.py`). With it disabled, no tempo is ever a burn tempo,
`weight_setter.decide_weights` always short-circuits `burn` to `False`, all emission flows to
the normal improvement-proportional winner distribution every tempo, and a weight-copying
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
