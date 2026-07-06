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

## Temporal burn (weight_setter)

**Currently disabled** (`core.constants.BURN_ENABLED = False`, issue #43): no tempo is ever
a burn tempo, `weight_setter.decide_weights` always short-circuits `burn` to `False`, and
all emission flows to the normal rolling-winner distribution (70% / 30%) every tempo. The
schedule below describes the feature as designed and as it runs when re-enabled.

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

Effective time-averaged split when enabled: **52.5% winner / 22.5% previous / 25% burned**.
The burn also reduces continuous alpha sell pressure during long reigns. **Note:** the
temporal schedule was also the anti-copy signal described above -- with it disabled, a
weight-copying validator is no longer penalised by burn-tempo divergence specifically
(commit-reveal and the earliest-commit winner tie-break still apply).

To re-enable: flip `BURN_ENABLED = True` in `src/core/constants.py` and ship that source
change to every validator together (this is consensus-critical and must never be a
per-operator runtime override -- see the constant's own comment and the module header note
in `core/constants.py`).
