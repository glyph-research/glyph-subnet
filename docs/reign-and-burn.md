# Reign and temporal burn

## King-of-the-hill (reign_worker)

Each round, the incumbent and all new challengers are evaluated on the **same**
beacon-seeded streams (paired comparison kills sampling variance). Rules:

- Score = compressed ÷ raw bytes (lower is better), gated on bit-exact round-trip on every
  stream, a 10 KiB/s decompress floor, and the compress budget.
- A challenger takes the crown only by beating the incumbent's ratio by **ε = 0.5%**
  (relative). Ties go to the **earliest commit block**, so copying a public codec is
  worthless.
- A challenger that does not win is **excluded forever** (one shot). The registration burn
  is therefore the spam-proof submission fee.
- The previous winner is a hot standby: if the incumbent's artifact vanishes or fails
  re-evaluation, it is promoted back.

Weights for the two live slots: **current winner 70% / previous winner 30%**.

## Temporal burn (weight_setter)

A 25% daily burn is applied *in time*, not as a static carve-out (DESIGN §6.1):

- Tempos are grouped into windows of 4. Exactly one tempo per window is a **burn tempo**:
  weights go 100% to UID 0.
- The burn position is `H(S ‖ window) mod 4`, where the seed `S` is derived from the most
  recent challenge round's per-stream outputs (sizes + blob hashes). Only validators that
  actually ran (or replayed) the evaluation can compute `S`.
- Combined with native **commit-reveal** weights, a copy-cat validator that never evaluates
  cannot reproduce the schedule: it guesses the burn position wrong 3 windows out of 4 (or
  copies stale reveals), diverging maximally on the tempos it gets wrong — which Yuma
  bonding punishes hard.

Effective time-averaged split: **52.5% winner / 22.5% previous / 25% burned**. The burn
also reduces continuous alpha sell pressure during long reigns.
