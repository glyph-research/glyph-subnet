"""Shared constants for the Glyph lossless-compression subnet.

Two classes of value live here:

* Network-wide constants (e.g. ``WINDOW_ANCHOR_BLOCK``) MUST be identical across every
  validator — they feed consensus (burn-tempo alignment), so they are committed in source,
  never read from per-operator environment. A coordinated change ships as a code change.
* Deployment-specific values (e.g. ``CHUTE_USERNAME``) only affect *where* a single operator
  runs and may be overridden via environment variables (documented in ``.env.example``).
"""

import os

# --- Chain / commitment -------------------------------------------------------
DEFAULT_NETUID = 117  # always overridable via --netuid
COMMITMENT_VERSION = 1
COMMITMENT_PREFIX = "glyph:"
COMPACT_COMMITMENT_PREFIX = "g1|"
# Commit-reveal of the commitment itself (exploit vector #9, front-running). A miner first
# publishes ``g1c|<sha256(repo|rev|salt)>`` (reveals nothing), then on the next block reveals
# ``g1r|repo|rev|salt``. The earliest-commit tie-break keys off the commit-phase block, so a
# mempool watcher who only learns repo|rev at reveal time can never commit earlier.
COMMIT_PHASE_PREFIX = "g1c|"
REVEAL_PHASE_PREFIX = "g1r|"
# Observability-only record of the current champion (issue #103), published by a validator
# on its own hotkey's commitment slot (unused by any other validator code path) whenever the
# crown changes. Distinct payload/prefix from the miner commit/reveal forms above, since it's
# a different kind of thing occupying the same on-chain commitment mechanism. NEVER trusted
# as ground truth by any validator's own scoring/promotion -- every validator always
# independently re-benchmarks the on-chain codec commitments (reigning champion included)
# every round; this exists purely as a cheap, auditable crown-change trail for
# tooling/dashboards and a bootstrapping cross-check signal.
WINNER_COMMITMENT_PREFIX = "g1w|"
WINNER_COMMITMENT_VERSION = 1
# On-chain, owner-controlled emergency burn override (issue #113): the subnet owner (whichever
# hotkey currently occupies BURN_UID on the live metagraph) can publish {v, force_burn} on its
# own commitment slot; force_burn=true makes every validator burn 100% every tempo regardless
# of the normal schedule, effective network-wide with no code deploy. Distinct prefix so it can
# never collide with the miner commit/reveal/compact forms above. Additive-only by design: a
# missing, malformed, or force_burn=false commitment always falls through to the unchanged
# existing schedule -- this can only ever force MORE burning, never suppress a scheduled one.
BURN_OVERRIDE_PREFIX = "g1b|"
BURN_OVERRIDE_VERSION = 1
# Commit-reveal polling/pruning (exploit vector #9 follow-ups, issue #21).
# A reveal lands ~1 block after its commit, so validators must observe commitments at
# roughly block cadence to capture the commit-phase block (full eval rounds are far too
# slow). One finney block is ~12s.
COMMIT_POLL_INTERVAL_SECS = 12
# A commit-phase digest only needs to survive until this validator processes its reveal.
# After this many blocks with no matching reveal the commit is treated as abandoned and
# pruned, so persisted ``commit_phase_seen`` stays bounded. ~300 blocks ≈ 1h of slack.
COMMIT_PHASE_MAX_AGE_BLOCKS = 300
# Precheck re-verification cadence (issue #96, defense-in-depth). A commitment's full
# security scan + artifact hash previously only ran once, ever, on first sight -- every
# later round only re-fetched manifest.json and trusted the originally-recorded hash
# forever. Even with ``rev`` now enforced to be a pinned, immutable git SHA (see
# ``CodecCommitment.validate_revision``), periodically forcing a full re-check is a second,
# independent safety net against anything a single first-pass check might have missed. Purely
# a local caching/performance tradeoff -- not consensus-critical, so it does not need to be
# identical across validators. ~7200 blocks =~ 24h at ~12s/block.
PRECHECK_FULL_RECHECK_INTERVAL_BLOCKS = 7200

# --- Rolling-winner policy -----------------------------------------------------
# current winner / previous winner. Effective split after the temporal burn is
# 52.5% / 22.5% / 25% burned (see burn_schedule).
WINNER_WEIGHTS = (0.70, 0.30)
WINNER_LIMIT = 2
DEFAULT_WIN_MARGIN = 0.05  # epsilon: 5% relative ratio improvement required to dethrone

# --- Codec artifact limits ------------------------------------------------------
DEFAULT_MAX_ARTIFACT_BYTES = 10 * 2**30  # 10 GiB
VRAM_CAP_BYTES = 24 * 2**30  # 24 GiB
RAM_CAP_BYTES = 32 * 2**30  # 32 GiB
# Max total bytes a codec may write to the validator host's disk during either DockerRunner
# path (issue #54): the classic --network none-from-start path and the miner-published-image
# warmup/seal/benchmark lifecycle (whose warmup downloads -- pip installs, model weights --
# also land in this same scratch mount). Network-wide, not a per-operator override, same
# rationale as VRAM_CAP_BYTES/RAM_CAP_BYTES.
SCRATCH_CAP_BYTES = 117 * 2**30  # 117 GiB

MAX_CHALLENGERS_PER_ROUND = 32

# Version stamp for the scoring surfaces that determine a codec's measured ratio (issue
# #104): eval/scoring.py's aggregation formula, BASELINE_LEVEL, the corpus source list/
# sampling, and the validity gates (roundtrip check, throughput floor). Bump this whenever a
# change to any of those would change a codec's measured ratio -- ScoreState entries stamped
# with an older value are dropped on state load (core.state.load_state) rather than trusted
# forever, so every hotkey (including a reigning champion, including one-shot-excluded
# losers) gets fairly re-benchmarked under the new rules instead of being silently compared
# against numbers computed under the old ones.
# v2 (issue #112): shard-randomized corpus sampling, fineweb -> fineweb-edu (2x/1x mix),
# retired pile shard 0 (burned range), flat-average scored_ratio. Invalidates every score
# computed under the exploitable prefix-bounded sampler -- the exploiting champion included.
SCORING_VERSION = 2

# --- Per-source evaluation (issue #10; remixed by issue #112) -------------------
# Score each miner on three 4 MiB windows: two random fineweb-edu windows and one pile
# window, each stream weighted equally in the final score (flat average -- see
# eval/scoring.scored_ratio). The "name:count" syntax sets per-source scored stream counts;
# a bare name falls back to --eval-streams. enwik9 runs as a one-window benchmark display
# only: it is not scored and does not affect validity. Window starts are salt-seeded (see
# eval/streams.derive_seed), so each validator independently picks fresh random windows
# every benchmarking round.
EVAL_SOURCE = "fineweb-edu:2,pile:1"
EVAL_BENCHMARK_SOURCE = "enwik9"
EVAL_STREAMS = 2
EVAL_BENCHMARK_STREAMS = 1
EVAL_STREAM_BYTES = 4 * 2**20  # 4 MiB per stream

# --- Gates -----------------------------------------------------------------------
THROUGHPUT_FLOOR_BPS = 10 * 1024  # >= 10 KiB/s decompress throughput, per GPU
# Compress/decompress wall-clock budget, per stream (issue #73). Previously derived from the
# dead whole-corpus 32 MiB STREAM_BYTES constant (~3277s) rather than the 4 MiB streams
# actually scored (EVAL_STREAM_BYTES / THROUGHPUT_FLOOR_BPS ~= 410s) -- a flat, slightly
# rounder value above that. There is a second, independently-defined copy of this exact value
# in eval/glyph_eval_runner.py (deliberately not importing this module -- see its docstring);
# keep both in sync by hand.
COMPRESS_BUDGET_SECS = 450.0
BASELINE_LEVEL = 19  # zstd -19: the vacant-crown floor a codec must beat

# --- Temporal burn schedule ------------------------------------------------------
# 25% daily burn applied temporally: 1 unpredictable tempo per 4-tempo window
# sets weights 100% to BURN_UID.
BURN_WINDOW_TEMPOS = 4
BURN_UID = 0
# Network-wide on/off switch for the temporal burn feature. Disabled at launch (issue #43),
# re-enabled (issue #88) after a live weight-copier (mirroring the validator's exact weight
# vector without running the real evaluation) was observed on netuid 117 -- the burn-tempo
# schedule is exactly the anti-copy signal that punishes this. With this True, the burn
# position each window (H(S || window) mod 4, see burn_schedule.py) sets weights 100% to
# BURN_UID; a copier that never evaluates cannot compute S and diverges maximally on the
# tempos it gets wrong. Burn-tempo alignment feeds consensus, so -- same rationale as
# WINDOW_ANCHOR_BLOCK -- this MUST be identical on every validator and is a committed
# constant, never a per-operator override. Flip and ship the change to all validators
# together to change it either direction; the burn_schedule module itself is left intact
# either way.
BURN_ENABLED = True
# Network-wide origin so every validator's window index aligns. This MUST be identical on
# every validator: a per-operator override would desync the burn windows and split consensus,
# so it is a committed constant (not an env var). 0 anchors the windows at genesis, which is a
# valid, fully deterministic choice — window boundaries are still fixed network-wide. To
# re-phase the windows at/after registration, change this value in source and ship it to all
# validators together (the validator/weight-setter `--window-anchor` flag exists only for
# isolated testing and must not diverge between mainnet validators).
WINDOW_ANCHOR_BLOCK = 0

# --- Chutes (SN64) evaluation backend -----------------------------------------
# GPU type pinned via NodeSelector.include so all validators measure identical
# compressed bytes (same-system determinism + reference SKU). As an
# integrated SN64 subnet, Chutes now *mandates* this exact SKU for the eval
# chutes -- deploys reject anything else ("TEE with node_selector
# include=['pro_6000'] is required now for integrated subnet chutes"), so this
# must stay 'pro_6000'. The RTX PRO 6000 (Blackwell) carries 96 GB; the include
# pin already fixes the SKU, so the VRAM floor below is a consistency check.
REFERENCE_SKU = "pro_6000"
REFERENCE_MIN_VRAM_GB = 80
# The Chutes account that builds/deploys/serves the eval chutes. Deployment-specific, not
# consensus-critical: every validator targets the same deployed chutes and can override their
# URLs via GLYPH_COMPRESS_CHUTE_URL / GLYPH_DECOMPRESS_CHUTE_URL (see runner_chutes.py), so this
# only needs to match the account that ran the deploy. Set GLYPH_CHUTE_USERNAME; defaults to "glyph".
CHUTE_USERNAME = os.environ.get("GLYPH_CHUTE_USERNAME", "glyph")
CHUTE_NAME = "glyph-runner"  # shared image name
# Compress and decompress run on SEPARATE deployed chutes (separate containers), so a codec
# cannot stash the raw input during compress and read it back during decompress -- the
# decompress worker only ever sees the blob (exploit-prevention #14).
CHUTE_COMPRESSOR_NAME = "glyph-compressor"
CHUTE_DECOMPRESSOR_NAME = "glyph-decompressor"

# --- Docker runner (local production alternative to Chutes) ------------------
# Every validator using --runner docker --docker-gpu must run on the SAME GPU model, for the
# same reason Chutes pins REFERENCE_SKU: compress_secs/decompress_secs (gated against
# THROUGHPUT_FLOOR_BPS) are only comparable across validators if the hardware is identical
# (same-system determinism). Unlike REFERENCE_SKU (enforced platform-side by Chutes'
# node_selector), nothing external enforces this for Docker, so DockerRunner checks it itself
# (nvidia-smi) and fails closed rather than silently running on whatever GPU is present -- see
# runner_docker.py's _verify_gpu_model(). Network-wide, not a per-operator override (same
# rationale as WINDOW_ANCHOR_BLOCK): a validator that quietly used a faster/slower card would
# desync throughput-floor gating from the rest of the network.
DOCKER_REFERENCE_GPU = "RTX 4090"
