"""Shared constants for the Glyph lossless-compression subnet.

Values follow glyph/DESIGN.md (v0.1). Two classes of value live here:

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
# ``g1r|repo|rev|salt``. The earliest-commit tie-break (DESIGN §3.5) keys off the commit-phase
# block, so a mempool watcher who only learns repo|rev at reveal time can never commit earlier.
COMMIT_PHASE_PREFIX = "g1c|"
REVEAL_PHASE_PREFIX = "g1r|"
# Commit-reveal polling/pruning (exploit vector #9 follow-ups, issue #21).
# A reveal lands ~1 block after its commit, so validators must observe commitments at
# roughly block cadence to capture the commit-phase block (full eval rounds are far too
# slow). One finney block is ~12s.
COMMIT_POLL_INTERVAL_SECS = 12
# A commit-phase digest only needs to survive until this validator processes its reveal.
# After this many blocks with no matching reveal the commit is treated as abandoned and
# pruned, so persisted ``commit_phase_seen`` stays bounded. ~300 blocks ≈ 1h of slack.
COMMIT_PHASE_MAX_AGE_BLOCKS = 300

# --- Rolling-winner policy (DESIGN §3.5) --------------------------------------
# current winner / previous winner. Effective split after the temporal burn is
# 52.5% / 22.5% / 25% burned (see burn_schedule).
WINNER_WEIGHTS = (0.70, 0.30)
WINNER_LIMIT = 2
DEFAULT_WIN_MARGIN = 0.05  # epsilon: 5% relative ratio improvement required to dethrone

# --- Codec artifact limits (DESIGN §4, §7) ------------------------------------
DEFAULT_MAX_ARTIFACT_BYTES = 10 * 2**30  # 10 GiB
VRAM_CAP_BYTES = 24 * 2**30  # 24 GiB
RAM_CAP_BYTES = 32 * 2**30  # 32 GiB

# --- Evaluation streams (DESIGN §7): 8 x 32 MiB = 256 MiB paired sample -------
STREAM_BYTES = 32 * 2**20
STREAMS_PER_ROUND = 8
SAMPLE_BYTES = STREAM_BYTES * STREAMS_PER_ROUND
MAX_CHALLENGERS_PER_ROUND = 32

# --- Gates (DESIGN §3.3, §7) --------------------------------------------------
THROUGHPUT_FLOOR_BPS = 10 * 1024  # >= 10 KiB/s decompress throughput, per GPU
# Compress wall-clock budget per stream, symmetric with the decompress floor.
COMPRESS_BUDGET_SECS = STREAM_BYTES / THROUGHPUT_FLOOR_BPS
BASELINE_LEVEL = 19  # zstd -19: the vacant-crown floor a codec must beat

# --- Temporal burn schedule (DESIGN §6.1) -------------------------------------
# 25% daily burn applied temporally: 1 unpredictable tempo per 4-tempo window
# sets weights 100% to BURN_UID.
BURN_WINDOW_TEMPOS = 4
BURN_UID = 0
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
# compressed bytes (DESIGN §4 same-system determinism + reference SKU).
REFERENCE_SKU = "a100"
REFERENCE_MIN_VRAM_GB = 24
# The Chutes account that builds/deploys/serves the eval chutes. Deployment-specific, not
# consensus-critical: every validator targets the same deployed chutes and can override their
# URLs via GLYPH_COMPRESS_CHUTE_URL / GLYPH_DECOMPRESS_CHUTE_URL (see runner_chutes.py), so this
# only needs to match the account that ran the deploy. Set GLYPH_CHUTE_USERNAME; defaults to "glyph".
CHUTE_USERNAME = os.environ.get("GLYPH_CHUTE_USERNAME", "glyph")
CHUTE_NAME = "glyph-runner"  # shared image name
# Compress and decompress run on SEPARATE deployed chutes (separate containers), so a codec
# cannot stash the raw input during compress and read it back during decompress -- the
# decompress worker only ever sees the blob (DESIGN §6; exploit-prevention #14).
CHUTE_COMPRESSOR_NAME = "glyph-compressor"
CHUTE_DECOMPRESSOR_NAME = "glyph-decompressor"
