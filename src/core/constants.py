"""Shared constants for the Glyph lossless-compression subnet.

Values follow glyph/DESIGN.md (v0.1). Anything marked TODO must be fixed at subnet
registration so it is identical network-wide.
"""

# --- Chain / commitment -------------------------------------------------------
DEFAULT_NETUID = 117  # always overridable via --netuid
COMMITMENT_VERSION = 1
COMMITMENT_PREFIX = "glyph:"
COMPACT_COMMITMENT_PREFIX = "g1|"

# --- Rolling-winner policy (DESIGN §3.5) --------------------------------------
# current winner / previous winner. Effective split after the temporal burn is
# 52.5% / 22.5% / 25% burned (see burn_schedule).
WINNER_WEIGHTS = (0.70, 0.30)
WINNER_LIMIT = 2
DEFAULT_WIN_MARGIN = 0.005  # epsilon: relative ratio improvement required to dethrone

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
# Network-wide origin so every validator's window index aligns. TODO: set to the
# subnet registration block at launch.
WINDOW_ANCHOR_BLOCK = 0

# --- Chutes (SN64) evaluation backend -----------------------------------------
# GPU type pinned via NodeSelector.include so all validators measure identical
# compressed bytes (DESIGN §4 same-system determinism + reference SKU).
REFERENCE_SKU = "a100"
REFERENCE_MIN_VRAM_GB = 24
CHUTE_USERNAME = "glyph"  # TODO: deploying account username
CHUTE_NAME = "glyph-runner"
