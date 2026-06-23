#!/usr/bin/env bash
# Build and deploy the glyph evaluation chutes to Chutes (SN64): the compressor and the
# decompressor as SEPARATE chutes (separate containers), so a codec cannot stash the raw
# input during compress and read it back during decompress (#14).
# Run from the repo root. Requires a logged-in chutes account (~/.chutes/config.ini) and,
# for validators invoking it, CHUTES_API_KEY in .env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

if ! command -v chutes >/dev/null 2>&1; then
    echo "chutes CLI not found. Install with: pip install chutes"
    exit 1
fi

echo "[glyph] building + deploying eval.chute_app:compressor_chute + decompressor_chute ..."
glyph-deploy-chute --build --deploy --public --accept-fee "$@"

echo "[glyph] deployed. Point validators at both chutes with --compress-chute-url /"
echo "        --decompress-chute-url (or GLYPH_COMPRESS_CHUTE_URL / GLYPH_DECOMPRESS_CHUTE_URL)."
