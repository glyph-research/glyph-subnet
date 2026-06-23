#!/usr/bin/env bash
# Reset validator runtime state (scores/winner history). Keeps the salt unless --all.
set -euo pipefail

STATE_DIR="${1:-./state}"
echo "[glyph] removing $STATE_DIR/validator_state.json"
rm -f "$STATE_DIR/validator_state.json"
if [[ "${2:-}" == "--all" ]]; then
    echo "[glyph] removing salt + full state dir $STATE_DIR"
    rm -rf "$STATE_DIR"
fi
echo "[glyph] done."
