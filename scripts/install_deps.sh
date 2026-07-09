#!/usr/bin/env bash
# Install Glyph subnet dependencies (system + python + pm2).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

echo "[glyph] installing system packages (zstd, Node.js/npm)..."
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y && sudo apt-get install -y zstd python3-venv nodejs npm
fi

echo "[glyph] creating venv and installing package..."
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

echo "[glyph] installing pm2 (for auto-update + service management)..."
if command -v npm >/dev/null 2>&1; then
    npm install -g pm2 || echo "[glyph] pm2 install failed; install Node.js/npm first"
else
    echo "[glyph] npm not found (no apt-get on this host?); install Node.js then: npm install -g pm2"
fi

echo "[glyph] building glyph-runner-default:latest (zstandard-enabled validator runner image)..."
if command -v docker >/dev/null 2>&1; then
    docker build -f docker/glyph-runner-default.Dockerfile -t glyph-runner-default:latest . \
        || echo "[glyph] docker build failed; validators need this image (see docs/VALIDATING.md)"
else
    echo "[glyph] docker not found; miners can skip this. Validators need Docker + this image."
fi

echo "[glyph] done. Next:"
echo "  cp .env.example .env   # set CHUTES_API_KEY"
echo "  pytest -q"
