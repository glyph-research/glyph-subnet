#!/usr/bin/env bash

set -euo pipefail

echo "Setting up Glyph repository hooks and script permissions..."

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "This must be run from inside the git repository."
    exit 1
fi

git config core.filemode false

mkdir -p .git/hooks
cat > .git/hooks/post-merge <<'EOF'
#!/usr/bin/env bash
chmod +x scripts/*.sh 2>/dev/null || true
echo "Glyph script permissions restored"
EOF

chmod +x .git/hooks/post-merge
chmod +x scripts/*.sh 2>/dev/null || true

echo "Setup complete."
