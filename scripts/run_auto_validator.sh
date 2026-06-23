#!/usr/bin/env bash

# Run the Glyph validator under PM2 and keep it updated from GitHub.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

DEFAULT_CHECK_INTERVAL=1200
DEFAULT_LOG_FILE="./logs/validator_auto_update.log"
DEFAULT_BACKUP_DIR="./backups"
DEFAULT_GITHUB_REPO="${GITHUB_REPO:-glyph-research/glyph-subnet}"
DEFAULT_VALIDATOR_PROC_NAME="glyph_auto_validator"
DEFAULT_MONITOR_PROC_NAME="glyph_update_monitor"
VERSION_FILE="src/core/__init__.py"
VERSION_VAR="__version_key__"
VERSION_LABEL="version key"

CHECK_INTERVAL="${CHECK_INTERVAL:-$DEFAULT_CHECK_INTERVAL}"
LOG_FILE="$DEFAULT_LOG_FILE"
BACKUP_DIR="$DEFAULT_BACKUP_DIR"
GITHUB_REPO="$DEFAULT_GITHUB_REPO"
GIT_BRANCH="${GIT_BRANCH:-}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/venv/bin/python}"
VALIDATOR_PROC_NAME="$DEFAULT_VALIDATOR_PROC_NAME"
MONITOR_PROC_NAME="$DEFAULT_MONITOR_PROC_NAME"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3 || true)"
fi

log() {
    local level="$1"
    shift
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    if [[ "$LOG_FILE" != "/dev/null" ]]; then
        mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
    fi
    echo "[$timestamp] [$level] $*" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }

show_help() {
    cat << EOF
Usage: $0 [AUTO_UPDATE_OPTIONS] [VALIDATOR_ARGS...]

Starts glyph-validator under PM2 and a PM2 monitor process that periodically checks
GitHub for a newer src/core/__init__.py __version_key__, pulls updates, reinstalls the
package, and restarts the validator.

Auto-update options:
  --check-interval SECONDS      Update check interval, minimum 60 seconds.
  --log-file PATH               Log file path. Default: $DEFAULT_LOG_FILE
  --backup-dir PATH             Backup directory. Default: $DEFAULT_BACKUP_DIR
  --github-repo OWNER/REPO      GitHub repo to check. Default: $DEFAULT_GITHUB_REPO
  --branch BRANCH               Git branch to pull/check. Default: current branch.
  --python PATH                 Python interpreter. Default: ./venv/bin/python
  --validator-proc-name NAME    PM2 validator process name.
  --monitor-proc-name NAME      PM2 monitor process name.
  --help, -h                    Show this help message.

Example:
  $0 --check-interval 1200 --network finney --netuid 117 \\
    --wallet-name validator_wallet --hotkey-name default --state-dir ./state
EOF
}

read_version_from_file() {
    local file_path="$1"
    "$PYTHON_BIN" - "$file_path" "$VERSION_VAR" <<'PY'
import ast
import sys
from pathlib import Path

path = Path(sys.argv[1])
name = sys.argv[2]
tree = ast.parse(path.read_text())
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                print(ast.literal_eval(node.value))
                raise SystemExit(0)
raise SystemExit(f"{name} not found in {path}")
PY
}

read_local_version() {
    read_version_from_file "$REPO_DIR/$VERSION_FILE"
}

read_remote_version() {
    local branch="$1"
    git fetch --quiet origin "$branch"
    git show "origin/$branch:$VERSION_FILE" | "$PYTHON_BIN" -c '
import ast
import sys

name = sys.argv[1]
branch = sys.argv[2]
version_file = sys.argv[3]
content = sys.stdin.read()
tree = ast.parse(content)
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                print(ast.literal_eval(node.value))
                raise SystemExit(0)
raise SystemExit(f"{name} not found in remote origin/{branch}:{version_file}")
' "$VERSION_VAR" "$branch" "$VERSION_FILE"
}

version_less_than() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import re
import sys

def parts(version):
    return [int(part) for part in re.findall(r"\d+", version)]

raise SystemExit(0 if parts(sys.argv[1]) < parts(sys.argv[2]) else 1)
PY
}

current_branch() {
    if [[ -n "$GIT_BRANCH" ]]; then
        echo "$GIT_BRANCH"
    elif git rev-parse --git-dir >/dev/null 2>&1; then
        git branch --show-current 2>/dev/null || echo "main"
    else
        echo "main"
    fi
}

create_backup() {
    local version="$1"
    local backup_path="$BACKUP_DIR/backup_$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "$backup_path"
    git rev-parse HEAD > "$backup_path/commit_hash.txt"
    echo "$version" > "$backup_path/version.txt"
    log_info "Backup created at $backup_path" >&2
    echo "$backup_path"
}

rollback_from_backup() {
    local backup_path="$1"
    if [[ ! -f "$backup_path/commit_hash.txt" ]]; then
        log_error "Backup is missing commit_hash.txt: $backup_path"
        return 1
    fi
    local commit_hash
    commit_hash="$(cat "$backup_path/commit_hash.txt")"
    log_warn "Rolling back to $commit_hash"
    git reset --hard "$commit_hash"
    "$PYTHON_BIN" -m pip install -e .
    pm2 restart "$VALIDATOR_PROC_NAME"
}

cleanup_old_backups() {
    mkdir -p "$BACKUP_DIR"
    find "$BACKUP_DIR" -maxdepth 1 -type d -name 'backup_*' -print | sort -r | tail -n +6 | xargs rm -rf 2>/dev/null || true
}

validate_config() {
    if ! [[ "$CHECK_INTERVAL" =~ ^[0-9]+$ ]] || [[ "$CHECK_INTERVAL" -lt 60 ]]; then
        log_error "--check-interval must be an integer >= 60"
        exit 1
    fi
    if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
        log_error "Python interpreter not found. Pass --python PATH or set PYTHON_BIN."
        exit 1
    fi
    if ! command -v pm2 >/dev/null 2>&1; then
        log_error "pm2 not found. Install with: npm install -g pm2"
        exit 1
    fi
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || LOG_FILE="/dev/null"
}

monitor_loop() {
    cd "$REPO_DIR"
    local branch
    branch="$(current_branch)"
    local current_version
    current_version="$(read_local_version)"
    log_info "Starting Glyph auto-update monitor"
    log_info "Repo: $GITHUB_REPO"
    log_info "Branch: $branch"
    log_info "Current $VERSION_LABEL: $current_version"
    log_info "Check interval: $CHECK_INTERVAL seconds"

    while true; do
        log_info "Checking for validator updates..."
        if ! git rev-parse --git-dir >/dev/null 2>&1; then
            log_error "Not running inside a git repository"
            sleep "$CHECK_INTERVAL"
            continue
        fi

        local latest_version
        if ! latest_version="$(read_remote_version "$branch" 2>>"$LOG_FILE")"; then
            log_warn "Could not read remote version"
            sleep "$CHECK_INTERVAL"
            continue
        fi

        current_version="$(read_local_version)"
        log_info "Local $VERSION_LABEL: $current_version; remote $VERSION_LABEL: $latest_version"

        if version_less_than "$current_version" "$latest_version"; then
            log_info "New $VERSION_LABEL available: $latest_version"
            local backup_path
            backup_path="$(create_backup "$current_version")"

            if git pull --ff-only origin "$branch"; then
                log_info "Pulled latest code; reinstalling package"
                if "$PYTHON_BIN" -m pip install -e . && pm2 restart "$VALIDATOR_PROC_NAME"; then
                    sleep 5
                    if pm2 describe "$VALIDATOR_PROC_NAME" | grep -q "online"; then
                        log_info "Validator updated and restarted successfully"
                        cleanup_old_backups
                    else
                        log_error "Validator is not online after restart; rolling back"
                        rollback_from_backup "$backup_path"
                    fi
                else
                    log_error "Install or restart failed; rolling back"
                    rollback_from_backup "$backup_path"
                fi
            else
                log_error "git pull failed. Commit/stash local changes before using auto-update."
                rm -rf "$backup_path"
            fi
        else
            log_info "No update needed"
        fi

        sleep "$CHECK_INTERVAL"
    done
}

json_args() {
    "$PYTHON_BIN" - "$@" <<'PY'
import json
import sys

args = sys.argv[1:]
if args:
    print(", " + ", ".join(json.dumps(arg) for arg in args))
else:
    print("")
PY
}

create_pm2_config() {
    local args_json="$1"
    local config_path="$REPO_DIR/app.config.js"
    cat > "$config_path" << EOF
module.exports = {
  apps: [{
    name: "$VALIDATOR_PROC_NAME",
    namespace: "glyph-subnet",
    script: "$PYTHON_BIN",
    args: ["-m", "validator"$args_json],
    cwd: "$REPO_DIR",
    env: { PYTHONPATH: "src" },
    min_uptime: "5m",
    max_restarts: 5
  }]
}
EOF
    echo "$config_path"
}

main() {
    cd "$REPO_DIR"
    local validator_args=()
    local internal_monitor=false
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --check-interval)
                CHECK_INTERVAL="$2"
                shift 2
                ;;
            --log-file)
                LOG_FILE="$2"
                shift 2
                ;;
            --backup-dir)
                BACKUP_DIR="$2"
                shift 2
                ;;
            --github-repo)
                GITHUB_REPO="$2"
                shift 2
                ;;
            --branch)
                GIT_BRANCH="$2"
                shift 2
                ;;
            --python)
                PYTHON_BIN="$2"
                shift 2
                ;;
            --validator-proc-name)
                VALIDATOR_PROC_NAME="$2"
                shift 2
                ;;
            --monitor-proc-name)
                MONITOR_PROC_NAME="$2"
                shift 2
                ;;
            --internal-monitor)
                internal_monitor=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                validator_args+=("$1")
                shift
                ;;
        esac
    done

    validate_config
    if [[ "$internal_monitor" == "true" ]]; then
        monitor_loop
        exit 0
    fi

    git config core.filemode false 2>/dev/null || true

    local branch
    branch="$(current_branch)"
    log_info "Starting Glyph validator with auto-update"
    log_info "Watching $GITHUB_REPO on branch $branch"

    if pm2 status | grep -q "$VALIDATOR_PROC_NAME"; then
        pm2 delete "$VALIDATOR_PROC_NAME"
    fi
    if pm2 status | grep -q "$MONITOR_PROC_NAME"; then
        pm2 delete "$MONITOR_PROC_NAME"
    fi

    local config_path
    config_path="$(create_pm2_config "$(json_args "${validator_args[@]}")")"
    pm2 start "$config_path"

    pm2 start "$0" \
        --name "$MONITOR_PROC_NAME" \
        --namespace "glyph-subnet" \
        --log "$LOG_FILE" \
        -- --internal-monitor \
        --check-interval "$CHECK_INTERVAL" \
        --log-file "$LOG_FILE" \
        --backup-dir "$BACKUP_DIR" \
        --github-repo "$GITHUB_REPO" \
        --branch "$branch" \
        --python "$PYTHON_BIN" \
        --validator-proc-name "$VALIDATOR_PROC_NAME" \
        --monitor-proc-name "$MONITOR_PROC_NAME"

    log_info "Auto-validator started"
    log_info "Status: pm2 status"
    log_info "Logs: pm2 logs $VALIDATOR_PROC_NAME or pm2 logs $MONITOR_PROC_NAME"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
