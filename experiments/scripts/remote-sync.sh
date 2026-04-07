#!/usr/bin/env bash
# Sync local code to remote server via git push + SSH pull.
# For main: updates the main clone in-place.
# For other branches: creates or updates a git worktree on the remote.
#
# Usage: scripts/remote-sync.sh [key=value overrides for remote_config.py]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load remote config (pass through any override args like host=orca03)
eval "$(python3 "$SCRIPT_DIR/remote_config.py" "$@")"

# Use the current local branch, not a config value
BRANCH="$(cd "$PROJECT_DIR" && git rev-parse --abbrev-ref HEAD)"
SAFE_BRANCH="${BRANCH//\//-}"

echo "==> Pushing local changes to origin/$BRANCH..."
cd "$PROJECT_DIR"
git push origin "$BRANCH"

if [ "$BRANCH" = "main" ]; then
    # Main branch: update the main clone in-place (existing behavior)
    echo "==> Pulling on remote ($host:$remote_dir)..."
    if ! ssh -o ConnectTimeout=10 "$host" "cd $remote_dir && git fetch origin && git checkout main && git pull origin main"; then
        echo "Error: SSH connection to $host failed." >&2
        echo "Check that the host is reachable and remote_dir exists." >&2
        exit 1
    fi
else
    # Non-main: create or update a worktree
    WORKTREE_DIR="${worktree_base}/${SAFE_BRANCH}"
    echo "==> Syncing worktree on remote ($host:$WORKTREE_DIR)..."
    if ! ssh -o ConnectTimeout=10 "$host" bash -s <<SYNC_EOF
        set -euo pipefail
        cd "$remote_dir"
        git fetch origin
        if [ -d "$WORKTREE_DIR" ]; then
            echo "Updating existing worktree at $WORKTREE_DIR"
            cd "$WORKTREE_DIR"
            git checkout "$BRANCH"
            git pull origin "$BRANCH"
        else
            echo "Creating new worktree at $WORKTREE_DIR"
            mkdir -p "${worktree_base}"
            git worktree add "$WORKTREE_DIR" "$BRANCH"
        fi
SYNC_EOF
    then
        echo "Error: Failed to sync worktree on $host." >&2
        exit 1
    fi
fi

echo "==> Sync complete."
