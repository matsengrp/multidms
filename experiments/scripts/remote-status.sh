#!/usr/bin/env bash
# Check remote git status and tmux sessions.
# Worktree-aware: checks the correct directory for the current branch.
#
# Usage: scripts/remote-status.sh [pipeline] [profile] [key=value ...]
#
# Arguments:
#   pipeline   Optional: simulation or spike. Shows directory listing if given.
#   profile    Optional: test or prod. If given, shows the specific output_dir.
#   overrides  Optional: host=quokka (overrides remote.yaml)
set -euo pipefail

# Separate positional args from key=value overrides
PIPELINE=""
PROFILE=""
OVERRIDES=()
for arg in "$@"; do
    if [[ "$arg" == *=* ]]; then
        OVERRIDES+=("$arg")
    elif [ -z "$PIPELINE" ]; then
        PIPELINE="$arg"
    elif [ -z "$PROFILE" ]; then
        PROFILE="$arg"
    fi
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load remote config (with overrides)
eval "$(python3 "$SCRIPT_DIR/remote_config.py" "${OVERRIDES[@]+${OVERRIDES[@]}}")"

# Branch detection and sanitization
BRANCH="$(cd "$PROJECT_DIR" && git rev-parse --abbrev-ref HEAD)"
SAFE_BRANCH="${BRANCH//\//-}"

# Determine working directory
if [ "$BRANCH" = "main" ]; then
    WORK_DIR="$remote_dir"
else
    WORK_DIR="${worktree_base}/${SAFE_BRANCH}"
fi

echo "==> Remote git status ($host:$WORK_DIR):"
ssh "$host" "cd \"$WORK_DIR\" && git log --oneline -3 && echo && git status -s" 2>/dev/null || echo "    (worktree not found)"

echo ""

echo "==> Active tmux sessions (smk-*${SAFE_BRANCH}*):"
ssh "$host" "tmux list-sessions 2>/dev/null | grep 'smk-.*${SAFE_BRANCH}' || echo '    (none)'"

if [ -n "$PIPELINE" ]; then
    # Map pipeline name to directory name
    case "$PIPELINE" in
        spike) PIPELINE_DIR="scv2-spike" ;;
        *)     PIPELINE_DIR="$PIPELINE" ;;
    esac

    if [ -n "$PROFILE" ]; then
        # Show the specific output_dir for this profile+branch
        OUTPUT_DIR="results-${PROFILE}-${SAFE_BRANCH}"
        echo ""
        echo "==> Results directory ($OUTPUT_DIR):"
        ssh "$host" "ls -la \"$WORK_DIR/experiments/$PIPELINE_DIR/$OUTPUT_DIR/\" 2>/dev/null || echo '    (not yet created)'"
    else
        # Show all results-* directories
        echo ""
        echo "==> Results directories for $PIPELINE:"
        ssh "$host" "ls -d \"$WORK_DIR/experiments/$PIPELINE_DIR/results\"*/ 2>/dev/null || echo '    (none)'"
    fi
fi
