#!/usr/bin/env bash
# Check remote git status and tmux sessions.
#
# Usage: scripts/remote-status.sh [pipeline] [key=value ...]
#
# Arguments:
#   pipeline   Optional: simulation or spike. Shows directory listing if given.
#   overrides  Optional: host=quokka (overrides remote.yaml)
set -euo pipefail

# Separate positional args from key=value overrides
PIPELINE=""
OVERRIDES=()
for arg in "$@"; do
    if [[ "$arg" == *=* ]]; then
        OVERRIDES+=("$arg")
    elif [ -z "$PIPELINE" ]; then
        PIPELINE="$arg"
    fi
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load remote config (with overrides)
eval "$(python3 "$SCRIPT_DIR/remote_config.py" "${OVERRIDES[@]+${OVERRIDES[@]}}")"

echo "==> Remote git status ($host:$remote_dir):"
ssh "$host" "cd \"$remote_dir\" && git log --oneline -3 && echo && git status -s"

echo ""

echo "==> Active tmux sessions (smk-*):"
ssh "$host" "tmux list-sessions 2>/dev/null | grep '^smk-' || echo '    (none)'"

if [ -n "$PIPELINE" ]; then
    # Map pipeline name to directory name
    case "$PIPELINE" in
        spike) PIPELINE_DIR="scv2-spike" ;;
        *)     PIPELINE_DIR="$PIPELINE" ;;
    esac
    echo ""
    echo "==> Results directory for $PIPELINE:"
    ssh "$host" "ls -la \"$remote_dir/experiments/$PIPELINE_DIR/results/\" 2>/dev/null || echo '    (not yet created)'"
fi
