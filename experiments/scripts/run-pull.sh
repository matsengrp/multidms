#!/usr/bin/env bash
# Pull pipeline results from the remote server via rsync.
#
# Usage: scripts/run-pull.sh <pipeline>
#
# Arguments:
#   pipeline   Pipeline name: simulation or spike
#
# Examples:
#   scripts/run-pull.sh simulation
#   scripts/run-pull.sh spike
set -euo pipefail

# Separate positional args from key=value overrides
POSITIONAL=()
OVERRIDES=()
for arg in "$@"; do
    if [[ "$arg" == *=* ]]; then
        OVERRIDES+=("$arg")
    else
        POSITIONAL+=("$arg")
    fi
done

if [ ${#POSITIONAL[@]} -lt 1 ]; then
    echo "Usage: $0 <pipeline> [key=value ...]"
    echo "  pipeline: simulation | spike"
    echo "  overrides: host=quokka (optional, overrides remote.yaml)"
    exit 1
fi

PIPELINE="${POSITIONAL[0]}"

# Validate pipeline name
if [[ "$PIPELINE" != "simulation" && "$PIPELINE" != "spike" ]]; then
    echo "Error: pipeline must be 'simulation' or 'spike', got '$PIPELINE'" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load remote config (with overrides)
eval "$(python3 "$SCRIPT_DIR/remote_config.py" "${OVERRIDES[@]+${OVERRIDES[@]}}")"

# Map pipeline name to directory name
case "$PIPELINE" in
    spike) PIPELINE_DIR="scv2-spike" ;;
    *)     PIPELINE_DIR="$PIPELINE" ;;
esac

LOCAL_DIR="$PROJECT_DIR/experiments/$PIPELINE_DIR/results/"
REMOTE_PATH="$host:$remote_dir/experiments/$PIPELINE_DIR/results/"

echo "==> Pulling $PIPELINE results from $host..."
mkdir -p "$LOCAL_DIR"
rsync -avz --progress "$REMOTE_PATH" "$LOCAL_DIR"

echo "==> Done. Results available at: experiments/$PIPELINE_DIR/results/"
