#!/usr/bin/env bash
# Pull pipeline results from the remote server via rsync.
# Worktree-aware: pulls from the correct remote directory for the current branch.
#
# Usage: scripts/run-pull.sh <pipeline> <profile> [key=value ...]
#
# Arguments:
#   pipeline   Pipeline name: simulation or spike
#   profile    Run profile: test or prod
#
# Examples:
#   scripts/run-pull.sh simulation prod
#   scripts/run-pull.sh spike test host=orca03
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

if [ ${#POSITIONAL[@]} -lt 2 ]; then
    echo "Usage: $0 <pipeline> <profile> [key=value ...]"
    echo "  pipeline: simulation | spike"
    echo "  profile:  test | prod"
    echo "  overrides: host=quokka (optional, overrides remote.yaml)"
    exit 1
fi

PIPELINE="${POSITIONAL[0]}"
PROFILE="${POSITIONAL[1]}"

# Validate pipeline name
if [[ "$PIPELINE" != "simulation" && "$PIPELINE" != "spike" ]]; then
    echo "Error: pipeline must be 'simulation' or 'spike', got '$PIPELINE'" >&2
    exit 1
fi

# Validate profile name
if [[ "$PROFILE" != "test" && "$PROFILE" != "prod" ]]; then
    echo "Error: profile must be 'test' or 'prod', got '$PROFILE'" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load remote config (with overrides)
eval "$(python3 "$SCRIPT_DIR/remote_config.py" "${OVERRIDES[@]+${OVERRIDES[@]}}")"

# Branch detection and sanitization
BRANCH="$(cd "$PROJECT_DIR" && git rev-parse --abbrev-ref HEAD)"
SAFE_BRANCH="${BRANCH//\//-}"

# Determine remote working directory
if [ "$BRANCH" = "main" ]; then
    WORK_DIR="$remote_dir"
else
    WORK_DIR="${worktree_base}/${SAFE_BRANCH}"
fi

OUTPUT_DIR="results-${PROFILE}-${SAFE_BRANCH}"

# Map pipeline name to directory name
case "$PIPELINE" in
    spike) PIPELINE_DIR="scv2-spike" ;;
    *)     PIPELINE_DIR="$PIPELINE" ;;
esac

LOCAL_DIR="$PROJECT_DIR/experiments/$PIPELINE_DIR/$OUTPUT_DIR/"
REMOTE_PATH="$host:$WORK_DIR/experiments/$PIPELINE_DIR/$OUTPUT_DIR/"

echo "==> Pulling $PIPELINE results from $host..."
echo "    Remote: $REMOTE_PATH"
echo "    Local:  $LOCAL_DIR"
mkdir -p "$LOCAL_DIR"
rsync -avz --progress "$REMOTE_PATH" "$LOCAL_DIR"

echo "==> Done. Results available at: experiments/$PIPELINE_DIR/$OUTPUT_DIR/"
