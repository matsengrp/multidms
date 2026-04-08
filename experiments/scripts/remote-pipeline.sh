#!/usr/bin/env bash
# Launch a pipeline on the remote server in a tmux session.
# Uses git worktrees for non-main branches and auto-generates output_dir.
#
# Usage: scripts/remote-pipeline.sh <pipeline> <profile>
#
# Arguments:
#   pipeline   Pipeline name: simulation or spike
#   profile    Run profile: test, experimental, or prod
#
# Examples:
#   scripts/remote-pipeline.sh simulation test
#   scripts/remote-pipeline.sh spike prod host=orca03
set -euo pipefail

# Separate positional args from key=value overrides (e.g. host=quokka)
POSITIONAL=()
OVERRIDES=()
for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
        continue
    elif [[ "$arg" == *=* ]]; then
        OVERRIDES+=("$arg")
    else
        POSITIONAL+=("$arg")
    fi
done

if [ ${#POSITIONAL[@]} -lt 2 ]; then
    echo "Usage: $0 <pipeline> <profile> [key=value ...]"
    echo "  pipeline: simulation | spike"
    echo "  profile:  test | experimental | prod"
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
if [[ "$PROFILE" != "test" && "$PROFILE" != "experimental" && "$PROFILE" != "prod" ]]; then
    echo "Error: profile must be 'test', 'experimental', or 'prod', got '$PROFILE'" >&2
    exit 1
fi

# Warn if working tree is dirty — uncommitted changes won't reach the remote
if [ -n "$(git status --porcelain)" ]; then
    echo "WARNING: You have uncommitted changes. The remote will not see them."
    echo "Commit first, or press Ctrl-C to abort."
    read -r -p "Continue anyway? [y/N] " reply
    [ "$reply" = "y" ] || exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Sync first (pass through host overrides)
"$SCRIPT_DIR/remote-sync.sh" "${OVERRIDES[@]+${OVERRIDES[@]}}"

# Load remote config (with overrides)
eval "$(python3 "$SCRIPT_DIR/remote_config.py" "${OVERRIDES[@]+${OVERRIDES[@]}}")"

# Branch name and sanitization
BRANCH="$(cd "$PROJECT_DIR" && git rev-parse --abbrev-ref HEAD)"
SAFE_BRANCH="${BRANCH//\//-}"

# Compute working directory and output directory
if [ "$BRANCH" = "main" ]; then
    WORK_DIR="$remote_dir"
else
    WORK_DIR="${worktree_base}/${SAFE_BRANCH}"
fi
OUTPUT_DIR="results-${PROFILE}-${SAFE_BRANCH}"
TMUX_SESSION="smk-${PIPELINE}-${SAFE_BRANCH}"

# Map pipeline name to directory containing the Snakefile
case "$PIPELINE" in
    spike) PIPELINE_DIR="scv2-spike" ;;
    *)     PIPELINE_DIR="$PIPELINE" ;;
esac
SNAKEFILE="experiments/${PIPELINE_DIR}/Snakefile"

echo "==> Launching $PIPELINE pipeline ($PROFILE) on $host"
echo "    Branch: $BRANCH (safe: $SAFE_BRANCH)"
echo "    Work dir: $WORK_DIR"
echo "    Output dir: $OUTPUT_DIR"
echo "    tmux: $TMUX_SESSION"

# Check for existing tmux session
if ssh "$host" "tmux has-session -t ${TMUX_SESSION} 2>/dev/null"; then
    echo "Error: tmux session '${TMUX_SESSION}' already exists on $host" >&2
    echo "    Attach: ssh $host -t \"tmux attach -t ${TMUX_SESSION}\"" >&2
    exit 1
fi

# Build profile config arg (empty for prod)
if [ "$PROFILE" = "test" ]; then
    CONFIG_ARGS="profile=test output_dir=${OUTPUT_DIR}"
else
    CONFIG_ARGS="output_dir=${OUTPUT_DIR}"
fi

# Build the remote command: pixi install, then snakemake with output_dir override
REMOTE_CMD="export PATH=\$HOME/.pixi/bin:\$PATH && cd ${WORK_DIR} && pixi install && pixi run snakemake -s ${SNAKEFILE} --config ${CONFIG_ARGS} -j4"

# Create tmux session and send command
ssh "$host" "tmux new-session -d -s ${TMUX_SESSION} && tmux send-keys -t ${TMUX_SESSION} '${REMOTE_CMD}' Enter"

echo "==> Pipeline launched in tmux session: $TMUX_SESSION"
echo "    Attach: ssh $host -t \"tmux attach -t $TMUX_SESSION\""
echo "    Status: pixi run remote-status -- $PIPELINE $PROFILE host=$(echo $host | cut -d@ -f2 2>/dev/null || echo $host)"
