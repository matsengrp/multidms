#!/usr/bin/env bash
# Launch a pipeline on the remote server in a tmux session.
#
# Usage: scripts/remote-pipeline.sh <pipeline> <profile>
#
# Arguments:
#   pipeline   Pipeline name: simulation or spike
#   profile    Run profile: test or prod
#
# Examples:
#   scripts/remote-pipeline.sh simulation test
#   scripts/remote-pipeline.sh spike prod
set -euo pipefail

# Separate positional args from key=value overrides (e.g. host=quokka)
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

# Sync first (pass through host overrides)
"$SCRIPT_DIR/remote-sync.sh" "${OVERRIDES[@]+${OVERRIDES[@]}}"

# Load remote config (with overrides)
eval "$(python3 "$SCRIPT_DIR/remote_config.py" "${OVERRIDES[@]+${OVERRIDES[@]}}")"

# Map pipeline name to pixi task prefix
if [ "$PIPELINE" = "simulation" ]; then
    PIXI_TASK="sim-${PROFILE}"
else
    PIXI_TASK="spike-${PROFILE}"
fi

TMUX_SESSION="smk-${PIPELINE}"

echo "==> Launching $PIPELINE pipeline ($PROFILE) on $host (tmux: $TMUX_SESSION)..."

# Build the remote command
REMOTE_CMD="export PATH=\$HOME/.pixi/bin:\$PATH && cd ${remote_dir} && pixi run ${PIXI_TASK}"

# Check for existing tmux session
if ssh "$host" "tmux has-session -t ${TMUX_SESSION} 2>/dev/null"; then
    echo "Error: tmux session '${TMUX_SESSION}' already exists on $host" >&2
    echo "    Attach: ssh $host -t \"tmux attach -t ${TMUX_SESSION}\"" >&2
    exit 1
fi

# Create tmux session and send command
ssh "$host" "tmux new-session -d -s ${TMUX_SESSION} && tmux send-keys -t ${TMUX_SESSION} '${REMOTE_CMD}' Enter"

echo "==> Pipeline launched in tmux session: $TMUX_SESSION"
echo "    Attach: ssh $host -t \"tmux attach -t $TMUX_SESSION\""
echo "    Status: scripts/remote-status.sh $PIPELINE"
