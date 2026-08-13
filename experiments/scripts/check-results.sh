#!/usr/bin/env bash
# Verify each pipeline's `results/` symlink resolves before the docs build.
#
# Every `docs/*.nblink` file resolves through
# `experiments/<pipeline>/results/<notebook>.ipynb`, but `results/` is
# gitignored and therefore absent from a fresh clone. Without it Sphinx
# fails with a bare "No such file or directory" that gives no hint the real
# cause is a missing symlink.
#
# This script only reports. It never creates, repairs, or repoints a
# symlink: which run is canonical is a judgement call about which numbers
# the docs should publish, and guessing it silently would be worse than
# failing.
#
# Usage:
#   check-results.sh            # check every pipeline
#   check-results.sh spike      # check one
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

pipeline_dir() {
    case "$1" in
        spike) echo "scv2-spike" ;;
        *)     echo "$1" ;;
    esac
}

pipeline_task() {
    case "$1" in
        simulation) echo "sim-prod" ;;
        *)          echo "$1-prod" ;;
    esac
}

if [ "$#" -gt 0 ]; then
    PIPELINES=("$@")
else
    PIPELINES=(simulation spike)
fi

status=0

for pipeline in "${PIPELINES[@]}"; do
    dir="$EXPERIMENTS_DIR/$(pipeline_dir "$pipeline")"
    link="$dir/results"
    rel="experiments/$(pipeline_dir "$pipeline")/results"

    if [ ! -d "$dir" ]; then
        echo "✗ $pipeline: no such pipeline directory: $dir" >&2
        status=1
        continue
    fi

    # Resolves as a directory: symlink or real directory, either is fine.
    if [ -d "$link/" ]; then
        if [ -L "$link" ]; then
            echo "✓ $pipeline: $rel -> $(readlink "$link")"
        else
            echo "✓ $pipeline: $rel (directory)"
        fi
        continue
    fi

    status=1

    if [ -L "$link" ]; then
        echo "✗ $pipeline: $rel is a broken symlink -> $(readlink "$link")" >&2
        echo "    Its target no longer exists. A symlink pointing into" >&2
        echo "    .worktrees/ breaks this way when that branch lands and the" >&2
        echo "    worktree is removed." >&2
    else
        echo "✗ $pipeline: $rel does not exist" >&2
        echo "    It is gitignored, so a fresh clone never has it." >&2
    fi

    # List what is actually on disk so the fix is a copy-paste away.
    # Built with a portable while-read loop: macOS ships bash 3.2, which has
    # no `mapfile`.
    candidates=()
    while IFS= read -r c; do
        [ -n "$c" ] && candidates+=("$c")
    done < <(cd "$dir" && find . -maxdepth 1 -type d -name 'results-*' \
        2>/dev/null | sed 's|^\./||' | sort)

    echo "" >&2
    if [ "${#candidates[@]}" -gt 0 ]; then
        echo "    Available runs in experiments/$(pipeline_dir "$pipeline")/:" >&2
        for c in "${candidates[@]}"; do
            echo "      $c" >&2
        done
        echo "" >&2
        echo "    Pick the one the docs should publish and link it:" >&2
        echo "      ln -sfn <run-dir> $rel" >&2
    else
        echo "    No run directories found. Produce one:" >&2
        echo "      pixi run $(pipeline_task "$pipeline")" >&2
        echo "    or fetch an existing one from the remote host:" >&2
        echo "      pixi run run-pull -- $pipeline prod host=<server>" >&2
        echo "    then link it:" >&2
        echo "      ln -sfn results-prod-<branch> $rel" >&2
    fi
    echo "" >&2
done

exit $status
