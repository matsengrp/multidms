#!/usr/bin/env python3
"""Read remote configuration from ~/.config/multidms-experiments/remote.yaml.

Uses a simple key:value parser (no PyYAML dependency) so this works with
the system Python outside of pixi.

CLI overrides (passed as key=value arguments) take precedence over the
config file.  Example::

    python3 remote_config.py host=orca03
"""

import os
import sys

CONFIG_PATH = os.path.expanduser("~/.config/multidms-experiments/remote.yaml")

REQUIRED_KEYS = ["host", "remote_dir"]


def load_remote_config(overrides=None):
    """Load and validate remote configuration.

    Expected format (simple YAML subset)::

        host: user@remote-server
        remote_dir: /home/user/projects/multidms

    Parameters
    ----------
    overrides : dict, optional
        Key/value pairs that override the config file.
    """
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Remote config not found at {CONFIG_PATH}", file=sys.stderr)
        print(
            "\nCreate it with:\n"
            f"  mkdir -p {os.path.dirname(CONFIG_PATH)}\n"
            f"  cat > {CONFIG_PATH} << 'EOF'\n"
            "  host: user@remote-server\n"
            "  remote_dir: /path/to/multidms\n"
            "  EOF",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = {}
    with open(CONFIG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Handle inline comments
            if " #" in line:
                line = line[: line.index(" #")]
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            cfg[key.strip()] = value.strip()

    # Apply CLI overrides
    if overrides:
        cfg.update(overrides)

    for key in REQUIRED_KEYS:
        if key not in cfg:
            print(
                f"Error: Missing required key '{key}' in {CONFIG_PATH}",
                file=sys.stderr,
            )
            sys.exit(1)

    return cfg


if __name__ == "__main__":
    # Parse CLI overrides: key=value arguments
    cli_overrides = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, _, v = arg.partition("=")
            cli_overrides[k] = v

    cfg = load_remote_config(cli_overrides)
    # Output as shell-friendly key=value pairs (quoted to prevent injection)
    import shlex

    for key, value in cfg.items():
        print(f"{key}={shlex.quote(value)}")
