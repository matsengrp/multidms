# Experiments

This directory contains reproducible analysis pipelines for the multidms manuscript. Each pipeline is an independent Snakemake workflow that executes parameterized Jupyter notebooks via papermill.

## Pipelines

| Pipeline | Directory | Description |
|----------|-----------|-------------|
| Simulation | `simulation/` | Synthetic DMS data generation, model fitting, and validation |
| SARS-CoV-2 Spike | `scv2-spike/` | *(Phase 2 — not yet implemented)* |

## Quick Start

```bash
# Run the simulation pipeline (test profile, <5 min)
pixi run sim-test

# Run the simulation pipeline (production profile)
pixi run sim-prod
```

## Remote Execution

For production runs on a remote server:

1. Create `~/.config/multidms-experiments/remote.yaml`:
   ```yaml
   host: user@server
   remote_dir: /path/to/multidms
   ```

2. Use the remote execution scripts:
   ```bash
   pixi run remote-sync                      # Push branch to remote
   pixi run remote-pipeline simulation prod   # Launch in tmux
   pixi run remote-status                     # Check progress
   pixi run run-pull simulation               # Pull results back
   ```

## Directory Structure

```
experiments/
├── README.md              # This file
├── scripts/               # Shared remote execution scripts
│   ├── remote_config.py   # Read ~/.config/multidms-experiments/remote.yaml
│   ├── remote-sync.sh     # Git push + SSH pull
│   ├── remote-pipeline.sh # Launch Snakemake in tmux on remote
│   ├── remote-status.sh   # Check remote git/tmux status
│   └── run-pull.sh        # Rsync results from remote
├── simulation/            # Simulation validation pipeline
│   └── (see simulation/README.md)
└── scv2-spike/            # SARS-CoV-2 spike pipeline (Phase 2)
```

## Design Principles

- **No large files in git**: All results are gitignored and produced at runtime
- **Source/output separation**: Source notebooks in `notebooks/`, executed output in `results/`
- **Config-driven**: All parameters in YAML files, not hardcoded in notebooks
- **CPU-only**: Model fitting runs on parallel CPUs — no GPU infrastructure required
- **Branch-based workflow**: No named-run system — run experiments on branches, merge if results look good
