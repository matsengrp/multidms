# Experiments

This directory contains reproducible analysis pipelines for the multidms manuscript. Each pipeline is an independent Snakemake workflow that executes parameterized Jupyter notebooks via papermill.

## Pipelines

| Pipeline | Directory | Description |
|----------|-----------|-------------|
| Simulation | `simulation/` | Synthetic DMS data generation, model fitting, and validation |
| SARS-CoV-2 Spike | `scv2-spike/` | Spike DMS data preparation, model fitting, and evaluation |

## Quick Start

```bash
# Run the simulation pipeline (test profile, <5 min)
pixi run sim-test

# Run the simulation pipeline (production profile)
pixi run sim-prod

# Run the spike pipeline (test profile, <10 min)
pixi run spike-test

# Run the spike pipeline (production profile)
pixi run spike-prod
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

## Interactive Dashboard

An interactive [marimo](https://marimo.io) dashboard is available for exploring `ModelCollection` results. It auto-discovers every `fit_collection.pkl` found below the directory it is launched from — pipeline outputs or any other fitted collection — and provides tabs for convergence diagnostics, global epistasis landscape, parameter correlation, replicate scatter, and sparsity analysis.

```bash
# Launch the dashboard (read-only mode)
pixi run dashboard

# Launch in edit mode (modify cells interactively)
pixi run dashboard-edit
```

The dashboard file is `experiments/dashboard.py`.

## Design Principles

- **No large files in git**: All results are gitignored and produced at runtime
- **Source/output separation**: Source notebooks in `notebooks/`, executed output in `results/`
- **Config-driven**: All parameters in YAML files, not hardcoded in notebooks
- **CPU-only**: Model fitting runs on parallel CPUs — no GPU infrastructure required
- **Branch-based workflow**: No named-run system — run experiments on branches, merge if results look good
