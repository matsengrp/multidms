# Handoff

Practical guide for whoever maintains `multidms` next. It covers the four
things that are hard to reconstruct from the code alone: how to get results
off the remote server and look at them, how to build and deploy the docs,
how the repository is laid out, and what state the manuscript work is in.

Everything below was verified against the repository on **2026-08-13**.

Reference material lives elsewhere and is not repeated here:

| For | Read |
|---|---|
| Package architecture, code style, dev commands | [`CLAUDE.md`](CLAUDE.md) |
| Contribution workflow | [`CONTRIBUTING.rst`](CONTRIBUTING.rst) |
| Pipeline internals, config tiers, run log | [`experiments/scv2-spike/README.md`](experiments/scv2-spike/README.md) |
| Pipeline index and remote setup | [`experiments/README.md`](experiments/README.md) |

---

## 1. Prerequisites

The project declares its environment tool: **use `pixi`**, never bare
`python`/`pip`.

```bash
pixi install          # one-command environment setup
pixi run test         # pytest + doctests
pixi run fmt-check    # black
```

> ⚠️ `pixi run lint` is **vacuous on `experiments/`** — `ruff` is configured
> to exclude that tree. To lint experiment code, pass paths explicitly:
> `pixi run ruff check experiments/scv2-spike/notebooks/_downstream.py`

### Remote access

Long fits run on the Matsen lab's shared servers, configured **outside the
repository** in `~/.config/multidms-experiments/remote.yaml`:

```yaml
host: ermine
remote_dir: /fh/fast/matsen_e/shared/multidms/multidms
```

`worktree_base` defaults to `<parent of remote_dir>/multidms-worktrees`.
Any key can be overridden per-invocation with `host=orca03`.

> Prefer hosts `orca01`–`orca05` (64 cores, 1.5 TB RAM). **Avoid `quokka`.**
> Check load before launching anything.

---

## 2. Fetching results and viewing them

### The one-paragraph version

A production run writes to `experiments/<pipeline>/results-<profile>-<branch>/`
on the **remote** host. `run-pull` rsyncs that directory back. The `results/`
symlink then points at whichever run is canonical, and both the docs build and
the dashboard read through it.

### Step by step

```bash
# 1. Launch (from the branch whose name sets the output directory)
pixi run remote-pipeline -- spike prod host=orca03

# 2. Monitor — poll sparsely, every 30s+, never in a tight loop
pixi run remote-status -- spike prod host=orca03

# 3. Fetch when finished
pixi run run-pull -- spike prod host=orca03

# 4. Point `results/` at what you just pulled
ln -sfn results-prod-<branch> experiments/scv2-spike/results
```

`pipeline` is `simulation` or `spike`; `profile` is `test`, `experimental`,
or `prod`.

**The output directory is derived from the current branch name**, not chosen
by you: branch `292-spike-fit-tuning` + profile `prod` →
`results-prod-292-spike-fit-tuning`. The remote tmux session is named
`smk-<pipeline>-<branch>`. Attach with:

```bash
ssh orca03 -t "tmux attach -t smk-spike-<branch>"
```

> Never leave tmux sessions running after `run-pull`. Clean up the remote
> worktree and session when a run is done.

### The `results/` symlink

`experiments/<pipeline>/results` is a **gitignored symlink** to a
`results-*` directory — it does not exist in a fresh clone. Create it by
hand, pointing at whichever run the docs should publish:

```bash
ln -sfn results-prod-292-spike-fit-tuning experiments/scv2-spike/results
ln -sfn results-prod-sim-vpl500-tol1e5    experiments/simulation/results
```

To see what is linked now and what runs are available:

```bash
pixi run check-results                              # both pipelines
bash experiments/scripts/check-results.sh spike     # just one
```

`check-results` **only reports** — it never creates or repoints a symlink.
Choosing a run decides which numbers the published docs show, so that call is
left to a human. When the link is missing or broken it lists every run on
disk and prints the `ln -sfn` command to fix it.

> ⚠️ **Never point `results/` into `.worktrees/`.** A worktree is removed when
> its branch lands, leaving a dangling symlink that breaks the docs build in
> the main clone. This has already happened once.

### The dashboard

```bash
pixi run dashboard        # read-only
pixi run dashboard-edit   # editable
```

An interactive [marimo](https://marimo.io) app for exploring fitted
`ModelCollection`s — convergence, GE landscape, parameter correlation,
replicate scatter, sparsity.

> ⚠️ **It discovers `*.pkl` files below the directory you launch it from
> (`cwd`), not from a fixed path.** Launch from the repository root to see
> every run. Dot-hidden directories (`.git/`, `.pixi/`, `.worktrees/`) are
> pruned from the search.

Two gotchas recorded from experience:

- **Pin `marimo<0.23`.** 0.23.x breaks `mo.ui.table` selection.
- Loading `fit_collection.pkl` for the spike prod run needs **~7 GB RSS**
  (the file is 1.76 GB). Prefer the exported CSVs when you only need numbers.

---

## 3. Building and deploying the docs

```bash
pixi run docs          # clean + build to docs/_build/html
pixi run docs-deploy   # build, then push to the gh-pages branch
```

Published at <https://matsengrp.github.io/multidms/> from the `gh-pages`
branch, via `ghp-import`.

### Why the docs need a completed run

Ten `docs/*.nblink` files point at **executed** notebooks inside
`experiments/<pipeline>/results/`:

```
docs/spike_evaluate.nblink → ../experiments/scv2-spike/results/evaluate.ipynb
```

Because `results/` is gitignored, a fresh clone has nothing to resolve, and
Sphinx fails with a bare, misleading error:

```
InputError: [Errno 2] No such file or directory:
  '../experiments/simulation/results/cross_validation.ipynb'
```

**This is not a Sphinx problem.** `pixi run docs` and `docs-deploy` now depend
on `check-results`, which runs first and stops the build with a readable error
naming the pipeline, the runs available on disk, and the `ln -sfn` command to
fix it. Sphinx never starts, so there is no half-built output to clean up.

> Diagnostic habit: when a docs build fails, run `readlink experiments/*/results`
> **first**.

### Adding a docs page for a new analysis

Every new pipeline analysis gets a page. Three steps:

1. Create `docs/spike_<analysis>.nblink`:
   ```json
   {"path": "../experiments/scv2-spike/results/<analysis>.ipynb"}
   ```
2. Add `spike_<analysis>` to the `Spike Analysis` toctree in `docs/index.rst`.
3. Give the notebook real narrative markdown. **These pages are the public
   documentation of the method, not an execution log.**

Verify with `pixi run docs` before opening a PR.

---

## 4. Repository structure

```
multidms/
├── multidms/              # the package
│   ├── jaxmodels.py       #   JAX-native core (equinox, BCOO sparse, jaxopt)
│   ├── data.py  model.py  #   pandas/binarymap wrapper API over jaxmodels
│   ├── model_collection.py#   parallel fitting over parameter grids, CV
│   ├── plot.py            #   ALL Altair rendering; classes delegate here
│   └── utils.py           #   mutation-string parsing, transforms
├── experiments/           # analysis pipelines (see below)
├── docs/                  # Sphinx sources + .nblink stubs
├── tests/
└── HANDOFF.md             # this file
```

The package has **two API layers**: `jaxmodels` is the JAX-native core;
`data.py`/`model.py` are the friendlier pandas-facing wrappers. Convert
between them with `jaxmodels.Data.from_multidms()`. See `CLAUDE.md` for the
full architecture.

### `experiments/`

| Directory | Status | What it is |
|---|---|---|
| `simulation/` | **Live pipeline** | Synthetic DMS with known ground truth. Manuscript Fig 2, S1–S5. |
| `scv2-spike/` | **Live pipeline** | SARS-CoV-2 spike DMS. Nine manuscript figures. |
| `scripts/` | **Infrastructure** | Remote execution + `check-results.sh`. |
| `dashboard.py`, `dashboard_helpers.py` | **Live tooling** | The marimo dashboard. |
| `convergence-lab/` | **Evidence record — keep** | A lab notebook, *not* a pipeline. Its README's "standing findings" are the cited justification for production hyperparameters in both live pipelines (e.g. `simulation/config/config.yaml` refers to it by name). Deleting it orphans those citations. |

### How a pipeline is wired

Both pipelines are Snakemake workflows executing parameterized notebooks via
papermill. Source notebooks live in `notebooks/`; executed copies land in
`results/`.

The spike DAG:

```
prepare_data ──► training_functional_scores.csv
     ├──────────────────────┐
     ▼                      ▼
 fit_models            cross_validation
     │                      │
     ▼                      │
 evaluate ──► mutations_df.csv, collection_muts.csv, …
     └──────────┬───────────┘
                ▼
      manuscript_figures ──► figures/*.pdf, *.png
```

### ⚠️ The config tier split — the thing most likely to be broken by accident

The config is split so a **downstream-only edit cannot invalidate a ~2h20m
model fit**:

| File | Holds | Invalidates the fit? |
|---|---|---|
| `config.yaml` | `fitting:` block, data sourcing, filtering | **Yes** |
| `config_downstream.yaml` | `lasso_choice`, colors, `domain_dict`, `figures:` | **No** |

Rules to preserve:

1. Do **not** add `config_downstream.yaml` to the `input:` of `prepare_data`,
   `cross_validation`, or `fit_models`. That silently restores the defect.
2. New downstream helpers go in `notebooks/_downstream.py`, **never**
   `_common.py` — the latter is `input:` on all four fit-tier rules.
3. `manuscript_figures` reads **CSVs only**; it must never load
   `fit_collection.pkl`.
4. Every config variant needs a matching `<name>_downstream.yaml` sibling.
   The path is derived by string substitution, so a missing sibling fails.

> **`n_processes: 6` is pinned for spike.** Steady-state RSS is ~35–38 GB per
> worker, so six workers need ~230 GB. Restoring `null` auto-sizes by core
> count (~64 workers) and needs multiple TB. This has OOM'd a host before.

Note `maxiter` is overloaded: top-level = outer sweeps; inside
`ge_kwargs`/`cal_kwargs` = inner solver steps.

### Config variants that look like cruft but are not

`config_recompute_false*.yaml` (three pairs) are unreachable from the
Snakefile by profile name and look like leftovers from a finished experiment.
They are **test fixtures**: `tests/test_config_tiers.py` iterates
`SPIKE_VARIANTS` and asserts on each. Deleting them fails four tests.

Removing them is a deliberate two-step change — edit `SPIKE_VARIANTS` first,
then delete the YAMLs.

---

## 5. State of the manuscript work

The active spine is **EPIC #290** — regenerating every manuscript figure from
the current model rather than the archived v0.4.0 notebook.

```
Phase 1  #291  ✅ landed (PR #303)  simulation convergence
Phase 2  #292  ✅ landed (PR #309)  spike refit + the nine-figure surface
Phase 3  #293  🔻 DESCOPED 2026-08-17 → standalone issue, spec'd and ready
Phase 4  #294  ✅ landed (PR #311)  naive per-condition baseline → Fig 3
Phase 5  #295  ✅ closed as delivered-by-Phase-2 (figures shipped in PR #309)
Phase 6  #296  ⬜ stub  ▶ UNBLOCKED  figure manifest + number-diff
Phase 7  #297  ⬜ stub              written handoff for manuscript revision
```

**The epic's remaining work is #296 then #297 — both local, no compute.** All
four fit-bearing phases have landed and every remote run the epic needs is done.

### Phase 3 was descoped — what that means

**The linear (Identity) baseline arm, SI Figure S10, is not part of EPIC #290
anymore.** It lives at **#293** as a standalone issue carrying its full spec,
and it is unblocked today: it needs only the `(tol, maxiter)` pair from #291
and the cached spike fit from #292, both landed.

Nothing in the repo implements it yet — there is no `linear_baseline.ipynb`, no
`rule linear_baseline`, no `spike.linear` config block, no `S10` entry in the
Snakefile's `FIGURE_NAMES`. A reader grepping for those and finding nothing is
seeing the correct state, not a broken checkout.

Consequences to carry into any manuscript work:

- **S10 is a carried-over-unchanged figure**, alongside S8 and S11–S15. It is
  the one SI figure still showing v0.4.0 output while its neighbours were refit
  under the new `(tol, maxiter)` and λ = 8.0e-05.
- **The linear-vs-sigmoid loss gap is unmeasured** — not "unchanged", and not
  "moved". The paper's S10 claim is untested by this work.
- The paper's **central methodological claim** (joint R² ≈ 3.4× naive) is
  unaffected: that is Figure 3, delivered by Phase 4.

> ⚠️ Whoever picks up #293 should re-check its §1a analysis against the
> *current* fit rather than the state of the world when the spec was written.

### Two live scientific results from the Phase 2 refit

**λ moved: `4.0e-05` → `8.0e-05`.** All three selection criteria (CV loss,
replicate correlation, stop-codon sparsity) now agree on the chosen rung, but
the two rungs sit within **0.16%** of each other — corroboration, not a
decisive vote. The Methods paragraph in `main.tex` needs updating.

**A419S retains its contrast — direction preserved.** At λ = 8.0e-05,
`2 ** avg(predicted_func_score)`:

| | Delta | BA.1 | BA.2 |
|---|---|---|---|
| phenotypic effect | 0.854 | 0.132 | 0.137 |
| fold vs Delta | — | 6.4× | 6.2× |

> ⚠️ The paper's **">1,000-fold"** figure describes **measured titers**, not
> the model's predicted enrichment ratio. The model reproduces the contrast
> *direction and ordering*. Stating it otherwise reads as a failed
> replication when it is not.

### A suspected defect that turned out not to be one

**The Figure S10 "erratum" was investigated and refuted. Do not report it.**

An earlier version of this document — and the bodies of #290, #296 and #297 —
stated that legacy notebook cell 103 plots the *sigmoid* collection's CV loss
inside the linear-model figure, making the published S10 middle panel an
erratum against the preprint. **That is false.** The spec work on #293 recovered
the notebook at `fc89753:notebooks/spike-analysis.ipynb` (the previously cited
`6c98b7b` does not resolve in this repo) and read it by 0-based cell index:

| idx | exec | what it actually does |
|---|---|---|
| 103 | 127 | the linear **fit call** — not a loss call |
| 104 | 128 | builds `linear_mc`, adds validation loss |
| 105 | 132 | `cross_validation_df = linear_mc.get_conditional_loss_df()` — **linear, and read** |
| 106 | 133 | renders `shrinkage_analysis_linear_models` — the S10 figure |

Execution counts are monotone 128 → 132 → 133, so the last write to
`cross_validation_df` before S10 rendered was the linear one. Cells 101/102
likewise rebind `sparsity_df` and `corr_df` from the linear collection, so
panels A and C are linear too.

> ⚠️ **Do not tell Hugh the preprint contains an S10 erratum.** Reporting a
> defect that is not there is worse than reporting nothing, and with Phase 3
> descoped there is no reproduction run left in the epic to catch the mistake
> before it reaches the manuscript.
>
> **The supportable sentence:** *"A suspected defect in S10 was investigated
> and refuted. S10 was not regenerated, so the check is not yet decisive —
> #293 carries the reproduction that would settle it."*

The evidential limit is real and is why #293 still treats this as live: stored
execution counts record *an* execution order, not proof the saved PDF came from
it. But "unverified" is not "erratum". Full analysis: **#293 §1a**.

The separate warning that cells 98–100 rebind module-level frames is still a
genuine *fragility* — the two arms share variable names, so a re-run in a
different order would silently mix them — and #293's spec keeps the arms in
separate namespaces for that reason.

### Notation (paper ↔ code)

| Paper | Meaning | Code |
|---|---|---|
| `β_m` | mutation effect in the reference experiment | `beta` |
| `Δ_{d,m}` | shift in experiment `d` vs reference | `shift` |
| `λ` | **"lasso regularization weight"** | `fusionreg` |
| `α_d` | experiment offset | `alpha` |
| `θ₀, θ₁` | sigmoid bias & scale | `theta` |

> The paper never says "fusion regularization". Use **λ / "lasso
> regularization weight"** in prose and captions; `fusionreg` in code.

---

## 6. Open loose ends

**Unmerged PRs:**

- **#282** — v0.4.0 ↔ main equivalence check. Finished work, open since
  2026-07-15. It is the evidence resolving #281 and half of #242. Land or
  close it.
- **#239** — `IndexError` fix in `mut_param_dataset_correlation`, open since
  2026-05-07.

**Backlog notes:**

- **#240** (bug) is live and unfixed: `fit_models_path` truncates paths
  silently. Lower severity today because prod's `config.yaml` sets no
  `strategy` key at all, so it defaults to `"independent"` — but a real trap
  if you switch to `"continuation"`.
- **#281** duplicates #242 and is already executed by PR #282.
- **#176 / #177 / #179** reference a **v2.0 release that never happened**
  (tags go 0.4.2 → 1.0.0 → 1.3.0). #177's target module, `biophysical.py`,
  no longer exists. Retire the "v2.0" vocabulary rather than trying to
  satisfy it.
- **#192 and #302** overlap substantially — both re-express mutation effects
  in a condition's own coordinates. Merge them when #302 is specced.
- **#295** was **closed as delivered-by-Phase-2**; the log-x requirement for
  Figure 5 was explicitly dropped, not deferred. Its (closed) body remains the
  richest record of the Figure 4 zoom regions and the Figure 5 x-axis warning —
  the axis is `2 ** avg_predicted_func_score`, the predicted enrichment ratio,
  **not** β and **not** shift, despite the legacy name `predicted_beta`. The
  closing audit comment restates both. Note also that the manuscript's
  ">1,000-fold" A419S claim refers to **measured titers**, not the model's
  predicted ratio (predicted: Delta 0.854 / BA.1 0.132 / BA.2 0.137).

**Housekeeping:**

- `experiments/loss-normalization/` is dead: nothing references it, and it
  still uses the pre-tier-split single-argument `load_config()`, so it could
  not run today. Safe to delete along with its `CLAUDE.md` line.
- Branch `246-convergence-lab` is **ahead of its remote by one commit**
  (`5ac7477`, "preserve uncommitted SWEEP_PLAN + sweep runner"). Push or
  discard it before deleting the branch.
- Two `.claude/worktrees/agent-*` worktrees hold unreviewed experiments
  (an `alpha_ridge` knob; a `BiasedSigmoid` GE with a fitted lower plateau).
