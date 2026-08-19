# Handoff

**For whoever writes the manuscript revision next.** This assumes you know
the science — deep mutational scanning, global epistasis, the joint shift
model, what the paper claims. It assumes you have never read this codebase.

It answers, in order: where the work stands, **how to get the results and
look at them** (start there if you have just cloned), what is still open,
which figures exist, how to run the pipeline yourself, how to build the docs,
and which scientific findings must survive the handoff.

Verified against the repository on **2026-08-18**.

Reference material lives elsewhere and is not repeated here:

| For | Read |
|---|---|
| Package architecture, code style, dev commands, the XLA JIT leak | [`CLAUDE.md`](CLAUDE.md) |
| Contribution workflow | [`CONTRIBUTING.rst`](CONTRIBUTING.rst) |
| Pipeline internals, config tiers, run log | [`experiments/scv2-spike/README.md`](experiments/scv2-spike/README.md) |
| Pipeline index and remote setup | [`experiments/README.md`](experiments/README.md) |
| **The legacy analysis notebook** — where the *published* manuscript figures were generated | [`SARS-CoV-2_spike_multidms` @ `6c98b7b`](https://github.com/matsengrp/SARS-CoV-2_spike_multidms/blob/6c98b7b607d7387b508cdaa192d659ee9fca7367/spike-analysis.ipynb) |

> ⭐ **That pinned commit is the old-vs-new comparison baseline.** The figures
> in the published preprint came from it, under multidms v0.4.0. When you need
> to know whether a number moved because the model changed or because the code
> changed, compare against `6c98b7b` rather than against the current legacy
> `main`, which has drifted.

---

## 1. Where things stand

**What you can trust today:**

- The **spike production fit** is final: 10 lasso rungs × 2 replicates, at
  `tol 1e-6 / maxiter 500`, with the chosen λ = **8.0e-05**. Its
  `fit_collection.pkl` is 1,761,133,462 bytes, md5
  `b2b4736073e475a6fd7b1b5260063d6c`.
- The **simulation fit** is final at 649,380,559 bytes, md5
  `f602baf4801bb9257a1f22281da99f49`.
- **10 of the manuscript's 22 figures** regenerate from the live pipeline
  today; §4 names the other 12 and who owns each.
- The two scientific results in §7 (**λ moved**, **A419S**) are measured on
  this fit and are ready to go into prose.

**What is not done:** six simulation figures (#316), the linear baseline arm
for S10 (#293), and the manuscript prose itself.

---

## 2. Getting the results and looking at them

**Start here if you have just cloned the repo.** This is the full path from a
fresh clone to viewing the manuscript figures and exploring the fits. It
should take under an hour, most of it waiting on the ~4.5 GB download.

### Step 1 — set up the environment

The project uses [pixi](https://pixi.sh). It manages its own Python; you do
not need conda, venv, or a system Python. **Never run bare `python` or `pip`
here** — everything goes through `pixi run`.

```bash
pixi install          # solves and creates .pixi/envs/ — a few minutes
pixi run test         # sanity check: 134 tests should pass
```

The default environment (Python **3.11**) is the one you want; `py39` /
`py310` / `py311` / `py312` exist only for the CI matrix
(`pixi run -e py39 test`). Everything below assumes the default.

> ⚠️ **`pixi.lock` is gitignored, so a fresh clone re-solves from scratch.**
> You do not get the exact environment that produced these results. That
> matters more than it looks: the `jax` / `jaxopt` pins in `pyproject.toml`
> carry comments explaining that resolver drift there previously caused a
> **SIGABRT inside jaxopt's proximal solver**. If you hit an inexplicable
> crash during fitting on a fresh machine, suspect the resolve before you
> suspect the model.

### Step 2 — configure the remote

Results are **not in git** — they are ~4.5 GB of pickles, CSVs and PDFs living
on `ermine`. Create this file once:

```bash
mkdir -p ~/.config/multidms-experiments
cat > ~/.config/multidms-experiments/remote.yaml << 'EOF'
host: your-username@ermine
remote_dir: /fh/fast/matsen_e/shared/multidms/multidms
EOF
```

`host` and `remote_dir` are both **required**; the scripts abort with a
template if the file is missing.

### Step 3 — fetch both payloads and link them

```bash
REMOTE=your-username@ermine
BASE=/fh/fast/matsen_e/shared/multidms/multidms/experiments

# spike — 3.3 GB
rsync -a --info=progress2 \
  "$REMOTE:$BASE/scv2-spike/results-prod-294-naive-baseline-arm/" \
  experiments/scv2-spike/results-prod-294-naive-baseline-arm/
ln -sfn results-prod-294-naive-baseline-arm experiments/scv2-spike/results

# simulation — 1.2 GB
rsync -a --info=progress2 \
  "$REMOTE:$BASE/simulation/results-prod-sim-vpl500-tol1e5/" \
  experiments/simulation/results-prod-sim-vpl500-tol1e5/
ln -sfn results-prod-sim-vpl500-tol1e5 experiments/simulation/results

pixi run check-results     # ✓ per pipeline, or an explanatory ✗
```

> ⚠️ **The `ln -sfn` lines are not optional.** `results` is a gitignored
> symlink, so it never survives a clone and nothing recreates it for you.
> Without it `pixi run docs` dies with a bare "No such file or directory" and
> the dashboard finds nothing — the most common failure on a fresh machine.

### Step 4 — view the figures

Rendered figures are in `results/figures/`, as **both PDF and PNG**:

```bash
open experiments/scv2-spike/results/figures/          # macOS
```

The manifest in §4 maps every manuscript figure number to its filename — read
it before hunting, because **two filenames are deliberately misleading traps**
(§4). The executed notebooks (`results/*.ipynb`) carry the same figures inline
with the code that made them.

### Step 5 — explore the fits interactively

An interactive [marimo](https://marimo.io) dashboard explores fitted
`ModelCollection`s — convergence, GE landscape, parameter correlation,
replicate scatter, sparsity.

```bash
pixi run dashboard        # read-only
pixi run dashboard-edit   # editable
```

> ⚠️ **Launch it from the repository root.** It discovers `*.pkl` files below
> the directory you launch it from (`cwd`), not from a fixed path. Dot-hidden
> directories (`.git/`, `.pixi/`, `.worktrees/`) are pruned from the search.

**The long waits are not hangs.** This is the most common way to mistake
working software for broken software here:

- Loading the spike `fit_collection.pkl` needs **~7 GB RSS** and takes a
  while. The file is 1.76 GB.
- **The Param Correlation tab is slow by design.** It is button-gated: select
  the fits, set the threshold, press **Plot**, then wait — **minutes** on a
  prod-sized collection. That is expected. Do not kill it.
- `marimo` is pinned `>=0.8,<0.23` in `pyproject.toml`; 0.23.x breaks
  `mo.ui.table` selection. Do not raise it.

### What is in a results directory

| File | What it holds |
|---|---|
| `figures/` | The rendered PDFs and PNGs listed in §4. |
| `mutations_df.csv` | Per-mutation β and shifts — **the table most manuscript numbers come from.** |
| `cross_validation_loss.csv` | CV loss per λ rung; the evidence for the λ choice. |
| `fit_sparsity.csv`, `library_replicate_correlation.csv` | The other two λ selection criteria. |
| `fit_convergence.csv`, `convergence_trajectory.csv` | Per-fit convergence; check before trusting any fit. |
| `fit_collection.pkl` | The fitted `ModelCollection`. Large (1.76 GB for spike), ~7 GB RSS to load. |
| `*.ipynb` | The executed notebooks, with outputs, for every pipeline stage. |

> ⭐ **Prefer the CSVs.** Almost every number in the manuscript can be read
> from `mutations_df.csv` and the three selection-criterion CSVs in seconds,
> without ever loading the pickle.

### Where this all lives on the remote

```
/fh/fast/matsen_e/shared/multidms/
├── multidms/                       ← canonical clone, on up-to-date main
│   └── experiments/
│       ├── scv2-spike/
│       │   ├── results-prod-294-naive-baseline-arm/   (3.3 GB)
│       │   └── results -> results-prod-294-naive-baseline-arm
│       └── simulation/
│           ├── results-prod-sim-vpl500-tol1e5/        (1.2 GB)
│           └── results -> results-prod-sim-vpl500-tol1e5
└── archive/
    ├── archive-2026-08-18/         ← 9 superseded payloads + loose dirs
    └── <existing 2023-2024 material>
```

Nothing was deleted in the cleanup, only moved — if you need an older run, it
is under `archive/`.

---

## 3. Issue status

Re-queried 2026-08-18: **14 open issues** — 13 once this change lands and
closes #297 — and **#282 is the only open PR** besides it.

### Live — someone should act on these

| # | What | Note |
|---|---|---|
| **#316** | Emit Fig 2 and S1–S5 from the simulation pipeline | The largest remaining figure gap. Mostly renames — see §4. |
| **#293** | Linear (Identity) baseline arm → SI S10 | Spec'd, unblocked, no compute dependency left. |
| **#282** | v0.4.0 ↔ main equivalence check | **The only open PR.** Open since 2026-07-15. Finished work; it is the evidence resolving #281. Land or close it. |
| **#318** | Write the three missing docs pages | The placeholders they replace were deleted. |
| **#313** | Spike prep diverges from legacy (codon deletions, replicate subset) | Affects data prep, not the fitted model. |
| **#312** | `mut_type()` mislabels in-frame codon deletions | Related to #313. |
| **#192** | Condition-specific mutation names in `get_mutations_df` | Fully spec'd. #302 was closed as a duplicate of it. |
| **#319** | `IndexError` in `mut_param_dataset_correlation` | Live on `main` at `model_collection.py:1471`, but **benign for the manuscript**: it needs a `(mut_param, x)` cell surviving in only one replicate, which arises under `strategy: continuation`. Every published figure comes from independent-strategy fits. Branch `fix/mut-param-correlation-1col` has a starting patch. |
| **#179** | Remove deprecated `phenotype_as_effect` | Small cleanup. |
| **#99** | Citation | Small. |

### Parked and reference-only

| # | What | Why |
|---|---|---|
| #290 | The epic this work was tracked under | Its phases are done or re-homed; close it when #316 and #293 land. |
| #243, #51 | Concurrent single-solve fitting; a ridge penalty that doesn't bias toward WT | Design questions, not manuscript blockers. |
| #281 | Re-express the 0.4.0 spike model in main's form | Already executed by PR #282. |

> **Issues labelled `question` or `wontfix` were kept only for reference.**
> They record decisions and dead ends, not work. Unless you find one that
> matters to you, they can safely be deleted.

> ⚠️ **#240** is closed, but its `fit_models_path` truncation bug was never
> fixed in code. It is live and low-severity:
> prod sets no `strategy` key, so it defaults to `"independent"`. It becomes a
> real trap only if you switch to `"continuation"`.

> **#295** was closed as delivered, and the log-x requirement for Figure 5 was
> **dropped, not deferred**. Its closed body remains the richest record of the
> Figure 4 zoom regions and the Figure 5 x-axis warning: that axis is
> `2 ** avg_predicted_func_score`, the predicted enrichment ratio — **not** β
> and **not** shift, despite the legacy name `predicted_beta`.

---

## 4. Figure status


The manuscript includes **22 figures** (via `\includegraphics` in
`main.tex`/`si.tex` at `f79ac4a`, excluding four commented-out template
placeholders). The current spike pipeline
(`experiments/scv2-spike/results/figures/`, symlinked to
`results-prod-294-naive-baseline-arm/`) regenerates **10** of them.
The other **12** are listed below with an owner for each.

**Regenerated (10):**

| Fig | File stem |
|---|---|
| 3 | `shift_distribution_correlation_naive` |
| 4 | `shift_by_site_heatmap_zoom` |
| 5 | `validation_titer_fold_change` |
| S6 | `raw_data_summary_barcodes_backgrounds_hist` |
| S7 | `replicate_functional_score_correlation_scatter` |
| S9 | `shrinkage_analysis_trace_plots_beta` |
| S11 | `percent_shifts_under_x_lineplot` |
| S12 | `shift_corr_Delta_BA2` |
| S16 | `convergence_all_lasso_lines` |
| S17 | `global_epistasis_and_prediction_correlations` |

**Missing (12):**

| Manuscript | File stem | Owner |
|---|---|---|
| Fig 2 | `simulation_validation` | #316 — renamed from `main_figure`, see below |
| S1 | `shift_heatmaps_supp` | #316 — built but unsaved (notebook cell 15), needs 2 new panels |
| S2 | `beta_recovery_supp` | #316 — renamed from `ground_truth_correlation` |
| S3 | `shift_corr_supp` | #316 — no current producer |
| S4 | `underdetermined_shifts_supp` | #316 — renamed from `sparsity_diagnostic` |
| S5 | `diff_sim_conditions_supp` | #316 — no current producer, needs 2 new sim conditions |
| S8 | `reference_model_comparison_params_scatter` | out of scope — no issue |
| S10 | `shrinkage_analysis_linear_models` | #293 |
| S13 | `structure_and_neighbor_statistics_scatter` | out of scope — no issue |
| S14 | `mut_effect_vs_shift_multiple_studies` | out of scope — no issue |
| S15 | `shift_corr_with_other_studies` | out of scope — no issue |
| Fig 1 | `summary_of_approach` | out of scope — no issue (hand-drawn schematic) |

> ⚠️ **The six simulation-figure rows (Fig 2, S1–S5) are mostly a naming
> mismatch, not an absent analysis.** Three already exist under
> different names in the simulation notebook (`main_figure` →
> `simulation_validation`; `ground_truth_correlation` →
> `beta_recovery_supp`; `sparsity_diagnostic` →
> `underdetermined_shifts_supp`); one is built but never saved (cell
> 15's heatmap, needed for S1 panel A); only S3 and S5 have no current
> producer at all. Regenerating the three renames and saving the
> fourth **does not re-run the simulation fits** — they are figure-tier
> work against the already-cached `fit_collection.pkl` (verified
> 2026-08-18: `snakemake --touch` cleared an mtime-only cascade with
> the 649,380,559-byte pickle staying byte-identical, and a subsequent
> forced dry run against a figure target re-ran only
> `manuscript_figures` and the `all` aggregator — `total: 2`, no
> fit-tier rule). S3, S5, and S1's panels B/C need two new
> simulation conditions and do require a simulate-and-fit pass. See
> #316 for the full breakdown.
>
> ⚠️ **Two filename traps — do not "fix" these.** Manuscript Figure 3 is
> labelled `fig:shift_distribution_correlation_effect` in the LaTeX but
> actually includes `..._naive.pdf` (an orphaned `..._effect.pdf` is an
> older, pre-naive-panel version). SI S13 is labelled
> `fig:shifts_3D_structure` but includes
> `structure_and_neighbor_statistics_scatter.pdf` (a decoy
> `shifts_3D_structure.pdf` also exists on disk and is not the included
> file). Both mismatches are deliberate: match on what's included, never
> on the label. The Figure 3 trap is also recorded in the Snakefile's
> `FIGURE_NAMES` comment; **S13's is not, because S13 has no producer
> yet** — if you write one (#316-style), carry this warning into it.

---


---

## 5. Running the pipeline

Do this whenever the revision needs numbers that do not exist yet: a new
model arm, a different λ grid, an ablation a reviewer asked for, a rerun after
a code change. Both pipelines are Snakemake workflows driven by a profile.

### Start with the test profile

```bash
pixi run spike-test   # ~10 min, 10% subsample
pixi run sim-test     # ~5 min
```

**Always run `-test` before `-prod`.** It exercises the whole DAG end to end
in minutes, so a broken config, a bad path, or a notebook that raises fails
immediately rather than after hours of fitting. The prod profiles are
`pixi run spike-prod` and `pixi run sim-prod`; spike also has `experimental`.

### Production runs go on a remote host

The spike prod fit needs ~35–38 GB RSS **per worker** with 20 workers. That is
a server job, not a laptop job.

```bash
# 1. pick an idle host first — a busy one will thrash at 20 workers
#    (lab tooling: `bip scout`; otherwise check load however you normally do)
# 2. launch, poll, fetch:
pixi run remote-pipeline -- spike prod host=orca03     # launches in tmux
pixi run remote-status  -- spike prod host=orca03      # poll progress
pixi run run-pull       -- spike prod host=orca03      # fetch results back
```

All three take `<pipeline> <profile> [key=value ...]`. The `--` matters: it
separates pixi's own arguments from the script's, and `host=` overrides
`~/.config/multidms-experiments/remote.yaml`, which you create once with
`host:` and `remote_dir:` keys (see `experiments/README.md`).

**Commit before launching.** The launcher warns on a dirty tree because the
remote checks out your branch — it never sees uncommitted work. When the run
finishes, kill the tmux session; leaving it holds the host.

> ⭐ **A new run cannot overwrite the manuscript payload.** `output_dir` is
> derived automatically as `results-<profile>-<branch>`, and non-`main`
> branches get their own remote worktree. Work on a branch and your run lands
> in its own directory; the manuscript's `results-prod-294-naive-baseline-arm`
> is untouched. This is the property that makes experimenting safe.

### Changing what gets fit

Fitting knobs live in the `fitting:` block of each pipeline's `config.yaml` —
λ grid, `tol`, `maxiter`, the loss. Editing that file **intentionally**
invalidates the fit, which is exactly what you want when changing the model.
The **config tier split** — which edits cost a refit and which do not — is
documented in
[`experiments/scv2-spike/README.md`](experiments/scv2-spike/README.md); read
it before editing anything under `config/`.

For a genuinely different configuration, prefer a **config variant** —
`config_<name>.yaml` plus its required `config_<name>_downstream.yaml`
sibling — over editing the production config in place. That keeps the
manuscript's configuration reproducible alongside your new one.

Pipeline internals, the DAG, and the run log of past production fits are in
[`experiments/scv2-spike/README.md`](experiments/scv2-spike/README.md); the
remote setup is in [`experiments/README.md`](experiments/README.md).

---

## 6. Building the docs

```bash
pixi run docs          # build to docs/_build/html
pixi run docs-deploy   # publish to gh-pages
```

> ⚠️ **The build needs the `results/` symlink.** The `.nblink` files in
> `docs/` point at `experiments/*/results/*.ipynb`. Those symlinks are not
> tracked in git, so a **fresh clone fails to build** until you fetch a
> results payload (§2) and recreate them. This is the usual cause of a
> mystifying docs failure on a machine that has never run a pipeline.

The docs render the executed pipeline notebooks directly, so the published
site reflects whichever run `results` points at.

---

## 7. Science to carry forward

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


*Verified against the repository on 2026-08-18.*
