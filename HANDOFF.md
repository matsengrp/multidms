# Handoff

**For whoever writes the manuscript revision next.** This assumes you know
the science — deep mutational scanning, global epistasis, the joint shift
model, what the paper claims. It assumes you have never read this codebase.

It answers, in order: where the work stands, what is still open, which
figures exist, where the results live, how to run the pipeline yourself, how
to look at the results, how to build the docs, and which scientific findings
must survive the handoff.

Verified against the repository on **2026-08-18**.

Reference material lives elsewhere and is not repeated here:

| For | Read |
|---|---|
| Package architecture, code style, dev commands, the XLA JIT leak | [`CLAUDE.md`](CLAUDE.md) |
| Contribution workflow | [`CONTRIBUTING.rst`](CONTRIBUTING.rst) |
| Pipeline internals, config tiers, run log | [`experiments/scv2-spike/README.md`](experiments/scv2-spike/README.md) |
| Pipeline index and remote setup | [`experiments/README.md`](experiments/README.md) |

---

## 1. Where things stand

**The compute is done.** Every fit the manuscript needs has run, and the
results are on disk, so you can write the revision without refitting anything.

That is a statement about what the *current* manuscript needs, not a warning
against running the pipeline. If the revision calls for a new arm — a reviewer
asks for an ablation, you want a different λ grid, you change the model —
running it is a normal, well-supported operation. **§5 is the guide for doing
that.** The one thing to avoid is refitting *by accident*, which happens
through the config tier split (§8) and produces a fit you did not intend.

The spine of the work was **EPIC #290** — regenerating every manuscript figure
from the current model rather than from the archived v0.4.0 notebook.

```
Phase 1  #291  ✅ landed (PR #303)  simulation convergence
Phase 2  #292  ✅ landed (PR #309)  spike refit + the nine-figure surface
Phase 3  #293  🔻 DESCOPED 2026-08-17 → standalone issue, spec'd and ready
Phase 4  #294  ✅ landed (PR #311)  naive per-condition baseline → Fig 3
Phase 5  #295  ✅ closed as delivered-by-Phase-2 (figures shipped in PR #309)
Phase 6  #296  ✅ landed (PR #317)  figure manifest
Phase 7  #297  ◀ this change       cleanup + this document
```

**What you can trust today:**

- The **spike production fit** is final: 10 lasso rungs × 2 replicates, at
  `tol 1e-6 / maxiter 500`, with the chosen λ = **8.0e-05**. Its
  `fit_collection.pkl` is 1,761,133,462 bytes, md5
  `b2b4736073e475a6fd7b1b5260063d6c`.
- The **simulation fit** is final at 649,380,559 bytes, md5
  `f602baf4801bb9257a1f22281da99f49`.
- **10 of the manuscript's 22 figures** regenerate from the live pipeline
  today; §3 names the other 12 and who owns each.
- The two scientific results in §8 (**λ moved**, **A419S**) are measured on
  this fit and are ready to go into prose.

**What is not done:** six simulation figures (#316), the linear baseline arm
for S10 (#293), and the manuscript prose itself.

---

## 2. Issue status

Re-queried 2026-08-18. **21 open issues; #282 is the only open PR.**

### Live — someone should act on these

| # | What | Note |
|---|---|---|
| **#316** | Emit Fig 2 and S1–S5 from the simulation pipeline | The largest remaining figure gap. Mostly renames — see §3. |
| **#293** | Linear (Identity) baseline arm → SI S10 | Spec'd, unblocked, no compute dependency left. |
| **#282** | v0.4.0 ↔ main equivalence check | **The only open PR.** Open since 2026-07-15. Finished work; it is the evidence resolving #281 and half of #242. Land or close it. |
| **#319** | `IndexError` in `mut_param_dataset_correlation` | **Live bug**, re-raised in this change — see below. |
| **#318** | Write the three missing docs pages | Filed in this change; the placeholders they replace were deleted. |
| **#313** | Spike prep diverges from legacy (codon deletions, replicate subset) | Affects data prep, not the fitted model. |
| **#312** | `mut_type()` mislabels in-frame codon deletions | Related to #313. |
| **#302** | Report mutational effects in each condition's own coordinates | Overlaps #192 — see below. |

### The one live bug worth knowing about

**#319 — `IndexError` in `mut_param_dataset_correlation` for
single-replicate cells.** Still live on `main` at
`multidms/model_collection.py:1471`. When a `(mut_param, x)` cell survives in
only one replicate, `.corr()` returns a 1×1 matrix and `.iloc[0, 1]` indexes
past its bound.

> ✅ **No manuscript figure is affected.** The bug triggers only on sparse
> shift solutions, which arise under `strategy: continuation`. Every published
> figure comes from independent-strategy fits, whose denser solutions never
> produce a single-replicate cell.

PR #239 proposed a fix and sat untouched from 2026-05-07; it was **closed
unmerged** in this change and the defect re-raised as #319 with a fresh
re-assessment. The branch `fix/mut-param-correlation-1col` is kept — its
patch and two tests are a starting point.

### Parked — real, but not blocking the manuscript

| # | What | Why parked |
|---|---|---|
| #243 | Concurrent (single-solve) fitting | Performance work. |
| #242 | Score the manuscript's parameters under the PR #164 objective | Half-answered by #282. |
| #241 | Reduced per-block optimization iterations | Superseded in practice by the tuning already done. |
| #197 | Learnable `FlexibleSigmoid` GE | Feature work. |
| #192 | Condition-specific mutation names | **Overlaps #302** — both re-express effects in a condition's own coordinates. Merge them when #302 is specced. |
| #51 | A ridge penalty that doesn't bias toward WT | Long-standing design question. |
| #155, #99 | Docs formatting, citation | Small. |

### Retire the vocabulary, not just the issues

> ⚠️ **#176 / #177 / #179 reference a v2.0 release that never happened.** The
> tags go 0.4.2 → 1.0.0 → 1.3.0. #177's target module, `biophysical.py`, no
> longer exists. Do not try to satisfy these as written — retire the "v2.0"
> framing first, then decide what (if anything) is left.

> **#281** duplicates #242 and is already executed by PR #282.
>
> **#240**'s `fit_models_path` truncation bug is live but low-severity: prod
> sets no `strategy` key, so it defaults to `"independent"`. It becomes a real
> trap only if you switch to `"continuation"`.
>
> **#295** was closed as delivered-by-Phase-2, and the log-x requirement for
> Figure 5 was **dropped, not deferred**. Its closed body remains the richest
> record of the Figure 4 zoom regions and the Figure 5 x-axis warning: that
> axis is `2 ** avg_predicted_func_score`, the predicted enrichment ratio —
> **not** β and **not** shift, despite the legacy name `predicted_beta`.

---

## 3. Figure status


The manuscript includes **22 figures** (via `\includegraphics` in
`main.tex`/`si.tex` at `f79ac4a`, excluding two commented-out template
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
> forced dry run against a figure target reported `total: 1`, rule
> `manuscript_figures` alone). S3, S5, and S1's panels B/C need two new
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
> file). Both mismatches are deliberate and already documented in the
> Snakefile's `FIGURE_NAMES` comment — match on what's included, never
> on the label.

---


---

## 4. Where the results live

Everything is on `ermine`, under the shared lab directory. The layout below
is the **post-cleanup** state as of 2026-08-18:

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
    ├── archive-2026-08-18/         ← 9 rescued payloads + loose dirs
    │   ├── results/                  (the 9 results-prod-* runs)
    │   └── stale-april-results/
    └── <existing 2023-2024 material>
```

The clone holds exactly one payload per pipeline, and `results` is a
**symlink** to it. Everything superseded lives under `archive/` — nothing was
deleted in the cleanup, only moved.

### Getting a copy locally

```bash
# from your local clone
rsync -a --info=progress2 \
  ermine:/fh/fast/matsen_e/shared/multidms/multidms/experiments/scv2-spike/results-prod-294-naive-baseline-arm/ \
  experiments/scv2-spike/results-prod-294-naive-baseline-arm/
ln -sfn results-prod-294-naive-baseline-arm experiments/scv2-spike/results
```

Recreate the `results` symlink after any fetch — it is **not** tracked in git,
and several things break without it (§7).

### What is in a results directory

| File | What it holds |
|---|---|
| `fit_collection.pkl` | The fitted `ModelCollection`. Large (1.76 GB for spike) and slow to load (~7 GB RSS). |
| `mutations_df.csv` | Per-mutation β and shifts — **the table most manuscript numbers come from.** |
| `cross_validation_loss.csv` | CV loss per λ rung; the evidence for the λ choice. |
| `fit_sparsity.csv`, `library_replicate_correlation.csv` | The other two λ selection criteria. |
| `fit_convergence.csv`, `convergence_trajectory.csv` | Per-fit convergence; check before trusting any fit. |
| `figures/` | The rendered PDFs and PNGs listed in §3. |
| `*.ipynb` | The executed notebooks, with outputs, for every pipeline stage. |

> ⭐ **Prefer the CSVs.** Almost every number in the manuscript can be read
> from `mutations_df.csv` and the three selection-criterion CSVs without ever
> loading the pickle.

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
λ grid, `tol`, `maxiter`, `n_processes`, the loss. Editing that file
**intentionally** invalidates the fit, which is exactly what you want when
changing the model. Read the tier split in §8 first so you know which edits
cost a refit and which do not.

For a genuinely different configuration, prefer a **config variant** —
`config_<name>.yaml` plus its required `config_<name>_downstream.yaml`
sibling — over editing the production config in place. That keeps the
manuscript's configuration reproducible alongside your new one.

### ⚠️ Set `n_processes` to at least the number of fits

The one non-obvious failure mode, and it looks like a scientific result when
it is not. JAX/XLA leaks executable JIT mappings across sequential fits in a
single process, so a worker handling more than ~5 fits dies with
`Unable to allocate section memory` — even on a host with 1.4 TB free.

Failures land **by queue position, not by hyperparameter**, so it presents as
"the high-λ rungs failed" when the λ value had nothing to do with it. Spike
pins `n_processes: 20` for 10 λ rungs × 2 replicates: one fit per worker, so
no process ever compiles twice. Never set it to `null` — auto-sizing counts
cores (~64) and ignores memory, which has OOM'd a host. Full writeup in
[`CLAUDE.md`](CLAUDE.md).

### Check convergence before trusting anything

`fit_convergence.csv` and `convergence_trajectory.csv` are the first things to
read after a run. A fit that hit `maxiter` without converging will still
produce figures — they will just be wrong.

Pipeline internals, the DAG, and the run log of past production fits are in
[`experiments/scv2-spike/README.md`](experiments/scv2-spike/README.md); the
remote setup is in [`experiments/README.md`](experiments/README.md).

---

## 6. Looking at the results interactively

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

**The long waits are not hangs.** This is the single most common way to
mistake working software for broken software here:

- Loading the spike prod `fit_collection.pkl` needs **~7 GB RSS** and takes a
  while. The file is 1.76 GB.
- **The Param Correlation tab is slow by design.** It is button-gated: select
  the fits, set the threshold, press **Plot**, then wait — **minutes** on a
  prod-sized collection. That is expected. Do not kill it.
- **Pin `marimo<0.23`.** 0.23.x breaks `mo.ui.table` selection.

> If you only need numbers, read the exported CSVs (§4) instead — it is
> seconds rather than minutes.

---

## 7. Building the docs

```bash
pixi run docs          # build to docs/_build/html
pixi run docs-deploy   # publish to gh-pages
```

> ⚠️ **The build needs the `results/` symlink.** The `.nblink` files in
> `docs/` point at `experiments/*/results/*.ipynb`. Those symlinks are not
> tracked in git, so a **fresh clone fails to build** until you fetch a
> results payload (§4) and recreate them. This is the usual cause of a
> mystifying docs failure on a machine that has never run a pipeline.

The docs render the executed pipeline notebooks directly, so the published
site reflects whichever run `results` points at.

---

## 8. Science to carry forward


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



**The linear (Identity) baseline arm, SI Figure S10, is not part of EPIC #290
anymore.** It lives at **#293** as a standalone issue carrying its full spec,
and it is unblocked today: it needs only the `(tol, maxiter)` pair from #291
and the cached spike fit from #292, both landed.

Nothing in the repo implements it yet — there is no `linear_baseline.ipynb`, no
`rule linear_baseline`, no `spike.linear` config block, no `S10` entry in the
Snakefile's `FIGURE_NAMES`. A reader grepping for those and finding nothing is
seeing the correct state, not a broken checkout.

Consequences to carry into any manuscript work:

- **S10 is a carried-over-unchanged figure**, alongside S8 and S13–S15. It is
  the one SI figure still showing v0.4.0 output while its neighbours were refit
  under the new `(tol, maxiter)` and λ = 8.0e-05.
- **The linear-vs-sigmoid loss gap is unmeasured** — not "unchanged", and not
  "moved". The paper's S10 claim is untested by this work.
- The paper's **central methodological claim** (joint R² ≈ 3.4× naive) is
  unaffected: that is Figure 3, delivered by Phase 4.

> ⚠️ Whoever picks up #293 should re-check its §1a analysis against the
> *current* fit rather than the state of the world when the spec was written.



| Paper | Meaning | Code |
|---|---|---|
| `β_m` | mutation effect in the reference experiment | `beta` |
| `Δ_{d,m}` | shift in experiment `d` vs reference | `shift` |
| `λ` | **"lasso regularization weight"** | `fusionreg` |
| `α_d` | experiment offset | `alpha` |
| `θ₀, θ₁` | sigmoid bias & scale | `theta` |

> The paper never says "fusion regularization". Use **λ / "lasso
> regularization weight"** in prose and captions; `fusionreg` in code.


### ⚠️ The config tier split — the thing most likely to be broken by accident

This is the mechanism that keeps a figure tweak from destroying a ~2h20m
model fit. Understand it before editing anything under `config/`.

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

> ⚠️ **`config.yaml` is itself a rule `input:`, so Snakemake hashes the file,
> not its meaning.** Even a comment-only edit marks the fit out of date. When
> you know the change cannot affect results, `snakemake --touch` re-stamps the
> outputs instead of refitting — but verify `fit_collection.pkl` is
> byte-identical afterward, and never point `--touch` at a hand-picked subset.

> **`n_processes: 20` is pinned for spike** — one worker per fit (10 λ rungs ×
> 2 replicates). Two independent reasons, both load-bearing:
>
> - **Never set it to `null`.** Auto-sizing picks workers by core count alone
>   (~64), ignoring memory. Steady-state RSS is ~35–38 GB per worker. This has
>   OOM'd a host.
> - **Never set it below the number of fits.** JAX/XLA leaks JIT mappings
>   across sequential fits in one process; a worker handling more than ~5 dies
>   outright. One fit per worker means no process compiles twice. See
>   [`CLAUDE.md`](CLAUDE.md).

Note `maxiter` is overloaded: top-level = outer sweeps; inside
`ge_kwargs`/`cal_kwargs` = inner solver steps.


`config_recompute_false*.yaml` (three pairs) are unreachable from the
Snakefile by profile name and look like leftovers from a finished experiment.
They are **test fixtures**: `tests/test_config_tiers.py` iterates
`SPIKE_VARIANTS` and asserts on each. Deleting them fails four tests.

Removing them is a deliberate two-step change — edit `SPIKE_VARIANTS` first,
then delete the YAMLs.

---

*This document replaced an installation-first handoff on 2026-08-18 (#297),
alongside the repository cleanup described above.*
