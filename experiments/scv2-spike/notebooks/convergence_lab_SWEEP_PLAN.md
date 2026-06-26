# Convergence-lab sweep — plan & living findings (#246)

**Status:** Stage A designed, not yet run · **Started:** 2026-06-24 · **Data:**
`results-prod-235-times-seen-threshold/training_functional_scores.csv` (full,
both replicates, conditions Delta / Omicron_BA1 / Omicron_BA2, reference =
Omicron_BA1, Sigmoid GE).

This file is the **source of truth** for the iterative sweep that follows the
first convergence-lab factorial (`convergence_lab_FINDINGS.md`). The goal is a
model configuration that converges **well, fast, and reproducibly**. It holds
the overall plan and a per-stage findings section we fill in as each stage
completes. Read it cold; it should explain both what we are doing and why.

---

## The diagnosis (measured from the 24-fit cache, not hypothesized)

The first factorial left an open question: is the small-α basin that the
`fixed-scale + no-warmstart` arm settles into genuinely degenerate, or just
different? **Measuring `α` and `Σβ²` directly across the cached fits answers it:
yes, it is degenerate.** There is an α/β see-saw in the Sigmoid GE.

| arm | α range | β L2-total (Σβ²) range |
|---|---|---|
| `warmstart=True`  | **3 – 8**   | **350 – 1400** ✓ healthy |
| `warmstart=False` | **1.4 – 3.3** | **1700 – 75,000** ⚠ exploded |

The sigmoid sees `α·φ`. When `warmstart=False`, the optimizer collapses α toward
~1.5 and lets β explode (up to Σβ² ≈ **75,000**, max|φ| ≈ **4751**) so that the
product `α·φ` stays bounded and the Huber loss (total ≈ 0.66, per-variant
`.mean()` scale) barely moves. Two near-degenerate `(α, β)` combinations fit
equally well, so **noise decides which basin each replicate lands in** — which
is precisely why loss looks flat, objective error oscillates, and replicate
shifts do not correlate.

The β-explosion is driven by **`warmstart=False`** (not by `recompute_scale`),
and it worsens sharply once `fusionreg > 0`. The user's original observation
("rep_1 φ ∈ ±300, rep_2 fine") was the *mild* end of a systematic effect.

> **Therefore the headline lever is `l2reg`** — an L2 penalty on β attacks the
> explosion at its root, independent of warmstart. The first factorial never
> swept it (`l2reg=0` throughout).

### Why the l2reg grid below is data-anchored, not blind

`l2reg` penalizes `Σβ²` against a loss already normalized to ≈ 0.66. For the
penalty to *bite*, `l2reg · Σβ²` must be comparable to the loss:

```
healthy basin   Σβ² ≈ 400     →  l2reg ≈ 0.66/400    ≈ 1.6e-3   begins to matter
exploded basin  Σβ² ≈ 20,000  →  l2reg ≈ 0.66/20000  ≈ 3.3e-5   begins to matter
catastrophic    Σβ² ≈ 75,000  →  l2reg ≈ 0.66/75000  ≈ 8.8e-6
```

The transition zone is `l2reg ∈ [~1e-5, ~2e-3]`. The Stage A grid
`[0, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]` brackets it: `3e-5` should barely touch the
healthy basin while crushing the exploded one; `3e-3` should start shrinking
even the healthy basin. That lets us trace the full L2 response curve and locate
the **knee** — the largest `l2reg` that kills the exploded basin without
distorting the healthy one.

---

## Architecture (decided)

- **Standalone parallel script** `convergence_lab_sweep.py` fits the grid with
  `fit_models(params, n_processes=N)` and writes one results pickle to
  `results/convergence_lab/sweep_<tag>.pkl`. The marimo notebook
  (`convergence_lab.py`) and the dashboard both **load** that pickle — no
  fitting happens in marimo (it is capped at `n_processes=1` because it cannot
  `spawn` workers safely). This mirrors the existing dashboard pattern.
- **Per-fit basin metrics** are extracted alongside the existing
  `converged` / `repl_corr`:
  - `max_abs_phi` — max |latent phenotype|, catches the explosion (≈300–4751).
  - `alpha_final` — catches the small-α basin (degenerate ≈ 1.5; healthy ≈ 3–8).
  - `beta_l2_norm` — `Σβ²`, the direct shrinkage measure the l2reg grid targets.

Extraction reference (validated against the cache): `α` via
`model._jax_model.α`; per-condition β via `model._jax_model.φ[cond].β`;
`beta_l2_norm = Σ_cond Σ_i β_cond[i]²`. The pickle is regenerated on demand and
**not committed** (the 24-fit cache was ~580 MB).

---

## Stage A — the β-explosion fix (data-anchored l2reg)

**40 fits** (trimmed for a fast local loop). `recompute_scale=False` and
`share_alpha=True` **locked** (proven / degeneracy-prone respectively).

| Axis | Levels |
|---|---|
| `l2reg` | `[0, 1e-4, 3e-4, 6e-4, 1e-3]` (knee-refocused — see timing probe below) |
| `warmstart` | `[False, True]` (does L2 *substitute* for warmstart?) |
| `fusionreg` | `[0, 8e-5]` (`8e-5` was the first ablation's worst regression) |
| `replicate` | `[rep_1, rep_2]` |

`scale_fusion_by_n` (the Delta-floor lever) and the `3.2e-4` fusionreg level are
**deferred to Stage B**, where they run against the stabilized winning basin — a
better-controlled read than crossing them into the L2 response curve here.

**Iteration caps (fast local loop):** `block_iters=25`, inner `maxiter=20`
(reduced from 50/50). A sweep needs to *locate the knee*, not produce
production-quality fits. See the timing probe.

### Timing & knee probe (single-fit, before the full run)

Measured per-fit cost and basin diagnostics to size the grid and the iter caps:

| block × inner | l2reg | time | α | Σβ² | final_obj_err |
|---|---|---|---|---|---|
| 50 × 50 | 0 | 86.6 s | 6.82 | 53,349 | conv |
| 50 × 50 | 1e-3 | 141.6 s | 115.2 | 99 | conv |
| **25 × 20** | 0 | **37.0 s** | 7.20 | 44,118 | 1.8e-4 |
| **25 × 20** | 3e-4 | **40.3 s** | 39.4 | **439** | 4.6e-3 |
| 15 × 10 | 3e-4 | 18.5 s | 21.6 | 876 | 2.0e-2 ✗ |

Two findings already locked in by this probe:

1. **`25 × 20` is the sweep sweet spot** — ~38 s/fit (2.3× faster) with α and
   Σβ² stable vs the full 50×50 fit, so basin diagnostics are trustworthy.
   `15 × 10` is rejected: `final_obj_err ≈ 2e-2` (not converged), params drift.
   → 40 fits @ `n_processes=4` ≈ **6–8 min**.

2. **The l2reg knee is near `3e-4`, and degeneracy is double-ended:**

   ```
   l2reg:   0  ─────────── 3e-4 ─────────── 1e-3
   Σβ²:   53,349           439              99
   α:       6.8             39              115
          β-EXPLODE       ⭐KNEE         β-COLLAPSE / α-EXPLODE
   ```

   `l2reg=0` lands in the β-explosion basin; `l2reg=1e-3` over-shrinks β to ~0
   and the see-saw runs the *other* way (α explodes to ~115). The useful basin
   is the **middle**, which is why the grid was refocused to
   `[0, 1e-4, 3e-4, 6e-4, 1e-3]` to bracket the knee tightly. **The "healthy
   basin" is bounded on both sides** — a correction to the original framing,
   which only named the explosion end.

Inner solver settings held at the notebook's current values
(`ge_kwargs / cal_kwargs = dict(tol=1e-4, maxiter=50, maxls=40, jit=True)`,
`block_iters=50`, `block_tol=1e-6`).

### Decisive question (Stage A)

> Is there an `l2reg` that holds the **healthy basin** (Σβ² ≈ 400, α ≈ 3–8) AND
> makes warmstart unnecessary — delivering clean convergence **and** replicate
> correlation in the **same** arm? No arm in the first factorial achieved both.

Secondary reads:
- Does `scale_fusion_by_n=True` lift the Delta floor / Delta's shift correlation
  specifically (vs. the equal-weight default)?
- Does the L2 knee coincide with the `repl_corr` maximum (i.e. is the
  reproducibility win *caused* by killing the explosion)?

### Results (Stage A)

**First run (40 fits, block=25/inner=20, n_processes=4) had to be killed:** two
of four workers ran **>72 CPU-minutes** on single cells without returning, vs the
~40 s/fit the single-fit probe predicted. `fit_models` pickles only after *all*
fits return, so the run produced no output.

**First diagnosis was WRONG and is retracted.** I attributed the runaway to
`warmstart=False + l2reg>0` being a "pathological optimization." An isolated
single-fit probe (correct `fit_one_model` API, `n_processes=1`) **refutes** this:

| cell | time | α | Σβ² | iters | final_err |
|---|---|---|---|---|---|
| `warm=False, l2=0`   | 35.3 s | 3.32 | 343,468 | 25 | 2.0e-4 |
| `warm=False, l2=3e-4`| **39.9 s** | 40.1 | **395** | 25 | 4.5e-3 |

The suspect cell completes in **40 seconds**, not 72 minutes. So the cell is fine;
the runaway is a **parallelism artifact**, not a property of the optimization.

**Real diagnosis — JAX/XLA thread oversubscription under `n_processes>1`:** each
`fit_models` worker runs a JAX process that, by default, spins XLA's CPU
threadpool across *all* cores. Run 4 workers at once → 4 × N_cores threads
contending for N cores → catastrophic context-switching (workers pinned at ~100%
CPU while wall-clock balloons and no fit finishes). The single-process probe
never hit this.

**Fix (baked into `convergence_lab_sweep.py`, not a shell flag):** the script
sets `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`, `OMP_NUM_THREADS=1`,
`OPENBLAS/MKL_NUM_THREADS=1` **at the top of the module, before `import multidms`**
(XLA reads these only at init, and multidms imports JAX at load). One thread per
worker → workers × 1 = cores, no oversubscription. A `--stage smoke` (8-cell)
target validates the threaded parallel path before the full 40-fit run.

> Machine has **14 physical cores**, so at `n_processes=4` even unpinned
> oversubscription is "only" ~4× — not enough to explain a 100× slowdown alone.

**UPDATE — thread-pinning did NOT fix it (H1 refuted), and the real fix is to
drop `fit_models` entirely.** The pinned `n_processes=4` smoke run still timed
out. Isolating tests then showed the hang is **not** about parallelism:

```
40 s   →  fit_one_model(...)              direct call            ✓
hang   →  fit_models(n_processes=4)       spawn pipeline         ✗
hang   →  fit_models(n_processes=1)       STILL spawn pipeline   ✗  ← decisive
```

`fit_models` routes **even `n_processes=1`** through a `multiprocessing`-spawn
worker (`_fit_fun` → `fit_one_model`); that spawn path hangs on this grid's
`l2reg > 0` cells. The first 24-fit factorial never hit it because it ran
`l2reg=[0.0]` only. The exact root cause inside the spawn worker was not chased
further — **parallelism is not needed here**, so the sweep was rewired to fit
**sequentially via a plain `fit_one_model` loop** (the path that completes in
~40 s/fit), rebuilding the same DataFrame with `stack_fit_models` so downstream
is unchanged. `--n-processes` was removed; the thread-pinning env vars are kept
(harmless, and they stop a single fit from grabbing all 14 cores).

> Net: no grid change, no parallelism. ~40 s/fit sequential → Stage A (40 fits)
> ≈ **25–30 min** locally. Slower than the hoped-for "few minutes," but it
> actually runs.

> **Net so far:** the `warmstart=False × L2` cells are valid and informative
> (`l2=3e-4` already shows the knee: Σβ² 343k→395, α 3.3→40), and the decisive
> question is still open — pending a *correctly-threaded* parallel re-run, not a
> grid change. No cells need dropping.

_Full L2 response curve + repl_corr recorded after the threaded re-run._

---

## Stage B — harmony tuning (runs only after Stage A is read)

Locks Stage A's winning `(l2reg, scale_fusion_by_n)` and tests the user's two
remaining hypotheses on a **stabilized** basin:

| Axis | Levels | Hypothesis under test |
|---|---|---|
| inner `maxiter` (`ge_kwargs`/`cal_kwargs`) | `[5, 10, 25]` | lower inner iters → blocks alternate "in more harmony" → fewer objective increases |
| count-warmstart | `[off, on]` | `include_counts=True` enables the *already-implemented* `log(pre_counts)`-weighted Ridge seed (`jaxmodels.Latent.warmstart`); currently silently OFF because the notebook builds `Data` without counts |
| `scale_fusion_by_n` | `[False, True]` | deferred from Stage A; targets the Delta-floor (Delta = least data) on the stabilized basin |
| `fusionreg` | `[0, 8e-5, 3.2e-4]` | restore the full prod span once a basin is fixed |

> Count-weighting only changes **which basin warmstart seeds**; the main Huber
> loss stays unweighted by design (conditions contribute equally). So it can
> move the starting point but not the objective.

### Decisive question (Stage B)

> Among configs that already converge + reproduce (from Stage A), does lowering
> inner `maxiter` reduce objective-error oscillation and wall-clock without
> sacrificing the replicate correlation? And does count-weighted warmstart seed
> a better/faster basin than the unweighted seed?

### Results (Stage B) — TO BE FILLED IN

_Pending Stage A._

---

## Deferred / not in scope yet

- **`share_alpha=False`** — a *small confirmatory* check only, run last. It adds
  α-freedom exactly where the degeneracy lives, so it is a diagnostic ("does
  per-condition α make the see-saw worse?"), not a candidate fix.
- A full crossing of every knob (~2300 fits) — rejected: most cells
  uninformative given what the staging already establishes.

---

## Reproducing

```bash
cd experiments/scv2-spike
# Fit Stage A in parallel (writes results/convergence_lab/sweep_stageA.pkl):
JAX_PLATFORM_NAME=cpu pixi run python notebooks/convergence_lab_sweep.py --stage A
# Then load + render in the notebook (no fitting in marimo):
JAX_PLATFORM_NAME=cpu pixi run marimo edit notebooks/convergence_lab.py
```

(Exact CLI finalized when the script is written.)
