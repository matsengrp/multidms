# Convergence lab findings (#246)

**Status:** complete · **Date:** 2026-06-22 · **Data:**
`results-prod-235-times-seen-threshold` (full, both replicates, 3 conditions
Delta / Omicron_BA1 / Omicron_BA2, reference = Omicron_BA1, Sigmoid GE)

This note is written to be read **cold**. It reports a 2×2×3×2 = 24-fit
factorial run locally via `convergence_lab.py`, scoring the `recompute_scale`
convergence fix on two metrics: (①) convergence and (②) replicate-shift
correlation — the metric that actually decides whether the model recovers true
condition-specific shifts.

The factorial cache (`results/convergence_lab/fit_collection.pkl`, ~580 MB) is
**not** committed — it is regenerated on demand by the notebook's
`if CACHE.exists()` guard. The numbers below are the committed record.

---

## The factorial

| Axis | Levels |
|------|--------|
| `recompute_scale` | `True` (mainline, recompute scale each sweep) vs `False` (fixed-scale fix) |
| `warmstart` | `True` (per-condition Ridge seed) vs `False` (β=0 start) |
| `fusionreg` | `0`, `8e-5`, `3.2e-4` (spans the prod grid; `8e-5` was the ablation's worst regression) |
| `replicate` | `rep_1`, `rep_2` |

`block_iters=50`, `block_tol=1e-6`, `l2reg=0`, `beta0_ridge=0`,
`share_alpha=True`. Fit serially (`n_processes=1`) — `fit_models` spawns
workers via `get_context("spawn")`, which a marimo notebook cannot use safely.

---

## Metric ① — convergence

`converged = final objective_error < 1e-6`. `inc` = number of sweeps where the
(unscaled) objective *increased* (oscillation proxy), per `ablation.py:72–81`.

| recompute_scale | warmstart | converged | iters (range) | obj-increases | α (range) | mean fit time |
|---|---|---|---|---|---|---|
| **False (fixed)** | **False** | **6/6 ✓** | **5–7** | **0–1** | **1.7–3.1** | **9.7 s ⭐ fastest** |
| False (fixed) | True | 2/6 | 21–50 | 0–21 | 3.7–8.2 | 56.5 s |
| True (recompute) | False | 4/6 | 4–50 | 0–2 | 1.4–3.3 | 33.8 s |
| True (recompute) | True | 0/6 ✗ | 50 (cap) | 0–46 | 3.1–7.6 | 93.7 s slowest |

**Findings (①):**

1. **Fixed-scale + no-warmstart converges every cell**, in 5–7 iterations, with
   zero/one objective increases and sane α (1.7–3.1). It is also the **fastest**
   arm (9.7 s/fit) because it does not burn the 50-iteration cap. This
   reproduces the prior ablation
   (`results/ablation-scale-vs-ridge/FINDINGS.md`) on full real data.
2. **Recompute-scale + warmstart is the worst** — 0/6 converged, up to **46**
   objective increases, slowest. This is the mainline production configuration.
3. **Warmstart *hurts* convergence.** Holding the scale axis fixed, turning
   warmstart on drops convergence (fixed: 6/6 → 2/6; recompute: 4/6 → 0/6) and
   raises the oscillation count. This is the **opposite** of the going-in
   hypothesis (that warmstart would be a free convergence speed-up).

The convergence half of the story is clean and safe: **`recompute_scale=False`,
`warmstart=False` is the convergence fix.**

---

## Metric ② — replicate-shift correlation (the science metric)

Pearson correlation of the shift parameters (`shift_Delta`,
`shift_Omicron_BA2`) between `rep_1` and `rep_2`, computed via the canonical
`ModelCollection.mut_param_dataset_correlation(x="fusionreg", r=1)`. **The
prior version of this analysis used `times_seen_threshold=0`; this is wrong** —
every canonical artifact (`results/evaluate.ipynb`, the dashboard,
`results/library_replicate_correlation.csv`) uses `times_seen_threshold=1`.
All numbers below use `times_seen_threshold=1` unless a sweep is shown.

### At fusionreg = 8e-5 (times_seen sweep)

| arm | ts=1 Delta / BA2 | ts=5 Delta / BA2 | converged? |
|---|---|---|---|
| fixed + **no-warmstart** | **0.04 / 0.07** | **0.03 / 0.07** | ✓ (6/6) |
| fixed + warmstart | 0.61 / 0.65 | 0.74 / 0.75 | ✗ (50-iter cap) |
| recompute + no-warmstart | 0.28 / 0.79 | 0.28 / 0.80 | ✓ |
| recompute + warmstart | 0.56 / 0.64 | 0.67 / 0.74 | ✗ (50-iter cap) |
| **prod baseline** (evaluate.ipynb) | 0.60 / 0.67 | — | — |

**Findings (②):**

1. **The arm that converges cleanly (fixed + no-warmstart) produces shift
   estimates that do not reproduce across replicates** — Pearson ≈ 0.04–0.07 at
   fr=8e-5.
2. **This irreproducibility is not a rare-mutation-noise artifact.** It is
   **flat across times_seen thresholds** (0.039 at ts=1 → 0.034 at ts=5). The
   times_seen filter is a pure row-mask on which mutations enter the
   correlation (`model.py:364–371`); it does not change shift values. Flatness
   means restricting to *better-observed* mutations does not improve agreement —
   so the disagreement is broad-spectrum, not driven by rarely-seen mutations.
3. **It is not a trivial α-scale or sign artifact.** Pearson is invariant to
   linear rescaling (`model_collection.py:1424`), so the 15× α gap between the
   fixed+nowarm arm (α≈1.7–3.1) and the warm arms (α≈6–8) cannot by itself
   produce the correlation gap; a global sign flip would give ≈ −0.6, not 0.04.
4. **The arms with good replicate correlation (0.6–0.75) are the warmstart
   arms — but at fr=8e-5 those fits did not converge** (they hit the 50-iter
   cap; see Metric ①). Good replicate correlation is, in this factorial,
   **only observed in non-converged fits.**

> ⚠ **Open alternative (not ruled out):** the small-α basin that fixed+nowarm
> settles into may itself be a less-identified / degenerate optimum (small α →
> the sigmoid runs in a near-linear regime, changing relative β values
> non-linearly — which Pearson does *not* protect against). Whether the
> small-α basin is genuinely worse, or just different, is the most important
> unexplored question this experiment raises.

---

## Verdict against the issue's decisive question

> **The scale fix and warmstart trade off.** Fixed-scale + no-warmstart
> converges cleanly and fast (6/6, 5–7 iters, sane α) but its replicate-shift
> correlation is ~0.04–0.07 — the replicates do not agree. This is not a
> rare-mutation-noise artifact (flat across times_seen) and not a trivial
> α-scale/sign artifact (Pearson is rescale-invariant; α differs 15×). The arms
> that *do* show good replicate correlation (0.6–0.75) are the warmstart arms,
> but at fr=8e-5 those did not converge. **So no single arm in this factorial
> delivers both clean convergence and reproducible shifts.**

This lands on the issue's fork: the scale bug is real and the `recompute_scale`
fix mechanically cures convergence — but fixing convergence *alone* (without
warmstart) produces shift estimates that do not reproduce across replicates.
The convergence fix is necessary but **not sufficient** for the scientific
goal. The next question is identifiability of the shift parameters in the
small-α basin, not the scale normalizer.

### Caveat on the baseline correspondence

The `fixed+warm` / `recompute+warm` arms land near the prod baseline
(0.60/0.67 at fr=8e-5), but this is a **loose** correspondence: the baseline
(`library_replicate_correlation.csv`) was produced over the **full** prod
fusionreg grid, potentially with **continuation-path** fitting (warm-starting
each fr step from the previous), whereas this factorial uses an independent
3-point grid. The match is suggestive, not a controlled validation.

---

## Reproducing

```bash
cd experiments/scv2-spike
# Regenerates the 580 MB cache on first run (~22 min, 24 serial fits on CPU),
# then renders instantly from cache:
JAX_PLATFORM_NAME=cpu pixi run marimo edit notebooks/convergence_lab.py
```

The `recompute_scale` toggle now lives in mainline `jaxmodels.fit()` (default
`True`, preserving current behavior). Cells 3–7 of the notebook compute both
metrics, the decisive correlation-vs-fusionreg plot, a reactive
α / β0 / predicted-floor diagnostic, and the α-bound probe.
