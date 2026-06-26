# convergence-lab

A fast, local harness for diagnosing the scv2-spike convergence and
reproducibility problem (issue #253; built on the `recompute_scale` fix, #246).
The full remote pipeline takes hours per turn; this fits tiny grids locally in
minutes so we can iterate on the two leading suspects (the α/β see-saw and the
L2 knee / fixed-scale interaction).

## How to run

    pixi run python experiments/convergence-lab/harness.py \
        --config grids/smoke.yaml --cache smoke

Writes `results/<cache>.pkl` — a dict `{"df_fits", "df_corr"}` (gitignored,
regenerated on demand). A new experiment is a new `grids/*.yaml`; the runner is
generic and owns the constant scv2-spike data.

- **df_fits** — one row per fit: the swept kwargs + `replicate`, plus basin
  metrics `alpha_final`, `beta_l2_norm` (Σ_cond Σ_i β²), `max_abs_phi`,
  `final_obj_err`, `converged`, `fit_time`.
- **df_corr** — replicate-shift Pearson correlation (the ultimate validation
  metric) across the primary swept axis, shift params only.

**Visual review:** explore the fits in the multidms dashboard — from this
directory run `pixi run dashboard` (it discovers `*.pkl` below cwd).

## Standing findings

- **α/β see-saw degeneracy** (measured): `warmstart=True` → α 3–8 / Σβ² 350–1400
  (healthy); `warmstart=False` → α 1.4–3.3 / Σβ² 1700–75000 (β exploded so α·φ
  stays bounded and Huber loss barely moves). Two near-degenerate basins fit
  equally well, so noise decides which one each replicate lands in.
- **L2 knee near `3e-4`**; double-ended degeneracy: β explodes at `l2reg=0`,
  β collapses + α explodes at `l2reg ≥ 1e-3`.
- **`recompute_scale=False`** (the fixed-scale objective normalizer) converges.
- **`fit_models` parallelism**: to be settled by `diagnostics/parallelism_probe.py`
  (see its verdict appended here once run). The harness ships sequential regardless.

## Run log

(Append one entry per harness run: date, cache name, config swept, what
df_fits/df_corr showed, the conclusion, the next step.)

- 2026-06-26 | cache=smoke | sweep: l2reg [0.0, 3e-4], warmstart=True, recompute_scale=False, fusionreg=0.0, Sigmoid, maxiter=25.
  df_fits (4 fits, ~185s/fit, wall 740s): at l2reg=0.0 -> alpha 7.4-8.2, Sigma-beta^2 36,785-40,371 (beta EXPLODED); at l2reg=3e-4 -> alpha 36-40, Sigma-beta^2 438-626 (beta collapsed, alpha exploded). All 4 converged=False (final_obj_err ~4-6e-6 at l2reg=0, ~3e-4 at l2reg=3e-4).
  df_corr (replicate-shift Pearson r): shift_Delta r=0.47 (l2reg=0) / 0.33 (3e-4); shift_Omicron_BA2 r=0.37 (l2reg=0) / 0.53 (3e-4).
  Conclusion: harness reproduces the double-ended L2 degeneracy + sub-convergence locally in ~12 min. Even warmstart=True + recompute_scale=False shows beta-explosion at l2reg=0. Next: sweep warmstart True/False at fixed l2reg to isolate the see-saw; widen l2reg ladder around the 3e-4 knee.
