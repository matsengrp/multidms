# convergence-lab

A fast, local harness for diagnosing the scv2-spike convergence and
reproducibility problem (issue #253; built on the `recompute_scale` fix, #246).
The full remote pipeline takes hours per turn; this fits tiny grids locally in
minutes so we can iterate on the two leading suspects (the α/β see-saw and the
L2 knee / fixed-scale interaction).

## How to run

    pixi run python experiments/convergence-lab/harness.py \
        --config grids/smoke.yaml --cache smoke
    # parallel worker count defaults to all-but-one core (capped at grid size);
    # override per machine with --n-processes N

Writes `results/<cache>/fit_collection.pkl` — a *true* fit collection (the raw
`stack_fit_models` frame, the same schema the scv2-spike pipeline writes;
gitignored, regenerated on demand). A new experiment is a new `grids/*.yaml`;
the runner is generic and owns the constant scv2-spike data.

The harness **only fits models** — it computes no derived metrics. Everything
downstream (replicate-shift correlation, basin diagnostics, plots) is computed
on the fly from a `ModelCollection` built over the frame, per experiment:

    import pickle
    from multidms.model_collection import ModelCollection
    mc = ModelCollection(pickle.load(open("results/smoke/fit_collection.pkl", "rb")))
    _, corr = mc.mut_param_dataset_correlation(x="fusionreg", return_data=True, r=1)

**Parallelism:** the harness always fits via `fit_models(n_processes=N)` (the
`spawn` worker pool). This is safe because the fit call lives under the script's
`if __name__ == "__main__":` guard — see the warning on
`multidms.model_collection.fit_models`. Each worker rebuilds its own dataset
copy, so peak RAM grows with `n_processes` × dataset size; size `--n-processes`
to the host's free memory, not just its core count.

**Visual review:** explore the fits in the multidms dashboard — from this
directory run `pixi run dashboard` (it discovers `fit_collection.pkl` below cwd).

## Standing findings

- **α/β see-saw degeneracy** (measured): `warmstart=True` → α 3–8 / Σβ² 350–1400
  (healthy); `warmstart=False` → α 1.4–3.3 / Σβ² 1700–75000 (β exploded so α·φ
  stays bounded and Huber loss barely moves). Two near-degenerate basins fit
  equally well, so noise decides which one each replicate lands in.
- **L2 knee near `3e-4`**; double-ended degeneracy: β explodes at `l2reg=0`,
  β collapses + α explodes at `l2reg ≥ 1e-3`. **Below the knee, L2 is inert
  against an active clip** (measured, #284, `cache=beta0-ridge-l2-scan`):
  `l2reg` at 1e-8/1e-7/1e-6 — three-to-four orders under the knee — moves
  nothing on top of `beta_clip_range=[-10,10]` across a 72-fit grid. Σβ² 0.96–1.01×
  and α 1.00–1.11× vs the `l2reg=0` baseline, replicate-shift r within ±0.007,
  all far under the pre-registered 2× / 0.05 effect thresholds. The clip is the
  binding constraint on β; a whisper of L2 underneath it is a no-op. (The knee
  is where L2 *starts* to bite — this pins the other end of that statement.)
- **A nonzero `beta0_ridge` (1e-4…1e-2) is also inert at this fixed block**
  (measured, #284, same grid — the axis's first sweep in this lab). **What it
  penalizes matters and is easy to get wrong:** `_beta_ridge_penalty`
  (`jaxmodels.py:533-540`) is NOT an intercept ridge — it penalizes each
  non-reference condition's β0 *difference from the reference*,
  `beta0_ridge · Σ_{d≠ref}(β0_d − β0_ref)²`, never the reference's own β0. So it
  is an **L2 shift-shrinkage on the intercepts**, structurally parallel to
  `fusionreg`'s L1 on the β shift — cross the two and read its effect *within*
  each fusionreg level, never marginalized. Its penalty is also **exactly 0 at
  initialization** wherever `beta0_init` pins all conditions to the same value
  (as every grid here does), and only bites as the β0s separate. Effect at
  1e-4…1e-2: α rises **monotonically** (16.1 → 16.5 → 17.5) and shift_Delta r at
  fusionreg=0 lifts 0.594 → 0.617, so the penalty genuinely touches the optimum —
  but every move is 1-2 orders below the effect thresholds. It is too weak to
  matter here, not structurally dead; a stronger axis was not tested.
- **Caveat on both of the above — they were measured where convergence was
  already 0/8.** #284's baseline is 0/8 converged at inner maxiter=10, so that
  grid could not have detected a convergence *gain* from any regularizer. The
  defensible claim is *"regularization cannot rescue convergence at an
  inadequate inner cap"*, NOT *"these regularizers are inert"* in general. #285
  re-runs the same axes at inner maxiter=100 (the only 100%-converging cell
  known) to separate the two.
- **A small `l2reg>0` tames the β-explosion across the whole prod fusion axis**
  (measured, #256, `cache=l2-fusion`): at `l2reg=0` Σβ² sits at ~16–19k for
  *every* `fusionreg` (0 → 6.4e-4); `l2reg=1e-4` collapses it ~25× into the
  healthy 350–840 band and `3e-4` further to ~190–300, both holding at the
  prod-max `fusionreg=6.4e-4`. `1e-4` is the working weight (α ~20–25); `3e-4`
  over-regularizes toward the see-saw's collapse end (α ~30–40). **Strong
  fusion is reproducibility-rescued only once β is penalized:** at `l2reg=0`
  the `6.4e-4` column collapses `shift_Delta` replicate-r to 0.09, but at
  `l2reg=1e-4` strong fusion *lifts* `shift_Omicron_BA2` r to 0.80 — the
  data-poor-condition distortion the path-fitter targets is itself a symptom
  of the unpenalized β-explosion.
- **β-bound choice — settled** (#263, Phase 1): head-to-head at cold-start /
  recompute_scale=false / tol=1e-6, **`beta_clip_range=[-10,10]` wins** the
  tri-criteria race against `l2reg=1e-4`. Clip is the *only* arm that converges
  to tol=1e-6 (6/6, with 4/6 crossings in the epic's 20–50-iter band); l2 never
  reaches tol (sentinel 101 on all 6). Clip keeps α~3 (healthy) while l2's α
  blows up to ~55 (the see-saw's collapse end). Clip's strong-fusion
  shift_Omicron_BA2 replicate-r reaches 0.81 vs l2's 0.66. Note the twist: clip
  wins with a *large* Σβ² (~110k–181k) that is harmless because the hard φ-cap
  keeps α·φ bounded — "large Σβ²" only signals an explosion when the fit *also*
  fails to converge (as at l2reg=0 in #256). The epic's downstream phases inherit
  the clip bound.
- **A free (per-condition) α deepens Delta's functional-score floor toward the
  true −3.5** (measured, #273, `cache=273-free-alpha`): sweeping
  `share_alpha` True/False across `fusionreg [0, 4e-5, 1.6e-4, 6.4e-4]` at PR
  270's exact fitting block (`warmstart=false`, `l2reg=0`, `recompute_scale=false`,
  Sigmoid, inner blocks maxiter=10/maxls=10, outer tol=1e-5/maxiter=50). Delta's
  predicted floor = `−α_Delta · sigmoid(φ_wt,Delta)` (the low asymptote of
  `α·(g(φ)−g(φ_wt))` as φ→−∞). At the prod-strength `fusionreg=6.4e-4`, freeing α
  lets Delta take its **own** α (5.2, decoupled from BA1's 17.8 / BA2's 14.4)
  and drops Delta's floor from **−2.16 (shared) → −3.09 (free)** — most of the
  way to the true −3.5, where the shared α is pinned by the other two conditions.
  Counter-intuitively the deeper floor comes with a *smaller* Delta α: freeing α
  lets the Delta latent push φ_wt more positive so `−α·sigmoid(φ_wt)` deepens.
  The effect grows with fusion (floor gap free−shared: −0.15 at fusionreg=0,
  −0.93 at 6.4e-4). BA1/BA2 floors barely move. All 16 fits hit maxiter=50
  (0/16 converged, same as PR 270's 0/18 at tol=1e-6 — the looser tol=1e-5
  didn't buy convergence), but obj_err is tiny (≤3e-4), so the α/floor numbers
  are trustworthy while the binary flag is not.
- **Softplus floor × free-α is a complementary 2×2 — only BOTH together floor the
  prod tail to the biological −3.5** (measured, #277, full in-harness 2×2:
  `cache=277-softplus-floor{,-off}` share_alpha=true, `…-freealpha` /
  `…-off-freealpha` share_alpha=false; `output_floor=-3.5` hinge 0.1 vs null;
  4 fusion × 2 reps each = 32 fits, 0 failed; Delta floor computed two ways —
  analytic `t(−α·g(φ_wt))` and model composition at φ=−1e4 — agreed to 0.0e0 on
  all 32). Delta floor (mean over 2 reps):
  ```
  fusionreg    free-OFF  free-ON   shared-OFF  shared-ON
  0.0           -5.24    -3.50      -5.09       -3.50
  4e-5          -4.82    -3.50      -4.69       -3.50
  1.6e-4        -3.45    -3.50      -3.22       -3.50
  6.4e-4(prod)  -3.09    -3.50      -2.16       -2.95
  ```
  The **softplus is a blunt clip**: it hard-truncates the *pre-activation* floor to
  −3.5 wherever that raw floor overshoots the hinge. At weak fusion (0…1.6e-4) both
  α regimes drive Delta's pre-activation floor to −11…−16 (far past −3.5), so ON
  pins every one of those cells to exactly −3.5. The floor is inside `predict_score`
  (which the loss uses), so turning it on also perturbs the fit everywhere — fitted
  α lands consistently higher ON than OFF in both regimes. **Free-α** does the
  opposite thing: it *deepens the pre-activation floor selectively* by letting
  Delta's own α decouple (Delta α at prod: shared 13.47 → free **5.23** OFF), which
  by itself takes the OFF prod floor −2.16 → −3.09. **The prod cell is the whole
  point:** softplus-alone gets −2.16 → −2.95 (α-driven, raw floor only −2.96 so the
  hinge barely bites); free-α-alone gets −2.16 → −3.09; **both together** are the
  *only* arm that reaches **−3.50** at prod, because free-α pushes the
  pre-activation floor (−4.80) past the hinge, which then clamps it. The two levers
  attack opposite regimes and compose at prod — neither alone floors the prod tail;
  together they do. Ships **default-off** (`output_floor=None`); the shared-α OFF
  control reproduced #274's separately-fit column to the last decimal, confirming
  determinism and a fair baseline.
- **`recompute_scale=False`** (the fixed-scale objective normalizer) converges.
- **Inner optimizer maxiter is the dominant convergence lever; `recompute_scale=False`
  + inner maxiter=100 is the only 100%-converging cell** (measured, 2026-07-10,
  `cache=maxiter-scan`, 48 fits = ge/cal `maxiter` {1,10,100} × `recompute_scale`
  {False,True} × fusionreg [0, 4e-5, 1.6e-4, 6.4e-4] × 2 reps, else #277's
  softplus-off block verbatim: cold-start, l2reg=0, Sigmoid, α_init=6,
  clip[-10,10], outer maxiter=50/tol=1e-5). Convergence rises monotonically with
  inner maxiter and is 0 at maxiter=1: **maxiter=1 → 0/16, maxiter=10 → 3/16,
  maxiter=100 → 12/16.** At maxiter=1 the optimizer barely moves (α≈init 6.1,
  Σβ²≈0.4 — essentially unfit); at maxiter=100 the converged fits land in the
  large-Σβ²/low-α clip basin the #263 finding calls healthy (α~4-6, Σβ²~60k–124k
  harmless under the φ-clip). **`recompute_scale=False` dominates `True` at
  maxiter=100 on both axes: 8/8 vs 4/8 converged AND 3.6× faster (mean fit_time
  788s vs 2808s).** So the fixed-scale objective isn't just convergent — it's the
  cheap, reliably-convergent one, and the inner cap must be ≥~100 to actually
  reach tol. NOTE: run serial or a *small* pool for heavy inner-maxiter grids —
  the m100 level deadlocked under the 16-worker spawn pool (collapsed to one
  process); `n_processes=4` ran clean (4 workers ~100% CPU each).
- **`fit_models` parallelism — settled** (`diagnostics/parallelism_probe.py`):
  `n_processes=2` (the real spawn path) ran the full data-size × l2reg staircase
  with **zero hangs**, at `l2reg=0` AND `l2reg=3e-4`. The l2reg-deadlock theory is
  **refuted**. The original local hangs were the **execution context** — marimo
  cells / `/tmp` scripts spawn workers without a clean `if __name__ ==
  "__main__"` guard, so each child re-runs module-level code on import. A
  properly-guarded script runs the identical spawn path cleanly.
- **Dataset duplication is real but harmless** (measured): each spawn worker
  carries its own dataset copy, so `n_processes=2` peaks **~0.9–1.9 GB above**
  the `n_processes=1` in-process baseline, and the gap **grows with data size**
  (+0.9 GB tiny → +1.9 GB full). But peak RSS at the full dataset is ~4.9 GB
  against ~13 GB free — duplication never exhausts RAM. Memory was NOT the cause
  of the hangs. The harness fits in parallel by default; size `--n-processes` to
  the host's free RAM (per-worker dataset copy), not just its core count.

## Run log

(Append one entry per harness run: date, cache name, config swept, what the
fit collection showed — basin diagnostics and replicate correlation computed
downstream from a ModelCollection — the conclusion, the next step.)

- 2026-07-15 | cache=beta0-ridge-l2-scan | **(#284)** sweep: `beta0_ridge` [1e-4, 1e-3, 1e-2] × `l2reg` [1e-8, 1e-7, 1e-6] × fusionreg [0.0, 4e-5, 1.6e-4, 6.4e-4], 2 reps = **72 fits**. Else `softplus-floor-off.yaml`'s fixed block VERBATIM (cold-start warmstart=false, recompute_scale=false, output_floor=null, share_alpha=true, Sigmoid, alpha_init=6, `beta_clip_range=[-10,10]` — the settled #263 bound, INHERITED not swept — `beta0_init` pins all three β0 to 0.0, scale_fusion_by_n=false, loss δ=1, outer maxiter=50/tol=1e-5, inner ge/cal maxiter=10/maxls=10/tol=1e-4), so the grid is a one-knob-at-a-time delta from its baseline (asserted in `diagnostics/test_beta0_ridge_l2_grid.py`, not eyeballed). Config: `grids/beta0-ridge-l2-scan.yaml`. Ran REMOTE on orca04 (64-core, 1.5 TB RAM, scouted idle at 0.3% CPU; GitHub reachable this time, so a real git clone at branch HEAD — no rsync fallback needed) at **n_processes=16**, wall **3320.1s (55m), 72 fit / 0 failed** (39,950s CPU-time in 3,320s wall = 12× speedup; the m100 16-worker deadlock does NOT reproduce at this grid's light inner maxiter=10 block). Downstream from `diagnostics/beta0_ridge_l2_report.py --cache beta0-ridge-l2-scan --baseline-cache 277-softplus-floor-off`.
  **What `beta0_ridge` actually penalizes — NOT the intercepts.** `_beta_ridge_penalty` (jaxmodels.py:533-540) penalizes each non-reference condition's β0 *difference from the reference*, `beta0_ridge · Σ_{d≠ref}(β0_d − β0_ref)²`; the reference's own β0 is never touched. It is an **L2 shift-shrinkage on the intercepts**, structurally parallel to `fusionreg`'s **L1 shift penalty on the β** (`fusionreg · Σ|β_d − β_ref|`, jaxmodels.py:568-577) — hence crossing the two, and hence reading the effect WITHIN each fusionreg level rather than marginalized over it. Note the inherited `beta0_init` starts all three β0 identical at 0.0, so the penalty is **exactly 0 at initialization** and only bites as the β0s separate.
  **Baseline computed here, not quoted — it existed nowhere.** The 277-softplus-floor-off entry says "pkl carries no converged/final_obj_err column", and the 0/16 it cites is #273's *different* free-alpha grid. Computed from the 8-fit pickle with the same report: **0/8 converged, median final_obj_err 2.29e-4, α 16.1, Σβ² 4191**. Baseline replicate-shift r — shift_Delta 0.594/0.571/0.604/0.498, shift_Omicron_BA2 0.382/0.482/0.619/0.767 across fusionreg 0/4e-5/1.6e-4/6.4e-4.
  Convergence + basin per (beta0_ridge, l2reg), **n=8 per cell** (4 fusionreg × 2 reps); the baseline row is n=8 for its whole cache:
    | beta0_ridge | l2reg | converged | median obj_err | α | Σβ² |
    |---|---|---|---|---|---|
    | *(baseline 0)* | *0* | *0/8* | *2.29e-4* | *16.1* | *4191* |
    | 1e-4 | 1e-8 | 0/8 | 2.20e-4 | 16.1 | 4190 |
    | 1e-4 | 1e-7 | 0/8 | 2.15e-4 | 16.1 | 4180 |
    | 1e-4 | 1e-6 | 0/8 | 1.74e-4 | 16.3 | 4020 |
    | 1e-3 | 1e-8 | 0/8 | 2.19e-4 | 16.5 | 4220 |
    | 1e-3 | 1e-7 | 0/8 | 2.13e-4 | 16.5 | 4210 |
    | 1e-3 | 1e-6 | 0/8 | 1.76e-4 | 16.7 | 4060 |
    | 1e-2 | 1e-8 | 0/8 | 2.16e-4 | 17.5 | 4220 |
    | 1e-2 | 1e-7 | 0/8 | 2.23e-4 | 17.6 | 4200 |
    | 1e-2 | 1e-6 | 0/8 | 1.96e-4 | 17.8 | 4060 |
  Replicate-shift Pearson r (rep_1 vs rep_2, `shift_*` rows only — there is no shift_Omicron_BA1, the reference carries no shift parameters), per cell across fusionreg. **Computed by slicing `mut_param_dataset_correlation` with `query=` per cell:** the bare call groups by (dataset_name, fusionreg) and mean-collapses everything else (model_collection.py:1444/787), which on this 72-fit frame would average the 9 (beta0_ridge, l2reg) fits per group — averaging away the two axes under test. Every cell reproduces the baseline column to ±0.02:
    | beta0_ridge | mut_param | fr=0 | fr=4e-5 | fr=1.6e-4 | fr=6.4e-4 |
    |---|---|---|---|---|---|
    | 1e-4 | shift_Delta | 0.595 | 0.571 | 0.604 | 0.498 |
    | 1e-3 | shift_Delta | 0.600 | 0.577 | 0.606 | 0.493 |
    | 1e-2 | shift_Delta | 0.617 | 0.589 | 0.606 | 0.486 |
    | 1e-4 | shift_Omicron_BA2 | 0.382 | 0.482 | 0.619 | 0.767 |
    | 1e-3 | shift_Omicron_BA2 | 0.383 | 0.484 | 0.619 | 0.768 |
    | 1e-2 | shift_Omicron_BA2 | 0.393 | 0.487 | 0.619 | 0.769 |
    (l2reg collapsed above — it moves nothing; the three l2 levels agree to ±0.003 within every beta0_ridge row.)
  **Adjudicated against #284's pre-registered thresholds** — an effect required ANY of: ≥1/8 converged where the baseline is 0/8; median final_obj_err ≥2× baseline; median shift-r moving ≥0.05; or α/Σβ² ≥2× at matched fusionreg. **Result: FLAT on all four, in all 9 cells.** Convergence 0/8 everywhere (primary DEGENERATE — 0.0 vs 0.0 discriminates nothing, so the pre-registered obj_err fallback adjudicates); obj_err ratios 0.76–0.97× (max deviation 24%, threshold 100%); α 1.00–1.11×, Σβ² 0.96–1.01×; shift-r deltas −0.006…+0.007 (threshold 0.05, so the largest is 7× under).
  Conclusion: **NULL — neither a sub-knee `l2reg` (1e-8…1e-6) nor a nonzero `beta0_ridge` (1e-4…1e-2) buys convergence or reproducibility on top of the settled clip bound.** This CONFIRMS and sharpens the standing L2 finding: the ~3e-4 knee is real, and three-to-four orders below it the penalty is simply inert against an active `beta_clip_range=[-10,10]`, which is already the binding constraint on β. The axes are not *quite* dead, though, and the directions are informative: α rises **monotonically** with beta0_ridge (16.1 → 16.5 → 17.5 at fixed l2reg) and Σβ² dips ~4% at l2reg=1e-6, so both penalties do touch the optimum — they are 1-2 orders of magnitude too weak to move it. **The honest limit of this result:** convergence was 0/8 at the baseline BEFORE the scan began, so this grid could never have detected a convergence *gain* — it can only say the regularizers did not rescue a fit that the inner cap had already pinned. Per the 2026-07-10 maxiter-scan, the inner cap is the dominant convergence lever (0/16 → 3/16 → 12/16 over inner maxiter 1/10/100), and this grid inherits the middle-of-the-road maxiter=10. So the defensible claim is *"regularization cannot rescue convergence at an inadequate inner cap"* — NOT *"these regularizers are inert"* in general.
  Next: **#285** (stub, blocked-by #284) re-runs these exact axes at inner maxiter=100 — the only 100%-converging cell known — which is what separates those two claims. Budget ~11h naive (m100 was 2h31m for 16 fits) and **do not reuse n_processes=16** (m100 deadlocked under it; use 4). A maxiter=100 `(0,0)` baseline does not exist yet and must be fit. Standing findings: the L2-knee finding is sharpened below; no finding is overturned.

- 2026-07-10 | cache=maxiter-scan | sweep: ge/cal_kwargs `maxiter` {1, 10, 100} (COUPLED — ge and cal move together) × `recompute_scale` {False, True} × fusionreg [0.0, 4e-5, 1.6e-4, 6.4e-4], 2 reps = 48 fits. Else #277 softplus-floor-off.yaml fixed block VERBATIM (cold-start warmstart=False, output_floor=null, share_alpha=true, Sigmoid, l2reg=0, alpha_init=6, beta_clip_range=[-10,10], loss δ=1, outer maxiter=50/tol=1e-5, inner maxls=10/tol=1e-4). Configs: grids/maxiter-scan-m{1,10,100}.yaml (three configs because the harness Cartesian-products every sweep axis — ge_kwargs and cal_kwargs listed together would 3×3-cross to 9, not the 3 coupled cells; one config per level then `pd.concat` the three fit_collection.pkl). Ran REMOTE on orca04 (64-core, scouted idle) — source tree rsync'd to HEAD 9121748 (GitHub unreachable from orca04 via the 1Password-gated forwarded agent, so no remote git fetch; the one required data CSV, gitignored, rsync'd explicitly). m1/m10 fit at n_processes=16 (wall 154s / 639s, 16 fit 0 failed each); **m100 deadlocked under the 16-worker spawn pool** (collapsed to a single limping process, ~46 min no output — the documented spawn-under-JAX race, widened by the heavier 100-iter compile) → rerun at **n_processes=4**, which ran clean (4 workers ~100% CPU each), wall 9049s (2h31m), 16 fit 0 failed. Downstream basin/convergence computed inline from the combined frame.
  Convergence rate + basin (α, Σβ² of ref-condition β, mean fit_time) per (ge/cal maxiter × recompute_scale):
    | maxiter | recompute | converged | α | Σβ² | fit_time |
    |---|---|---|---|---|---|
    | 1 | False | 0/8 | 6.13 | 0.4 | 132s |
    | 1 | True | 0/8 | 6.13 | 0.4 | 137s |
    | 10 | False | 0/8 | 15.5 | 4,096 | 550s |
    | 10 | True | 3/8 | 11.9 | 7,641 | 558s |
    | 100 | False | **8/8** | 5.59 | 60,959 | 788s |
    | 100 | True | 4/8 | 4.31 | 124,271 | 2,808s |
  Convergence rises monotonically with inner maxiter (0/16 → 3/16 → 12/16 over 1/10/100). At maxiter=1 the optimizer barely moves off init (α≈6.1=alpha_init, Σβ²≈0.4 — essentially unfit, so its "0/8 not converged" is trivial, not pathological). At maxiter=100 the converged fits sit in the large-Σβ²/low-α clip basin the #263 finding calls HEALTHY (α~4-6, Σβ² 60k–124k held harmless by the [-10,10] φ-clip — "large Σβ²" only signals explosion when the fit ALSO fails to converge).
  Conclusion: the inner optimizer cap is the DOMINANT convergence lever here — it must be ≥~100 to reach the outer tol=1e-5 at all; 1 and 10 are simply too few inner steps. And `recompute_scale=False` (fixed-scale objective) dominates `True` at maxiter=100 on BOTH axes: 8/8 vs 4/8 converged AND 3.6× faster (788s vs 2808s). This sharpens the terse standing "recompute_scale=False converges" into a quantified 2-D result and confirms the fixed-scale objective is the cheap, reliably-convergent one. Standing findings updated. Next: with 100% convergence now reachable (recompute=False, inner maxiter=100), re-run the reproducibility criterion (replicate-shift Pearson r via ModelCollection.mut_param_dataset_correlation) on this converged cell to check the α/β-basin choice is stable replicate-to-replicate now that fits actually reach tol; also record the n_processes=4 (not 16) requirement for heavy-maxiter grids in the harness docs.

- 2026-07-07 | cache=273-free-alpha | sweep: share_alpha [True, False] × fusionreg [0.0, 4e-5, 1.6e-4, 6.4e-4], 2 reps (16 fits), PR 270 fitting block verbatim (warmstart=False, recompute_scale=False, l2reg=0, Sigmoid, alpha_init=6, beta_clip [-10,10], inner ge/cal_kwargs maxiter=10/maxls=10/tol=1e-4) + 3 deltas from PR 270 (outer tol 1e-6→1e-5, 4-point fusionreg, share_alpha swept). Independent fitting (#273). Wall 718s at n_processes=13 (local, Apple M4 Max). Numbers computed inline from a ModelCollection over the frame (dashboard-only exploration; no committed analysis code). Delta floor = −α_Delta·sigmoid(φ_wt,Delta), the low asymptote of α·(g(φ)−g(φ_wt)) as φ→−∞ (analytic and model-GE-at-φ=−1e4 evaluations agreed to 0.0e0).
  Delta α + floor (mean over reps), shared vs free, across fusionreg 0 / 4e-5 / 1.6e-4 / 6.4e-4: SHARED α → α 16.8/17.0/14.8/13.5, floor −5.08/−4.69/−3.22/−2.16. FREE α → Delta's own α 16.8/18.1/12.9/5.2 (decoupled from BA1 17.8 / BA2 14.4 at 6.4e-4), floor −5.24/−4.82/−3.45/−3.09. At the prod-strength fusionreg=6.4e-4 freeing α drops Delta's floor from −2.16 → −3.09 (gap −0.93), most of the way to the true −3.5 that the shared α (pinned by BA1/BA2) cannot reach. Effect grows with fusion (floor gap free−shared −0.15 at fr=0 → −0.93 at fr=6.4e-4); BA1/BA2 floors barely move. Counter-intuitively the deeper floor comes with a *smaller* Delta α — freeing α lets the Delta latent push φ_wt more positive so −α·sigmoid(φ_wt) deepens.
  Convergence: 0/16 converged (all hit maxiter=50), same as PR 270's 0/18 at tol=1e-6 — the looser tol=1e-5 did NOT buy convergence. But obj_err ≤3e-4 (range 5.9e-5–3.0e-4), so the α/floor numbers are trustworthy; the binary flag is not (same maxiter-truncation pattern as the l2-fusion / smoke runs).
  Conclusion: CONFIRMS the PR 270 hypothesis — the single shared α is the reason Delta under-fits its low tail. A per-condition α lets Delta decouple and pull its floor toward the true −3.5, with the effect concentrated at strong fusion (exactly where PR 270 saw the −2.5 shortfall). Standing findings updated (new free-α floor finding). Next: confirm on the full scv2-spike pipeline (share_alpha=false prod run) that the deeper floor holds against held-out data and does not reintroduce the per-condition sigmoid degeneracy the shared-α default guards against; optionally pair with a small l2reg>0 (the β-explosion tamer) since this run sits in the l2reg=0 regime.

- 2026-07-06 | cache=beta-control-clip + beta-control-l2 | Phase 1 (#263), EPIC #262. sweep: arm {clip[-10,10], l2reg=1e-4} × fusionreg [0.0, 4e-5, 6.4e-4], 2 reps (12 fits). Base regime: cold-start (warmstart=false), recompute_scale=false, share_alpha=true, Sigmoid, maxiter=100 (ceiling), tol=1e-6. Independent fitting. Local (M4 Max). Downstream from diagnostics/beta_control_report.py.
  Basin diagnostics (Σβ², α per cell): clip arm → Σβ² 110,597–180,962, α 2.96–3.39, **all 6 converged=True** (final_obj_err 6e-8–9e-7). l2 arm → Σβ² 221–300, α 52.5–58.8, all 6 converged=False (final_obj_err 8e-5 at fusionreg=0 rising to ~2.8e-4 at 6.4e-4). This is the α/β see-saw in the open: clip caps φ so α stays low (~3) and the fit converges cleanly despite a large Σβ²; l2 collapses β and drives α to ~55, and it never reaches tol=1e-6. Large clip-Σβ² is NOT an explosion — α·φ is bounded by the clip and the objective converges (contrast #256's l2reg=0 Σβ²~16–19k which did NOT converge to tol).
  maxiter each needs (outer iters to cross tol=1e-6 / 1e-4): clip → 1e-6 at {fusionreg=0: 60,26 · 4e-5: 30,39 · 6.4e-4: 55,13}, 1e-4 at 6–8 iters everywhere (4/6 of the 1e-6 crossings land in the epic's 20–50 band, the other two at 13 and 60). l2 → 1e-6 NEVER crossed (sentinel 101 on all 6); 1e-4 crossed only at fusionreg=0 (iter 88/89) and never at fusionreg≥4e-5 (101). The clip arm is the ONLY arm that reaches tol=1e-6, and it does so cheaply.
  Replicate-shift Pearson r (shift_* rows, per arm, across fusionreg 0/4e-5/6.4e-4): clip → shift_Delta 0.49/0.46/0.59, shift_Omicron_BA2 0.38/0.56/0.81. l2 → shift_Delta 0.49/0.44/0.49, shift_Omicron_BA2 0.33/0.36/0.66. Strong fusion RESCUES the data-poor shift_Omicron_BA2 under both arms, but clip lifts it higher (0.81 vs 0.66) and also lifts shift_Delta at prod-max fusion (0.59 vs 0.49) — clip dominates the reproducibility criterion at every fusion strength.
  Conclusion (tri-criteria — speed × basin health × rising reproducibility): **clip[-10,10] wins, decisively and on all three axes.** Speed: clip is the only arm that converges to tol=1e-6 (l2 never does), landing 4/6 fits in the 20–50 band. Basin health: clip keeps α~3 (healthy) while l2's α blows up to ~55 (the see-saw's collapse end) — the low Σβ² of l2 is the symptom, not the cure. Reproducibility: clip's strong-fusion shift_Omicron_BA2 r reaches 0.81 vs l2's 0.66. This overturns the gauge argument's tie-break expectation (it predicted clip would win, but on a *healthier Σβ²* — instead clip wins with a *large* Σβ² held harmless by the hard φ-cap). **The epic's downstream phases (#264 recompute_scale, #265 warmstart) inherit `beta_clip_range=[-10,10]` as the β-bound.** Next: Phase 2 / #264 — vary recompute_scale at the clip bound.

- 2026-07-07 | cache=277-softplus-floor (ON) + 277-softplus-floor-off (OFF control) | MATCHED in-harness A/B: same 8-cell grid fit twice — sweep fusionreg [0.0, 4e-5, 1.6e-4, 6.4e-4] × 2 reps, share_alpha=true, warmstart=false, recompute_scale=false, Sigmoid, alpha_init=6.0, beta_clip_range=[-10,10], maxiter=50, tol=1e-5 (#274's free-alpha.yaml fixed block VERBATIM) — once with output_floor=-3.5 (softplus, hinge=0.1), once with output_floor=null, so the floor is the ONLY difference. 16 fits, 0 failed, ~307s + ~326s at n_processes=8 (local, Apple M4 Max). The OFF control reproduces #274's separately-fit shared-α column to the last decimal (floor −5.08/−4.69/−3.22/−2.16, α 16.81/17.01/14.76/13.47), validating both the fit determinism and #274's cross-PR baseline.
  Delta floor + fitted shared α (mean over 2 reps); floor computed two ways (analytic `t(−α·sigmoid(φ_wt))` and model composition at φ=−1e4) — agreed to 0.0e0:
    | fusionreg | OFF floor | ON floor | Δfloor | OFF α | ON α | Δα | ON pre-act | clamped? |
    |---|---|---|---|---|---|---|---|---|
    | 0.0 | −5.08 | −3.50 | +1.58 | 16.81 | 20.61 | +3.80 | −15.74 | yes |
    | 4.0e-5 | −4.69 | −3.50 | +1.19 | 17.01 | 21.34 | +4.32 | −15.75 | yes |
    | 1.6e-4 | −3.22 | −3.50 | −0.28 | 14.76 | 19.24 | +4.48 | −16.43 | yes |
    | 6.4e-4 | −2.16 | −2.94 | −0.79 | 13.47 | 17.36 | +3.90 | −2.96 | no |
  Conclusion: the softplus perturbs the FIT at every fusion strength — fitted shared α is a consistent +3.8…+4.5 higher ON than OFF (+29% at prod), and because the floor is inside predict_score (which the loss uses) this is a real optimization effect, not fit-to-fit noise. Two regimes: at weak fusion (0…1.6e-4) the shared-α sigmoid drives Delta's raw pre-activation floor to −15…−17 (far past the biological −3.5) and the softplus HARD-TRUNCATES it to exactly −3.5 (the assay detection floor); at prod fusion (6.4e-4) the raw floor only reaches −2.96 so the hinge itself clips almost nothing (−2.96→−2.94), but the OFF→ON floor still deepens −2.16→−2.94 driven by the genuinely higher fitted α the floor induced, not by the hinge biting. vs #274 free-α (which DEEPENS the under-shooting prod floor −2.16→−3.09 by selectively decoupling Delta's α): the softplus is a BLUNT clip that floors all three conditions and mainly bites the over-shooting weak-fusion tail — it does NOT pull the prod floor to −3.5 on its own. Complementary, opposite regimes. Convergence: pkl carries no converged/final_obj_err column (computed downstream); #274 saw 0/16 at these tols; all 16 fits completed, floor introduced no failures. Next: for a floored prod recommendation, pair the softplus with free-α or l2reg>0 (softplus alone does not fix the prod under-fit); counts-path floor deferred to a separate stub. [SUPERSEDED 2026-07-08 by the free-α arm below — the "pair with free-α" recommendation is now measured directly, not inferred cross-PR.]

- 2026-07-08 | cache=277-softplus-floor-freealpha (ON) + 277-softplus-floor-off-freealpha (OFF control) | The MISSING free-α arm of the 2×2. Same 8-cell grid as the 2026-07-07 shared-α pair but share_alpha=FALSE (the #274 free-α regime, per-condition α) — fit twice, output_floor=-3.5 vs null. 16 fits, 0 failed; wall 1309s (ON) + 1912s (OFF) at n_processes=1. **Ran serial (n_processes=1) deliberately:** the free-α grids deadlocked under the spawn pool (`n_processes=8`) — all 8 workers went to 0% CPU with a shared multiprocessing pipe_handle after a worker re-spawned, a spawn-under-JAX race the heavier free-α compilation widened; serial fits the identical models with no pool. All 16 verified genuinely free-α (per-condition α dict on the fitted model, not shared scalar). Delta floor computed two ways (analytic + model φ=−1e4) — agreed to 0.0e0 on all evals.
  Full 2×2 Delta floor + Delta's own α (mean over 2 reps):
    | fusionreg | free OFF floor | free ON floor | shared OFF floor | shared ON floor | free OFF α_Δ | free ON α_Δ | shared α (OFF/ON) |
    |---|---|---|---|---|---|---|---|
    | 0.0 | −5.24 | −3.50 | −5.09 | −3.50 | 17.37 | 19.16 | 16.81/20.61 |
    | 4.0e-5 | −4.82 | −3.50 | −4.69 | −3.50 | 18.07 | 21.20 | 17.01/21.34 |
    | 1.6e-4 | −3.45 | −3.50 | −3.22 | −3.50 | 12.85 | 17.54 | 14.76/19.24 |
    | 6.4e-4 | −3.09 | −3.50 | −2.16 | −2.95 | 5.23 | 11.00 | 13.47/17.36 |
  Conclusion: the two levers are complementary and compose AT PROD. Free-α *deepens the pre-activation floor selectively* — Delta's α at prod drops shared-13.47 → free-5.23 (OFF), taking the OFF prod floor −2.16 → −3.09 with no clip. The softplus is a *blunt clip* to −3.5 wherever the raw pre-activation floor overshoots the hinge. At prod: softplus-alone −2.16→−2.95 (raw floor only −2.96, hinge barely bites, α-driven); free-α-alone −2.16→−3.09; **BOTH together −2.16→−3.50** — the only arm reaching the biological floor at prod, because free-α's deeper −4.80 pre-activation floor now exceeds the hinge and gets clamped. At weak fusion (0…1.6e-4) both α regimes overshoot to −11…−16 pre-activation, so ON pins every cell to exactly −3.5 regardless of share_alpha. This directly measures (not infers) the 2026-07-07 "pair softplus with free-α for a floored prod recommendation" next-step: confirmed — neither lever alone floors the prod tail, together they do. Standing finding rewritten to the 2×2. Next: none for this experiment; the softplus × free-α interaction is settled. Counts-path floor still deferred to stub #276.

- 2026-06-30 | cache=l2-fusion | sweep: l2reg [0.0, 1e-4, 3e-4] × fusionreg [0.0, 4e-5, 6.4e-4], 2 reps (18 fits), warmstart=True, recompute_scale=False, share_alpha=True, Sigmoid, maxiter=25. Independent fitting (#256). Wall 1138s at n_processes=4 (local, Apple M4 Max). Downstream numbers from diagnostics/l2_fusion_report.py.
  Basin diagnostics (Σβ², α per cell): at l2reg=0.0 → Σβ² 16,181–19,222, α 5.9–8.2 (β EXPLODED at EVERY fusionreg, incl. 6.4e-4). l2reg=1e-4 → Σβ² 554–841, α 19.5–24.6 (β tamed ~25× into the healthy 350–1400 band, holds across all fusion). l2reg=3e-4 → Σβ² 188–298, α 30.5–40.4 (β tamed further but α climbing toward the see-saw collapse end). All 18 converged=False, but final_obj_err is tiny (4e-6 at l2reg=0 to ~7e-4 at l2reg>0) — maxiter=25 truncation, not a pathology; β-magnitude and r are trustworthy, the binary flag is not.
  Replicate-shift Pearson r (per l2reg slice, across fusionreg 0/4e-5/6.4e-4): l2reg=0 → shift_Delta 0.45/0.50/0.09, shift_Omicron_BA2 0.26/0.43/0.20 (strong fusion COLLAPSES shift_Delta at l2reg=0 — the unpenalized-β regime). l2reg=1e-4 → shift_Delta 0.47/0.45/0.33, shift_Omicron_BA2 0.37/0.43/0.80 (strong fusion now LIFTS shift_Omicron_BA2 to 0.80). l2reg=3e-4 → shift_Delta 0.40/0.37/0.41, shift_Omicron_BA2 0.43/0.49/0.76.
  Conclusion: a small l2reg>0 SOLVES the β-explosion across the entire prod fusion axis; l2reg=1e-4 is the working weight (β healthy, α not yet over-driven). The data-poor-condition distortion under strong fusion is itself a symptom of the unpenalized explosion — once β is penalized, strong fusion rescues rather than wrecks replicate-r (shift_Omicron_BA2 0.20→0.80 at fusionreg=6.4e-4). Standing findings updated (L2-knee finding extended to 2-D). Next: maxiter sweep at l2reg=1e-4 to convert the tiny-but-nonzero final_obj_err into converged=True; optionally a continuation-path comparison along fusionreg at l2reg=1e-4 (separate experiment — independent fitting already suffices to tame β).

- 2026-06-26 | cache=smoke | sweep: l2reg [0.0, 3e-4], warmstart=True, recompute_scale=False, fusionreg=0.0, Sigmoid, maxiter=25.
  df_fits (4 fits, ~185s/fit, wall 740s): at l2reg=0.0 -> alpha 7.4-8.2, Sigma-beta^2 36,785-40,371 (beta EXPLODED); at l2reg=3e-4 -> alpha 36-40, Sigma-beta^2 438-626 (beta collapsed, alpha exploded). All 4 converged=False (final_obj_err ~4-6e-6 at l2reg=0, ~3e-4 at l2reg=3e-4).
  df_corr (replicate-shift Pearson r): shift_Delta r=0.47 (l2reg=0) / 0.33 (3e-4); shift_Omicron_BA2 r=0.37 (l2reg=0) / 0.53 (3e-4).
  Conclusion: harness reproduces the double-ended L2 degeneracy + sub-convergence locally in ~12 min. Even warmstart=True + recompute_scale=False shows beta-explosion at l2reg=0. Next: sweep warmstart True/False at fixed l2reg to isolate the see-saw; widen l2reg ladder around the 3e-4 knee.

- 2026-06-27 | diagnostics/parallelism_probe.py --baseline | two-axis staircase (data-size tiny/small/medium/full × l2reg [0.0, 3e-4]), n_processes=2 spawn vs n_processes=1 in-process, cheap iters (block maxiter=6 / inner 8), per-step peak RSS sampled.
  np=2 (spawn): 8/8 PASS, ZERO hangs. Wall 9-36s/step. Peak RSS climbs with data size: l2reg=0 -> 3690/3798/3956/4907 MB (tiny/small/medium/full); l2reg=3e-4 -> 4012/4225/4510/4687 MB (l2reg>0 costs ~+300-500 MB).
  np=1 (in-process baseline, l2reg=0): tiny/small/medium/full -> 2754/2674/2662/3048 MB. (Baseline truncated after l2reg=0 by an operator kill; the l2reg=0 column is sufficient for the spawn-vs-in-process comparison.)
  Spawn overhead = np2 - np1 at l2reg=0: +936/+1124/+1294/+1859 MB — grows with data size, the signature of per-worker dataset duplication. But 4.9 GB peak << ~13 GB free.
  Conclusion: fit_models parallelism does NOT hang (l2reg-deadlock theory refuted); dataset duplication is real (~1 GB/worker) but never exhausts RAM. The original local hangs were the marimo/`/tmp` no-__main__-guard execution context. Next: none for parallelism — settled.

- 2026-06-29 | cache=smoke | sweep: l2reg [0.0, 3e-4], warmstart=True, recompute_scale=False, fusionreg=0.0, Sigmoid, maxiter=25. Harness now PARALLEL (fit_models, n_processes=4 default).
  Output: results/smoke/fit_collection.pkl — a TRUE fit collection (raw stack_fit_models frame, ModelCollection-loadable, dashboard-discoverable). 4/4 fit, wall 246s (vs ~740s sequential on 2026-06-26 — the 4 fits ran concurrently). No metrics stored in the pickle.
  Downstream (computed on the fly from ModelCollection over the frame, NOT stored): replicate-shift Pearson r reproduces the 2026-06-26 numbers — shift_Delta r=0.47 (l2reg=0) / 0.33 (3e-4); shift_Omicron_BA2 r=0.37 (l2reg=0) / 0.53 (3e-4).
  Conclusion: parallel harness produces an identical-schema fit collection the dashboard reads directly; correlation/basin analysis is now per-experiment downstream work, not baked into the harness output. Next: same science as before — sweep warmstart True/False at fixed l2reg to isolate the see-saw; widen the l2reg ladder around the 3e-4 knee.
