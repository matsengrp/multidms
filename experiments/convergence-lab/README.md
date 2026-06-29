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
  β collapses + α explodes at `l2reg ≥ 1e-3`.
- **`recompute_scale=False`** (the fixed-scale objective normalizer) converges.
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
