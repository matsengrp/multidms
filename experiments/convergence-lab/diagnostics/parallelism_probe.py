"""Two-axis parallelism + memory probe for ``fit_models`` (#253).

Settles WHY ``fit_models`` hung locally, and whether memory is the cause. The
scv2-spike pipeline runs ``fit_models`` (spawn workers, ``n_processes>1``) from a
papermill notebook and works on servers — but every pipeline config uses
``l2reg: 0.0``, so the ``l2reg>0`` + spawn path was never exercised there. This
walks a data-size x l2reg staircase at ``n_processes=2`` (the real spawn path),
recording BOTH wall-clock and peak resident memory (RSS) per step.

Why memory: the original hypothesis was that parallel fits duplicate the dataset
per worker and exhaust RAM. With ``spawn`` (not ``fork``) each worker re-imports
and rebuilds its own ``multidms.Data`` from the CSV, so datasets ARE duplicated.
The data-size axis (tiny -> full) is therefore also the memory-growth axis; this
probe reads the gauge a wall-clock-only probe could not. A worker OOM-killed by
the OS looks like a hang from the parent (child dies, queue stays empty), so we
distinguish DIED (child exited non-zero, e.g. SIGKILL) from HANG (child still
alive at timeout).

Fits are deliberately cheap (low ``maxiter``): a deadlock or memory blowup
manifests regardless of iteration count, so we do not pay for convergence here.

A proper ``if __name__ == "__main__":`` guard ensures spawn re-imports cleanly
(unlike the earlier marimo/``/tmp`` probes that confounded the diagnosis).

Run::

    PROBE=experiments/convergence-lab/diagnostics/parallelism_probe.py
    pixi run python $PROBE             # np=2 spawn staircase
    pixi run python $PROBE --baseline  # also the np=1 in-process twin

``--baseline`` reruns the same staircase at ``n_processes=1`` (in-process) so a
failing step has a sequential twin for comparison (same fits, no spawn, one
process's memory) — isolating whether a problem is spawn-specific.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import psutil  # noqa: E402

# Reuse the harness's constant data loader and path/REF constants.
sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))
import harness  # noqa: E402

# Generous per-step ceiling: a healthy `full`-size step (2 parallel fits on the
# whole dataset) is minutes even with cheap iters; a true deadlock never returns
# and still trips this. Calibrated after the smoke run measured ~140-228s for a
# single full fit at maxiter=25 (this probe uses far fewer iters).
STEP_TIMEOUT_S = 420
# All four sizes survive — the data-size axis IS the memory-growth axis.
SIZES = [("tiny", 200), ("small", 1000), ("medium", 5000), ("full", None)]
L2_LADDER = [0.0, 3e-4]
# Cheap fits: the probe measures parallel-execution mechanics + memory, not
# convergence. A deadlock/OOM is independent of iteration count.
_INNER = dict(tol=1e-4, maxiter=8, maxls=40, jit=True)
_BLOCK_MAXITER = 6
_RSS_POLL_S = 0.4


def _peak_rss_mb(pid: int, stop_evt: threading.Event, out: dict) -> None:
    """Poll a process subtree's RSS until ``stop_evt`` is set; record peak MB.

    Sums the RSS of ``pid`` and all its descendants (the spawn workers are
    children of the step process), sampling every ``_RSS_POLL_S`` seconds.
    Stores ``out["peak_mb"]`` — the maximum total RSS observed across samples.

    Args:
        pid: The step process PID (its children are the fit workers).
        stop_evt: Set by the parent when the step finishes / is killed.
        out: Mutable dict to receive ``peak_mb``.
    """
    peak = 0.0
    while not stop_evt.is_set():
        try:
            proc = psutil.Process(pid)
            procs = [proc] + proc.children(recursive=True)
            total = 0
            for p in procs:
                try:
                    total += p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            peak = max(peak, total)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        time.sleep(_RSS_POLL_S)
    out["peak_mb"] = round(peak / 1e6, 1)


def _fit_step(size_n, l2reg, n_processes, q):
    """Child entry: fit 2 models in parallel on a data subset; report seconds."""
    import warnings

    warnings.filterwarnings("ignore")
    import multidms
    from multidms.model_collection import fit_models

    rep_data = harness.load_rep_data()
    datasets = list(rep_data.values())[:2]
    if size_n is not None:
        # Subset each Data's variants by rebuilding from a truncated CSV view.
        # Take head(size_n) PER CONDITION (not a blind head of the replicate):
        # the CSV is condition-ordered, so a blind head() would drop the
        # reference condition entirely and multidms.Data would reject it
        # ("reference must be in condition factor levels"). Per-condition heads
        # guarantee every condition — including the reference — is represented.
        import pandas as pd

        raw = pd.read_csv(harness.DATA_CSV).fillna({"aa_substitutions": ""})
        datasets = []
        for rep in sorted(raw["replicate"].unique())[:2]:
            df_rep = (
                raw[raw["replicate"] == rep]
                .groupby("condition", dropna=False, group_keys=False)
                .head(size_n)
            )
            df_agg = (
                df_rep.groupby(["condition", "aa_substitutions"], dropna=False)
                .agg({"func_score": "mean"})
                .reset_index()
            )
            datasets.append(
                multidms.Data(
                    df_agg,
                    alphabet=multidms.AAS_WITHSTOP_WITHGAP,
                    reference=harness.REF,
                    assert_site_integrity=False,
                    name=f"rep_{rep}",
                    verbose=False,
                )
            )

    params = {
        "dataset": datasets,
        "l2reg": [l2reg],
        "warmstart": [True],
        "recompute_scale": [False],
        "share_alpha": [True],
        "fusionreg": [0.0],
        "ge_type": ["Sigmoid"],
        "maxiter": [_BLOCK_MAXITER],
        "tol": [1e-6],
        "ge_kwargs": [dict(_INNER)],
        "cal_kwargs": [dict(_INNER)],
    }
    t0 = time.time()
    fit_models(params, n_processes=n_processes)
    q.put(time.time() - t0)


def run_staircase(n_processes: int) -> None:
    """Walk the size x l2reg staircase, recording time + peak RSS per step.

    Stops at the first step that does not return cleanly, distinguishing DIED
    (the step process exited non-zero — e.g. an OOM SIGKILL) from HANG (still
    alive at the timeout — a true deadlock). Peak RSS is sampled for every step,
    including failing ones, so a memory blowup is visible even when the step
    is killed.

    Args:
        n_processes: Passed to ``fit_models`` (2 = real spawn path; 1 = in-process).
    """
    ctx = mp.get_context("spawn")
    vm = psutil.virtual_memory()
    print(f"\n=== staircase (n_processes={n_processes}) ===", flush=True)
    print(
        f"  system: total={vm.total / 1e9:.1f}GB available={vm.available / 1e9:.1f}GB",
        flush=True,
    )
    rows = []
    stop = False
    for l2reg in L2_LADDER:
        for size_label, size_n in SIZES:
            q = ctx.Queue()
            p = ctx.Process(target=_fit_step, args=(size_n, l2reg, n_processes, q))
            p.start()

            mem: dict = {"peak_mb": float("nan")}
            stop_evt = threading.Event()
            sampler = threading.Thread(
                target=_peak_rss_mb, args=(p.pid, stop_evt, mem), daemon=True
            )
            sampler.start()

            p.join(STEP_TIMEOUT_S)

            if p.is_alive():
                # Still running at the ceiling -> true deadlock / hang.
                stop_evt.set()
                sampler.join(timeout=2)
                p.terminate()
                p.join()
                peak = mem.get("peak_mb", float("nan"))
                rows.append(
                    (size_label, l2reg, n_processes, "HANG", STEP_TIMEOUT_S, peak)
                )
                print(
                    f"  {size_label:<7} l2reg={l2reg:<7} np={n_processes} -> HANG "
                    f"(>{STEP_TIMEOUT_S}s, peak {peak}MB)  STOPPING",
                    flush=True,
                )
                stop = True
                break

            # Process finished — stop sampling and read the verdict.
            stop_evt.set()
            sampler.join(timeout=2)
            peak = mem.get("peak_mb", float("nan"))

            if p.exitcode != 0 or q.empty():
                # Child exited non-zero (e.g. OOM SIGKILL) or produced no time.
                rows.append((size_label, l2reg, n_processes, "DIED", p.exitcode, peak))
                print(
                    f"  {size_label:<7} l2reg={l2reg:<7} np={n_processes} -> DIED "
                    f"(exitcode={p.exitcode}, peak {peak}MB)  STOPPING",
                    flush=True,
                )
                stop = True
                break

            secs = round(q.get(), 1)
            rows.append((size_label, l2reg, n_processes, "PASS", secs, peak))
            print(
                f"  {size_label:<7} l2reg={l2reg:<7} np={n_processes} -> PASS "
                f"({secs}s, peak {peak}MB)",
                flush=True,
            )
        if stop:
            break

    print("\n  step  data     l2reg    np  result  secs/code  peak_MB")
    for i, (size_label, l2reg, npr, result, val, peak) in enumerate(rows, 1):
        print(
            f"  {i:>4}  {size_label:<7} {l2reg:<7} {npr:>2}  {result:<6}  "
            f"{str(val):<9}  {peak}"
        )


def main() -> None:
    """CLI: default np=2 staircase; ``--baseline`` adds the np=1 twin."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="also run the staircase at n_processes=1 (in-process)",
    )
    args = ap.parse_args()
    run_staircase(n_processes=2)
    if args.baseline:
        run_staircase(n_processes=1)


if __name__ == "__main__":
    main()
