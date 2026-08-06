"""Render the #291 before/after shift-sparsity comparison table.

Phase 1 of EPIC #290 raised the simulation pipeline's inner block-solver
cap (``ge_kwargs``/``cal_kwargs`` ``maxiter``) from 10 to 100. This script
compares shift sparsity across the full ``fusionreg`` ladder and all six
simulated datasets, before and after that change.

Both source runs live under ``experiments/simulation/results*/``, which is
gitignored (root ``.gitignore``: ``experiments/*/results*``). The two
``fit_sparsity.csv`` files are therefore committed alongside this script so
the table is reproducible from a fresh clone:

    291-fit_sparsity-before.csv  inner maxiter = 10
        (from results-prod-287-config-tier-split)
    291-fit_sparsity-after.csv   inner maxiter = 100
        (from results-prod-291-sim-inner-maxiter)

Sparsity is ``(x == 0).mean()`` over the shift coefficients — exact zeros,
no tolerance (``multidms/model_collection.py``).

Usage:
    pixi run python experiments/simulation/diagnostics/291_sparsity_table.py \
        > experiments/simulation/diagnostics/291-sparsity-before-after.md
"""

import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).parent
BEFORE_CSV = HERE / "291-fit_sparsity-before.csv"
AFTER_CSV = HERE / "291-fit_sparsity-after.csv"

# Ground-truth shift sparsity of the simulation: 10 non-identical sites, of
# which 6 are shifted, so the true shift vector is mostly exact zeros.
GROUND_TRUTH = 0.81

# The exit criterion from issue #291: at these two fusionreg values, every
# observed_phenotype library must gain at least this much sparsity.
GATE_FUSIONREG = [3.2e-4, 6.4e-4]
GATE_MIN_DELTA = 0.20

KEY = ["dataset_name", "fusionreg", "mut_type"]


def load(path, label):
    """Read one fit_sparsity.csv, keeping the shift_h2 rows."""
    df = pd.read_csv(path)
    df = df[df["mut_param"] == "shift_h2"]
    return df[KEY + ["sparsity", "measurement_type"]].rename(
        columns={"sparsity": label}
    )


def build():
    """Join the two runs on (dataset, fusionreg, mut_type) and add the delta."""
    before = load(BEFORE_CSV, "before")
    after = load(AFTER_CSV, "after").drop(columns="measurement_type")
    merged = before.merge(after, on=KEY, how="outer", validate="one_to_one")
    merged["delta"] = merged["after"] - merged["before"]
    return merged.sort_values(["mut_type", "dataset_name", "fusionreg"])


def fmt_ladder(df, mut_type):
    """Render one mut_type as a fusionreg x dataset table of before -> after."""
    sub = df[df["mut_type"] == mut_type]
    datasets = sorted(sub["dataset_name"].unique())
    lines = [
        "| fusionreg | "
        + " | ".join(d.replace("_func_score", "") for d in datasets)
        + " |",
        "|---" * (len(datasets) + 1) + "|",
    ]
    for reg in sorted(sub["fusionreg"].unique()):
        row = [f"{reg:.1e}"]
        for ds in datasets:
            cell = sub[(sub["fusionreg"] == reg) & (sub["dataset_name"] == ds)]
            if cell.empty:
                row.append("—")
                continue
            b, a, d = cell.iloc[0][["before", "after", "delta"]]
            row.append(f"{b:.4f} → {a:.4f} ({d:+.3f})")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def gate_report(df):
    """Check the issue #291 exit criterion and report pass/fail per library."""
    sub = df[
        (df["mut_type"] == "nonsynonymous")
        & (df["measurement_type"] == "observed_phenotype")
        & (df["fusionreg"].round(10).isin([round(r, 10) for r in GATE_FUSIONREG]))
    ].sort_values(["fusionreg", "dataset_name"])
    lines = [
        "| fusionreg | dataset | before | after | delta | ≥ +0.20 |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in sub.iterrows():
        ok = "✓" if r["delta"] >= GATE_MIN_DELTA else "✗"
        lines.append(
            f"| {r['fusionreg']:.1e} | {r['dataset_name'].replace('_func_score', '')} "
            f"| {r['before']:.4f} | {r['after']:.4f} | {r['delta']:+.3f} | {ok} |"
        )
    passed = bool((sub["delta"] >= GATE_MIN_DELTA).all()) and not sub.empty
    return "\n".join(lines), passed


def monotonicity_report(df):
    """Report whether the after-curve rises with fusionreg, per dataset."""
    sub = df[df["mut_type"] == "nonsynonymous"]
    lines = [
        "| dataset | monotone non-decreasing in fusionreg? | first drop |",
        "|---|---|---|",
    ]
    for ds in sorted(sub["dataset_name"].unique()):
        curve = (
            sub[sub["dataset_name"] == ds]
            .sort_values("fusionreg")
            .reset_index(drop=True)
        )
        # Step-to-step change along the fusionreg ladder in the AFTER run.
        # A negative step means the repaired curve turns over there.
        steps = curve["after"].diff()
        drop_positions = [i for i, s in enumerate(steps) if s < 0]
        if not drop_positions:
            lines.append(f"| {ds.replace('_func_score', '')} | ✓ yes | — |")
        else:
            pos = drop_positions[0]
            prev = curve.iloc[pos - 1]
            here = curve.iloc[pos]
            span = (
                f"{prev['fusionreg']:.1e}→{here['fusionreg']:.1e}: "
                f"{prev['after']:.4f}→{here['after']:.4f}"
            )
            lines.append(f"| {ds.replace('_func_score', '')} | ✗ no | {span} |")
    return "\n".join(lines)


def main():
    """Print the full markdown report to stdout."""
    df = build()
    gate, passed = gate_report(df)

    print("# #291 shift sparsity — inner maxiter 10 → 100")
    print()
    print(
        "Shift (`shift_h2`) sparsity across the full `fusionreg` ladder and all "
        "six simulated datasets. Sparsity is the fraction of shift coefficients "
        "that are exactly zero. Generated by `291_sparsity_table.py` from the "
        "two committed `fit_sparsity.csv` snapshots."
    )
    print()
    print(f"Ground-truth shift sparsity of the simulation: **{GROUND_TRUTH}**.")
    print()
    print("## Exit criterion (issue #291)")
    print()
    print(
        "`nonsynonymous` / `observed_phenotype` must gain **≥ +0.20** at "
        "`fusionreg` 3.2e-4 and 6.4e-4:"
    )
    print()
    print(gate)
    print()
    print(f"**Criterion {'MET' if passed else 'NOT MET'}.**")
    print()
    print("## Monotonicity of the repaired ladder (target, not a gate)")
    print()
    print(monotonicity_report(df))
    print()
    for mut_type in ["nonsynonymous", "stop"]:
        print(f"## Full ladder — `{mut_type}`")
        print()
        print("Cells are `before → after (delta)`.")
        print()
        print(fmt_ladder(df, mut_type))
        print()


if __name__ == "__main__":
    main()
