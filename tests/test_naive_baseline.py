"""Tests for the naive per-condition baseline arm (EPIC #290 Phase 4).

The naive arm fits each condition independently and derives "shifts" by
subtracting beta vectors. These tests pin the three properties the
comparison against joint fitting depends on:

* the regularizers really are inert for a single condition, so the arm
  needs no lambda ladder (V1);
* naive shifts are exact differences of betas on a shared index (V2);
* the plain mutation-index intersection is safe, because a label can only
  appear in every condition when the wildtype letter agrees (V3).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

SPIKE_NOTEBOOKS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments",
    "scv2-spike",
    "notebooks",
)
sys.path.insert(0, SPIKE_NOTEBOOKS)


@pytest.fixture(scope="module")
def tiny_single_condition_df():
    """A small single-condition frame that retains its wildtype row.

    ``jaxmodels.Data.from_multidms`` takes ``x_wt = X[0]`` and asserts
    ``x_wt.sum() == 0``, so the empty-substitution row is mandatory.
    """
    rng = np.random.default_rng(0)
    muts = [f"M{s}A" for s in range(1, 40)]
    rows = [{"aa_substitutions": "", "func_score": 0.0, "condition": "Delta"}]
    for _ in range(500):
        k = int(rng.integers(1, 4))
        subs = " ".join(rng.choice(muts, size=k, replace=False))
        rows.append(
            {
                "aa_substitutions": subs,
                "func_score": float(rng.normal(-0.5, 1.0)),
                "condition": "Delta",
            }
        )
    return pd.DataFrame(rows)


def _fit_config(fusionreg, beta0_ridge):
    """A prod-shaped fitting config with the two regularizers parameterized."""
    return {
        "maxiter": 50,
        "tol": 1e-6,
        "recompute_scale": False,
        "l2reg": 1e-6,
        "fusionreg": fusionreg,
        "beta0_ridge": beta0_ridge,
        "ge_type": "Sigmoid",
        "warmstart": False,
        "alpha_init": 6.0,
        "share_alpha": True,
        "beta_clip_range": [-10.0, 10.0],
        "loss_kwargs": {"δ": 1.0},
        "ge_kwargs": {"tol": 1e-4, "maxiter": 10, "maxls": 40, "jit": True},
        "cal_kwargs": {"tol": 1e-4, "maxiter": 10, "maxls": 40, "jit": True},
        "beta0_init": {"Delta": 0.0},
    }


class _FakeModel:
    """Stands in for a fitted single-condition model.

    Mirrors the real ``get_mutations_df`` contract: indexed by ``mutation``,
    carrying ``wts``/``sites``/``muts`` plus ``beta_{condition}`` and
    ``times_seen_{condition}``.
    """

    def __init__(self, condition, betas):
        self._condition = condition
        self._betas = betas

    def get_mutations_df(self, times_seen_threshold=1):
        muts = list(self._betas)
        return pd.DataFrame(
            {
                "wts": [m[0] for m in muts],
                "sites": [int(m[1:-1]) for m in muts],
                "muts": [m[-1] for m in muts],
                f"beta_{self._condition}": [self._betas[m] for m in muts],
                f"times_seen_{self._condition}": [5] * len(muts),
            },
            index=pd.Index(muts, name="mutation"),
        )


def test_regularizers_are_inert_for_a_single_condition(tiny_single_condition_df):
    """V1: fusionreg and beta0_ridge cannot act when there are no shifts.

    Every fusion and ridge term in ``jaxmodels`` is guarded by
    ``d != reference_condition``. With one condition that branch never runs,
    which is why the naive arm needs no lambda ladder -- the claim the whole
    cost model rests on. Assert bit-identical betas, not merely close: any
    drift at all would mean the arm has acquired a lambda dependence.
    """
    from _downstream import fit_single_condition

    model_a, _ = fit_single_condition(
        tiny_single_condition_df, 1, "Delta", _fit_config(0.0, 0.0)
    )
    model_b, _ = fit_single_condition(
        tiny_single_condition_df, 1, "Delta", _fit_config(1e6, 1e6)
    )

    beta_a = model_a.get_mutations_df(times_seen_threshold=1)["beta_Delta"]
    beta_b = model_b.get_mutations_df(times_seen_threshold=1)["beta_Delta"]

    pd.testing.assert_index_equal(beta_a.index, beta_b.index)
    np.testing.assert_allclose(beta_a.to_numpy(), beta_b.to_numpy(), atol=0, rtol=0)


def test_naive_shift_invariants():
    """V2: shifts are exact differences of betas on one shared index.

    Two properties hold by construction and catch index misalignment, the
    failure mode the wildtype-letter trap threatens:

    * the reference condition's shift is identically zero;
    * for any two conditions, ``shift_c - shift_c' == beta_c - beta_c'``.
    """
    from _downstream import derive_naive_shifts

    shared = ["M1A", "M2A", "M3A"]
    models = {
        (1, "Omicron_BA1"): _FakeModel(
            "Omicron_BA1", dict(zip(shared, [0.1, -0.2, 0.3]))
        ),
        (1, "Delta"): _FakeModel("Delta", dict(zip(shared, [0.5, -0.1, 0.9]))),
        (1, "Omicron_BA2"): _FakeModel(
            "Omicron_BA2", dict(zip(shared, [0.2, 0.4, -0.3]))
        ),
    }

    out = derive_naive_shifts(models, reference="Omicron_BA1")

    ref = out[out["condition"] == "Omicron_BA1"]
    np.testing.assert_allclose(ref["naive_shift"].to_numpy(), 0.0, atol=0)

    piv_shift = out.pivot(index="mutation", columns="condition", values="naive_shift")
    piv_beta = out.pivot(index="mutation", columns="condition", values="beta")
    np.testing.assert_allclose(
        (piv_shift["Delta"] - piv_shift["Omicron_BA2"]).to_numpy(),
        (piv_beta["Delta"] - piv_beta["Omicron_BA2"]).to_numpy(),
        atol=1e-12,
    )


def test_naive_shifts_use_the_inner_join():
    """A mutation missing from one condition drops out of every condition.

    A union would silently pair non-equivalent labels, because 32 spike sites
    carry different wildtype letters across conditions.
    """
    from _downstream import derive_naive_shifts

    models = {
        (1, "Omicron_BA1"): _FakeModel(
            "Omicron_BA1", {"M1A": 0.1, "M2A": -0.2, "M3A": 0.3}
        ),
        (1, "Delta"): _FakeModel("Delta", {"M1A": 0.5, "M2A": -0.1}),
    }

    out = derive_naive_shifts(models, reference="Omicron_BA1")

    assert set(out["mutation"]) == {"M1A", "M2A"}
    assert "M3A" not in set(out["mutation"])


def test_empty_intersection_raises():
    """An empty join means the key spaces disagree -- a bug, not missing data."""
    from _downstream import derive_naive_shifts

    models = {
        (1, "Omicron_BA1"): _FakeModel("Omicron_BA1", {"M1A": 0.1}),
        (1, "Delta"): _FakeModel("Delta", {"K9R": 0.5}),
    }

    with pytest.raises(ValueError, match="empty"):
        derive_naive_shifts(models, reference="Omicron_BA1")


def test_wt_agreement_guard():
    """V3: a mutation on a site whose wildtype letter disagrees must raise.

    The "plain intersection is automatically correct" premise holds only
    because a label can appear in every condition's index only when the
    wildtype letter agrees. Verify it rather than trusting it.
    """
    from _downstream import assert_wt_agreement

    conditions = ["Delta", "Omicron_BA1", "Omicron_BA2"]
    naive_muts = pd.DataFrame(
        {"mutation": ["T19I"], "wts": ["T"], "sites": [19], "muts": ["I"]}
    )

    good_map = pd.DataFrame(
        {"sites": [19], "Delta": ["T"], "Omicron_BA1": ["T"], "Omicron_BA2": ["T"]}
    )
    assert assert_wt_agreement(naive_muts, good_map, conditions) == 0

    bad_map = pd.DataFrame(
        {"sites": [19], "Delta": ["I"], "Omicron_BA1": ["T"], "Omicron_BA2": ["T"]}
    )
    with pytest.raises(ValueError, match="wildtype"):
        assert_wt_agreement(naive_muts, bad_map, conditions)


def test_wt_agreement_rejects_unmapped_sites():
    """A site absent from site_map is unverified, not agreeing.

    ``nunique`` skips NaN, so an all-NaN merge row scores 0 distinct letters
    and would pass a bare ``> 1`` disagreement test without ever having been
    checked. The guard must reject it instead.
    """
    from _downstream import assert_wt_agreement

    naive_muts = pd.DataFrame(
        {"mutation": ["T19I"], "wts": ["T"], "sites": [19], "muts": ["I"]}
    )
    missing_site_map = pd.DataFrame(
        {"sites": [20], "Delta": ["A"], "Omicron_BA1": ["A"], "Omicron_BA2": ["A"]}
    )

    with pytest.raises(ValueError, match="absent from site_map"):
        assert_wt_agreement(
            naive_muts, missing_site_map, ["Delta", "Omicron_BA1", "Omicron_BA2"]
        )


def test_missing_reference_for_every_replicate_raises():
    """No reference anywhere means no shifts -- fail clearly, not on concat."""
    from _downstream import derive_naive_shifts

    models = {(1, "Delta"): _FakeModel("Delta", {"M1A": 0.5})}

    with pytest.warns(UserWarning, match="reference condition"):
        with pytest.raises(ValueError, match="No naive shifts"):
            derive_naive_shifts(models, reference="Omicron_BA1")


def test_shifts_are_derived_per_replicate_not_across_them():
    """``derive_naive_shifts`` intersects within a replicate, not across.

    This is the documented contract, and downstream code depends on knowing
    it: ``manuscript_figures`` must intersect the two replicates itself
    before pairing them, or a ``pivot_table`` would union them and leave NaN
    columns that quietly change every reported ``n``.
    """
    from _downstream import derive_naive_shifts

    models = {
        (1, "Omicron_BA1"): _FakeModel("Omicron_BA1", {"M1A": 0.1, "M2A": -0.2}),
        (1, "Delta"): _FakeModel("Delta", {"M1A": 0.5, "M2A": -0.1}),
        (2, "Omicron_BA1"): _FakeModel("Omicron_BA1", {"M2A": 0.3, "M3A": 0.4}),
        (2, "Delta"): _FakeModel("Delta", {"M2A": 0.7, "M3A": 0.9}),
    }

    out = derive_naive_shifts(models, reference="Omicron_BA1")

    per_replicate = out.groupby("replicate")["mutation"].apply(set)
    assert per_replicate[1] == {"M1A", "M2A"}
    assert per_replicate[2] == {"M2A", "M3A"}

    # The union is what a naive pivot would produce; the paired index is what
    # a replicate scatter may actually use.
    wide = out.pivot_table(
        index="mutation", columns=["replicate", "condition"], values="naive_shift"
    )
    assert len(wide) == 3
    assert len(wide.dropna()) == 1
