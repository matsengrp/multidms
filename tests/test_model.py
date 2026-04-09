"""Tests for the new Model v2 wrapper around jaxmodels backend.

This is an optimized version of test_model_v2.py with:
- Session-scoped data fixtures to avoid repeated data creation
- Module-scoped fitted model fixtures to avoid repeated fitting
- Reduced iteration counts where sufficient for testing
"""

import pytest
import multidms
import numpy as np
import pandas as pd
from io import StringIO


# Test data fixture
# NOTE: Must include wildtype (empty aa_substitutions) for each condition
TEST_FUNC_SCORES_STR = """condition,aa_substitutions,func_score
a,,0.0
a,M1E,2.0
a,G3R,-7.0
a,G3P,-0.5
a,M1W,2.3
b,,0.0
b,M1E,1.0
b,P3R,-5.0
b,P3G,0.4
b,M1E P3G,2.7
b,M1E P3R,-2.7
b,P2T,0.3
"""
TEST_FUNC_SCORES = pd.read_csv(StringIO(TEST_FUNC_SCORES_STR))
# Fill NaN in aa_substitutions with empty string (wildtype)
TEST_FUNC_SCORES["aa_substitutions"] = TEST_FUNC_SCORES["aa_substitutions"].fillna("")


@pytest.fixture(scope="session")
def simple_data():
    """Create a simple Data object for testing (session-scoped for reuse)."""
    return multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=True,
        include_counts=False,
    )


@pytest.fixture(scope="session")
def single_condition_data():
    """Create a single-condition Data object for testing (session-scoped)."""
    return multidms.Data(
        TEST_FUNC_SCORES.query("condition == 'a'"),
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=False,
        include_counts=False,
    )


@pytest.fixture(scope="session")
def count_data():
    """Data object with synthetic count data for testing count_loss paths."""
    count_df = TEST_FUNC_SCORES.copy()
    rng = np.random.default_rng(42)
    count_df["pre_count"] = rng.integers(50, 200, size=len(count_df))
    count_df["post_count"] = rng.integers(10, 100, size=len(count_df))
    return multidms.Data(
        count_df,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=True,
        include_counts=True,
    )


@pytest.fixture(scope="module")
def fitted_simple_model(simple_data):
    """Pre-fitted model for read-only tests (module-scoped to share across tests)."""
    model = multidms.Model(simple_data)
    model.fit(maxiter=3, tol=1e-6, warmstart=False)
    return model


@pytest.fixture(scope="module")
def fitted_single_condition_model(single_condition_data):
    """Pre-fitted single-condition model (module-scoped)."""
    model = multidms.Model(single_condition_data)
    model.fit(maxiter=3, warmstart=False)
    return model


@pytest.fixture(scope="module")
def fitted_model_with_reg(simple_data):
    """Pre-fitted model with regularization (module-scoped)."""
    model = multidms.Model(simple_data, l2reg=0.01)
    model.fit(maxiter=5, warmstart=False)
    return model


# ==================== Model Initialization Tests ====================


def test_model_init_default_params(simple_data):
    """Test Model initialization with default parameters."""
    model = multidms.Model(simple_data)

    assert model.data is simple_data
    assert model._ge_type == "Sigmoid"
    assert model._loss_type == "functional_score_loss"
    assert model._l2reg == 0.0
    assert model._fusionreg == 0.0
    assert model._beta0_ridge == 0.0
    assert model.params is None  # Not fitted yet


def test_model_init_custom_params(simple_data):
    """Test Model initialization with custom parameters."""
    model = multidms.Model(
        simple_data,
        ge_type="Identity",
        loss_type="count_loss",
        l2reg=0.01,
        fusionreg=0.05,
        beta0_ridge=0.1,
    )

    assert model._ge_type == "Identity"
    assert model._loss_type == "count_loss"
    assert model._l2reg == 0.01
    assert model._fusionreg == 0.05
    assert model._beta0_ridge == 0.1


def test_model_init_invalid_ge_type(simple_data):
    """Test that invalid ge_type raises ValueError."""
    with pytest.raises(ValueError, match="ge_type must be"):
        multidms.Model(simple_data, ge_type="InvalidType")


def test_model_init_invalid_loss_type(simple_data):
    """Test that invalid loss_type raises ValueError."""
    with pytest.raises(ValueError, match="loss_type must be"):
        multidms.Model(simple_data, loss_type="invalid_loss")


def test_model_repr(simple_data):
    """Test Model string representation."""
    model = multidms.Model(simple_data, ge_type="Sigmoid")
    repr_str = repr(model)

    assert "Model" in repr_str
    assert "Sigmoid" in repr_str
    assert "functional_score_loss" in repr_str


# ==================== Model Fitting Tests ====================


def test_model_fit_basic(simple_data):
    """Test that Model.fit() completes without error."""
    model = multidms.Model(simple_data)
    # Note: warmstart=False because test data doesn't have counts
    # Reduced iterations: 3 instead of 5
    result = model.fit(maxiter=3, tol=1e-6, warmstart=False)

    # Check that fit returns self for method chaining
    assert result is model

    # Check that model is now fitted
    assert model.params is not None
    assert model._jax_model is not None
    assert model._jax_data_sets is not None


def test_model_fit_single_condition(single_condition_data):
    """Test fitting a single-condition model."""
    model = multidms.Model(single_condition_data)
    model.fit(maxiter=3, warmstart=False)

    assert model.params is not None
    assert len(model._jax_data_sets) == 1
    assert "a" in model._jax_data_sets


def test_model_fit_with_identity_ge(simple_data):
    """Test fitting with Identity global epistasis (linear model)."""
    model = multidms.Model(simple_data, ge_type="Identity")
    model.fit(maxiter=3, warmstart=False)

    assert model.params is not None


def test_model_fit_with_warmstart(simple_data):
    """Test fitting with warmstart enabled (no count data needed)."""
    model = multidms.Model(simple_data)
    model.fit(warmstart=True, maxiter=3)
    assert model.params is not None


def test_model_fit_without_warmstart(simple_data):
    """Test fitting without warmstart."""
    model = multidms.Model(simple_data)
    model.fit(warmstart=False, maxiter=3)

    assert model.params is not None


def test_model_fit_convergence_trajectory(simple_data):
    """Test that convergence trajectory is recorded with all columns."""
    model = multidms.Model(simple_data)
    model.fit(maxiter=5, warmstart=False)

    traj_df = model.convergence_trajectory_df
    assert traj_df is not None
    assert isinstance(traj_df, pd.DataFrame)
    assert len(traj_df) > 0

    # Original columns still present
    for col in [
        "iteration",
        "objective_total_trajectory",
        "objective_error_trajectory",
        "loss_trajectory",
    ]:
        assert col in traj_df.columns

    # New per-variant normalized loss
    assert "loss_per_variant_trajectory" in traj_df.columns

    # Block-level diagnostics for all 4 blocks
    for block in ["calibration", "beta0", "beta_nonbundle", "beta_bundle"]:
        for suffix in ["error", "stepsize", "iter_num"]:
            assert f"{block}_{suffix}" in traj_df.columns

    # Shared alpha column
    assert "alpha" in traj_df.columns

    # Per-condition columns
    conditions = simple_data.conditions
    for cond in conditions:
        # theta is only tracked when count data is present
        assert f"theta_{cond}" not in traj_df.columns
        assert f"beta0_{cond}" in traj_df.columns
    # Sparsity only for non-reference conditions
    ref = simple_data.reference
    for cond in conditions:
        if cond != ref:
            assert f"sparsity_{cond}" in traj_df.columns
    assert f"sparsity_{ref}" not in traj_df.columns


def test_convergence_trajectory_block_values(fitted_simple_model):
    """Test that block-level diagnostic values are reasonable."""
    traj_df = fitted_simple_model.convergence_trajectory_df
    for block in ["calibration", "beta0", "beta_nonbundle", "beta_bundle"]:
        assert (traj_df[f"{block}_error"] >= 0).all()
        assert (traj_df[f"{block}_stepsize"] > 0).all()
        assert (traj_df[f"{block}_iter_num"] >= 0).all()


def test_convergence_trajectory_per_variant_loss(fitted_simple_model):
    """Test that loss_per_variant_trajectory is correctly normalized."""
    traj_df = fitted_simple_model.convergence_trajectory_df
    # The ratio loss / loss_per_variant should be constant (= n_variants_total)
    ratios = traj_df["loss_trajectory"] / traj_df["loss_per_variant_trajectory"]
    np.testing.assert_allclose(ratios, ratios.iloc[0])
    # n_variants_total must be a positive integer
    n_variants = ratios.iloc[0]
    assert n_variants > 0
    assert n_variants == int(n_variants)


def test_convergence_trajectory_single_condition(fitted_single_condition_model):
    """Test trajectory columns for single-condition model (no sparsity)."""
    traj_df = fitted_single_condition_model.convergence_trajectory_df
    assert len(traj_df) > 0
    # Shared alpha, beta0 for the condition (no theta without count data)
    assert "alpha" in traj_df.columns
    assert "theta_a" not in traj_df.columns
    assert "beta0_a" in traj_df.columns
    # No sparsity columns (reference is the only condition)
    sparsity_cols = [c for c in traj_df.columns if c.startswith("sparsity_")]
    assert len(sparsity_cols) == 0


def test_per_condition_loss_columns_exist(fitted_simple_model, simple_data):
    """Test that per-condition loss columns exist in trajectory."""
    traj_df = fitted_simple_model.convergence_trajectory_df
    for cond in simple_data.conditions:
        assert f"loss_{cond}" in traj_df.columns
        assert f"loss_per_variant_{cond}" in traj_df.columns


def test_per_condition_loss_sum_all_iterations(fitted_simple_model, simple_data):
    """Test that per-condition losses sum to total loss for every iteration."""
    traj_df = fitted_simple_model.convergence_trajectory_df
    conditions = simple_data.conditions
    per_condition_sum = sum(traj_df[f"loss_{cond}"] for cond in conditions)
    np.testing.assert_allclose(per_condition_sum, traj_df["loss_trajectory"], rtol=1e-5)


def test_per_condition_loss_non_negative(fitted_simple_model, simple_data):
    """Test that all per-condition loss values are non-negative."""
    traj_df = fitted_simple_model.convergence_trajectory_df
    for cond in simple_data.conditions:
        assert (traj_df[f"loss_{cond}"] >= 0).all()
        assert (traj_df[f"loss_per_variant_{cond}"] >= 0).all()


def test_per_condition_loss_per_variant_normalization(simple_data):
    """Test that loss_per_variant_{d} = loss_{d} / n_variants_for_d."""
    model = multidms.Model(simple_data)
    model.fit(maxiter=3, warmstart=False)
    traj_df = model.convergence_trajectory_df

    for cond in simple_data.conditions:
        # Use the jax data variant count (excludes wildtype row)
        n_variants = model._jax_data_sets[cond].functional_scores.shape[0]
        expected = traj_df[f"loss_{cond}"] / n_variants
        np.testing.assert_allclose(
            traj_df[f"loss_per_variant_{cond}"], expected, rtol=1e-10
        )


def test_single_condition_loss_equals_total(fitted_single_condition_model):
    """For single-condition model, per-condition loss equals total loss."""
    traj_df = fitted_single_condition_model.convergence_trajectory_df
    assert len(traj_df) > 0
    np.testing.assert_allclose(
        traj_df["loss_a"], traj_df["loss_trajectory"], rtol=1e-10
    )
    np.testing.assert_allclose(
        traj_df["loss_per_variant_a"],
        traj_df["loss_per_variant_trajectory"],
        rtol=1e-10,
    )


def test_convergence_trajectory_theta_with_count_data(count_data):
    """Test that theta columns ARE tracked when count data is present."""
    model = multidms.Model(count_data, loss_type="count_loss")
    model.fit(maxiter=3, warmstart=False)
    traj_df = model.convergence_trajectory_df
    assert len(traj_df) > 0
    for cond in count_data.conditions:
        assert f"theta_{cond}" in traj_df.columns
        # theta values should be positive (exp of logθ)
        assert (traj_df[f"theta_{cond}"] > 0).all()


# ==================== share_alpha Tests ====================


def test_fit_share_alpha_false(simple_data):
    """Test that share_alpha=False produces per-condition alpha dict."""
    model = multidms.Model(simple_data)
    model.fit(maxiter=3, warmstart=False, verbose=False, share_alpha=False)

    jm = model._jax_model
    assert isinstance(jm.α, dict)
    assert set(jm.α.keys()) == set(simple_data.conditions)
    for v in jm.α.values():
        assert v.shape == ()

    # Trajectory should have alpha_{cond} columns, not "alpha"
    traj_df = model.convergence_trajectory_df
    assert "alpha" not in traj_df.columns
    for cond in simple_data.conditions:
        assert f"alpha_{cond}" in traj_df.columns


def test_fit_share_alpha_true_default(simple_data):
    """Test that default share_alpha=True produces scalar alpha."""
    model = multidms.Model(simple_data)
    model.fit(maxiter=3, warmstart=False, verbose=False)

    jm = model._jax_model
    assert not isinstance(jm.α, dict)
    assert jm.α.shape == ()

    traj_df = model.convergence_trajectory_df
    assert "alpha" in traj_df.columns
    for cond in simple_data.conditions:
        assert f"alpha_{cond}" not in traj_df.columns


# ==================== get_mutations_df Tests ====================


def test_get_mutations_df_before_fit_raises(simple_data):
    """Test that get_mutations_df raises error before fitting."""
    model = multidms.Model(simple_data)

    with pytest.raises(ValueError, match="Model has not been fitted"):
        model.get_mutations_df()


def test_get_mutations_df_after_fit(fitted_simple_model):
    """Test get_mutations_df returns correct format after fitting."""
    muts_df = fitted_simple_model.get_mutations_df()

    # Check it's a DataFrame
    assert isinstance(muts_df, pd.DataFrame)

    # Check mutation is the index
    assert muts_df.index.name == "mutation"

    # Check required columns exist (wide format: beta_a, beta_b, shift_b)
    assert "beta_a" in muts_df.columns
    assert "beta_b" in muts_df.columns
    assert "shift_b" in muts_df.columns

    # Check we have one row per mutation (wide format)
    n_mutations = len(fitted_simple_model.data.mutations)
    assert len(muts_df) == n_mutations

    # Check mutation values are strings (index)
    assert all(isinstance(m, str) for m in muts_df.index)

    # Check beta values are numeric
    assert pd.api.types.is_numeric_dtype(muts_df["beta_a"])
    assert pd.api.types.is_numeric_dtype(muts_df["beta_b"])


def test_get_mutations_df_contains_all_mutations(fitted_simple_model):
    """Test that get_mutations_df includes all mutations from data."""
    muts_df = fitted_simple_model.get_mutations_df()
    mutations_in_df = set(muts_df.index)
    mutations_in_data = set(fitted_simple_model.data.mutations)

    assert mutations_in_df == mutations_in_data


def test_get_mutations_df_single_condition(fitted_single_condition_model):
    """Test get_mutations_df with single condition."""
    muts_df = fitted_single_condition_model.get_mutations_df()

    # Should only have beta_a column (no shift columns for single condition)
    assert "beta_a" in muts_df.columns
    assert "shift_a" not in muts_df.columns  # No shift for reference condition


def test_get_mutations_df_has_predicted_func_score_columns(fitted_simple_model):
    """Test that get_mutations_df includes predicted_func_score columns."""
    muts_df = fitted_simple_model.get_mutations_df()

    assert "predicted_func_score_a" in muts_df.columns
    assert "predicted_func_score_b" in muts_df.columns

    # Values should be numeric and finite
    for col in ["predicted_func_score_a", "predicted_func_score_b"]:
        assert pd.api.types.is_numeric_dtype(muts_df[col])
        assert not muts_df[col].isna().any()
        assert np.all(np.isfinite(muts_df[col]))


def test_get_mutations_df_predicted_func_score_matches_add_phenotypes(
    fitted_simple_model,
):
    """Test predicted_func_score columns match add_phenotypes_to_df on single mutants."""
    muts_df = fitted_simple_model.get_mutations_df()

    # Build a single-mutant DataFrame for all mutations in all conditions
    rows = []
    for condition in fitted_simple_model.data.conditions:
        for mutation in fitted_simple_model.data.mutations:
            rows.append({"condition": condition, "aa_substitutions": mutation})
    df_single_muts = pd.DataFrame(rows)

    result = fitted_simple_model.add_phenotypes_to_df(df_single_muts)

    for condition in fitted_simple_model.data.conditions:
        cond_result = result[result["condition"] == condition]
        for _, row in cond_result.iterrows():
            mutation = row["aa_substitutions"]
            expected = row["predicted_func_score"]
            actual = muts_df.loc[mutation, f"predicted_func_score_{condition}"]
            assert abs(expected - actual) < 1e-5, (
                f"Mismatch for {mutation} in {condition}: "
                f"add_phenotypes={expected}, get_mutations_df={actual}"
            )


def test_identity_ge_predicted_func_score_equals_alpha_times_beta(simple_data):
    """Test that for Identity GE, predicted_func_score ≈ alpha * beta for reference.

    With g = identity and x_wt = 0 for the reference condition:
    predicted_func_score = alpha * (phi(X_i) - phi(x_wt))
                         = alpha * ((beta0 + beta_i) - beta0)
                         = alpha * beta_i
    """
    model = multidms.Model(simple_data, ge_type="Identity")
    model.fit(maxiter=5, warmstart=False, verbose=False)

    muts_df = model.get_mutations_df()
    ref = model.data.reference

    alpha = float(model._jax_model.α)
    expected = muts_df[f"beta_{ref}"].values * alpha
    actual = muts_df[f"predicted_func_score_{ref}"].values

    assert np.allclose(actual, expected, atol=1e-5), (
        f"For Identity GE on reference condition, "
        f"predicted_func_score should equal alpha * beta.\n"
        f"alpha={alpha}, max diff={np.max(np.abs(actual - expected))}"
    )


def test_get_mutations_df_predicted_func_score_single_condition(
    fitted_single_condition_model,
):
    """Test predicted_func_score with a single condition model."""
    muts_df = fitted_single_condition_model.get_mutations_df()

    assert "predicted_func_score_a" in muts_df.columns
    assert not muts_df["predicted_func_score_a"].isna().any()


def test_get_mutations_df_has_shift_column(fitted_model_with_reg):
    """Test that get_mutations_df includes shift column."""
    muts_df = fitted_model_with_reg.get_mutations_df()

    # Check shift_b column exists (wide format)
    assert "shift_b" in muts_df.columns

    # Reference condition should not have shift column
    assert "shift_a" not in muts_df.columns


def test_get_mutations_df_shift_values(fitted_model_with_reg):
    """Test that shift values are calculated correctly."""
    muts_df = fitted_model_with_reg.get_mutations_df()

    # For each mutation, shift_b should equal beta_b - beta_a (wide format)
    for mutation in fitted_model_with_reg.data.mutations:
        row = muts_df.loc[mutation]
        beta_a = row["beta_a"]
        beta_b = row["beta_b"]
        shift_b = row["shift_b"]

        # shift should be beta_b - beta_a
        expected_shift = beta_b - beta_a
        assert abs(shift_b - expected_shift) < 1e-6


# ==================== get_variants_df Tests ====================


def test_get_variants_df_before_fit_raises(simple_data):
    """Test that get_variants_df raises error before fitting."""
    model = multidms.Model(simple_data)

    with pytest.raises(ValueError, match="Model has not been fitted"):
        model.get_variants_df()


def test_get_variants_df_after_fit(fitted_simple_model):
    """Test get_variants_df returns correct format after fitting."""
    vars_df = fitted_simple_model.get_variants_df()

    # Check it's a DataFrame
    assert isinstance(vars_df, pd.DataFrame)

    # Check required columns exist
    assert "condition" in vars_df.columns
    assert "aa_substitutions" in vars_df.columns
    assert "func_score" in vars_df.columns
    assert "predicted_func_score" in vars_df.columns

    # Check predicted_func_score is numeric
    assert pd.api.types.is_numeric_dtype(vars_df["predicted_func_score"])

    # Check we have predictions for all variants
    assert len(vars_df) == len(fitted_simple_model.data.variants_df)


def test_get_variants_df_predictions_are_numeric(fitted_simple_model):
    """Test that predictions are valid numeric values."""
    vars_df = fitted_simple_model.get_variants_df()

    # Check no NaN values in predictions
    assert not vars_df["predicted_func_score"].isna().any()

    # Check predictions are finite
    assert np.all(np.isfinite(vars_df["predicted_func_score"]))


def test_get_variants_df_single_condition(fitted_single_condition_model):
    """Test get_variants_df with single condition."""
    vars_df = fitted_single_condition_model.get_variants_df()

    # Should only have one condition
    assert vars_df["condition"].nunique() == 1
    assert vars_df["condition"].iloc[0] == "a"


# ==================== Integration Tests ====================


def test_model_fit_improves_predictions(simple_data):
    """Test that fitting improves predictions (reduces error)."""
    # Fit with very few iterations
    model1 = multidms.Model(simple_data)
    model1.fit(maxiter=1, warmstart=False)
    vars_df1 = model1.get_variants_df()

    # Calculate error
    error1 = np.mean((vars_df1["func_score"] - vars_df1["predicted_func_score"]) ** 2)

    # Fit with more iterations (reduced from 20 to 10)
    model2 = multidms.Model(simple_data)
    model2.fit(maxiter=10, warmstart=False)
    vars_df2 = model2.get_variants_df()

    # Calculate error
    error2 = np.mean((vars_df2["func_score"] - vars_df2["predicted_func_score"]) ** 2)

    # More iterations should reduce error (or at least not increase it significantly)
    assert error2 <= error1 * 1.1  # Allow 10% tolerance for numerical variation


def test_model_with_regularization(simple_data):
    """Test that models can be fitted with different regularization."""
    # No regularization
    model1 = multidms.Model(simple_data, l2reg=0.0)
    model1.fit(maxiter=5, warmstart=False)
    muts_df1 = model1.get_mutations_df()

    # With L2 regularization
    model2 = multidms.Model(simple_data, l2reg=1.0)
    model2.fit(maxiter=5, warmstart=False)
    muts_df2 = model2.get_mutations_df()

    # Regularized model should have smaller parameter magnitudes (use beta_a)
    beta_mag1 = np.abs(muts_df1["beta_a"]).mean()
    beta_mag2 = np.abs(muts_df2["beta_a"]).mean()

    assert beta_mag2 < beta_mag1


def test_model_deterministic_with_same_params(simple_data):
    """Test that model fitting is deterministic."""
    model1 = multidms.Model(simple_data)
    model1.fit(maxiter=5, warmstart=False)
    muts_df1 = model1.get_mutations_df()

    model2 = multidms.Model(simple_data)
    model2.fit(maxiter=5, warmstart=False)
    muts_df2 = model2.get_mutations_df()

    # Results should be very close (allowing for small numerical differences)
    # Compare beta_a values (wide format)
    assert np.allclose(
        muts_df1["beta_a"].values,
        muts_df2["beta_a"].values,
        rtol=1e-5,
        atol=1e-8,
    )


# ==================== add_phenotypes_to_df Tests ====================


def test_add_phenotypes_to_df_before_fit_raises(simple_data):
    """Test that add_phenotypes_to_df raises error before fitting."""
    model = multidms.Model(simple_data)
    df_new = pd.DataFrame({"condition": ["a"], "aa_substitutions": ["M1E"]})

    with pytest.raises(ValueError, match="Model has not been fitted"):
        model.add_phenotypes_to_df(df_new)


def test_add_phenotypes_to_df_basic(fitted_simple_model):
    """Test basic functionality of add_phenotypes_to_df."""
    df_new = pd.DataFrame(
        {"condition": ["a", "a", "b"], "aa_substitutions": ["M1E", "G3R", "M1E"]}
    )

    result = fitted_simple_model.add_phenotypes_to_df(df_new)

    # Check it returns a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check it has the prediction column
    assert "predicted_func_score" in result.columns

    # Check it has the same number of rows
    assert len(result) == len(df_new)

    # Check predictions are numeric
    assert pd.api.types.is_numeric_dtype(result["predicted_func_score"])

    # Check no NaN predictions
    assert not result["predicted_func_score"].isna().any()


def test_add_phenotypes_to_df_preserves_input_columns(fitted_simple_model):
    """Test that add_phenotypes_to_df preserves all input columns."""
    df_new = pd.DataFrame(
        {
            "condition": ["a", "b"],
            "aa_substitutions": ["M1E", "M1E"],
            "extra_col": ["foo", "bar"],
        }
    )

    result = fitted_simple_model.add_phenotypes_to_df(df_new)

    # Check original columns are preserved
    assert "condition" in result.columns
    assert "aa_substitutions" in result.columns
    assert "extra_col" in result.columns

    # Check values are preserved
    assert result["extra_col"].tolist() == ["foo", "bar"]


def test_add_phenotypes_to_df_wildtype(fitted_simple_model):
    """Test predictions on wildtype (empty substitutions)."""
    df_new = pd.DataFrame({"condition": ["a", "b"], "aa_substitutions": ["", ""]})

    result = fitted_simple_model.add_phenotypes_to_df(df_new)

    # Wildtype should have prediction of ~0 (effect relative to WT)
    assert abs(result.loc[0, "predicted_func_score"]) < 1e-6
    assert abs(result.loc[1, "predicted_func_score"]) < 1e-6


def test_add_phenotypes_to_df_unseen_mutations_raises(fitted_simple_model):
    """Test that unseen mutations raise an informative error."""
    df_new = pd.DataFrame(
        {
            "condition": ["a"],
            "aa_substitutions": ["M1Z"],  # Z is not a valid amino acid in training
        }
    )

    with pytest.raises(ValueError, match="mutations not seen during training"):
        fitted_simple_model.add_phenotypes_to_df(df_new)


def test_add_phenotypes_to_df_missing_required_columns_raises(fitted_simple_model):
    """Test that missing required columns raise errors."""
    # Missing aa_substitutions
    df_no_subs = pd.DataFrame({"condition": ["a"]})
    with pytest.raises(ValueError, match="lacks column 'aa_substitutions'"):
        fitted_simple_model.add_phenotypes_to_df(df_no_subs)

    # Missing condition
    df_no_cond = pd.DataFrame({"aa_substitutions": ["M1E"]})
    with pytest.raises(ValueError, match="lacks column 'condition'"):
        fitted_simple_model.add_phenotypes_to_df(df_no_cond)


def test_add_phenotypes_to_df_invalid_condition_raises(fitted_simple_model):
    """Test that invalid conditions raise an error."""
    df_new = pd.DataFrame(
        {"condition": ["c"], "aa_substitutions": ["M1E"]}  # 'c' not in training data
    )

    with pytest.raises(ValueError, match="Invalid conditions"):
        fitted_simple_model.add_phenotypes_to_df(df_new)


def test_add_phenotypes_to_df_non_unique_index_raises(fitted_simple_model):
    """Test that non-unique indices raise an error."""
    df_new = pd.DataFrame(
        {"condition": ["a", "a"], "aa_substitutions": ["M1E", "G3R"]}, index=[0, 0]
    )  # Duplicate index

    with pytest.raises(ValueError, match="must have unique indices"):
        fitted_simple_model.add_phenotypes_to_df(df_new)


def test_add_phenotypes_to_df_overwrite_cols(fitted_simple_model):
    """Test overwrite_cols parameter."""
    df_new = pd.DataFrame(
        {
            "condition": ["a"],
            "aa_substitutions": ["M1E"],
            "predicted_func_score": [999.0],  # Existing column
        }
    )

    # Should raise error without overwrite_cols=True
    with pytest.raises(ValueError, match="already contains column"):
        fitted_simple_model.add_phenotypes_to_df(df_new)

    # Should work with overwrite_cols=True
    result = fitted_simple_model.add_phenotypes_to_df(df_new, overwrite_cols=True)
    assert result["predicted_func_score"].iloc[0] != 999.0  # Should be overwritten


def test_add_phenotypes_to_df_custom_column_names(fitted_simple_model):
    """Test custom column name parameters."""
    df_new = pd.DataFrame({"cond": ["a"], "subs": ["M1E"]})

    result = fitted_simple_model.add_phenotypes_to_df(
        df_new,
        condition_col="cond",
        substitutions_col="subs",
        predicted_phenotype_col="my_prediction",
    )

    # Check custom column name was used
    assert "my_prediction" in result.columns
    assert "predicted_func_score" not in result.columns


def test_add_phenotypes_to_df_multi_mutation_variants(fitted_simple_model):
    """Test predictions on variants with multiple mutations."""
    df_new = pd.DataFrame(
        {
            "condition": ["b"],
            "aa_substitutions": ["M1E P3G"],  # Double mutant seen in training
        }
    )

    result = fitted_simple_model.add_phenotypes_to_df(df_new)

    # Should have a prediction
    assert not result["predicted_func_score"].isna().any()
    assert np.isfinite(result["predicted_func_score"].iloc[0])


def test_add_phenotypes_to_df_consistency_with_get_variants_df(fitted_simple_model):
    """Test that predictions match get_variants_df for training data."""
    # Get predictions on training data using get_variants_df
    training_preds = fitted_simple_model.get_variants_df()

    # Get predictions using add_phenotypes_to_df on same data
    df_test = training_preds[["condition", "aa_substitutions"]].copy()
    test_preds = fitted_simple_model.add_phenotypes_to_df(df_test)

    # Predictions should match (allowing for small numerical differences)
    for idx in training_preds.index:
        expected = training_preds.loc[idx, "predicted_func_score"]
        actual = test_preds.loc[idx, "predicted_func_score"]
        assert (
            abs(expected - actual) < 1e-5
        ), f"Prediction mismatch at index {idx}: {expected} vs {actual}"


def test_add_phenotypes_to_df_with_explicit_parameters(simple_data):
    """Test predictions with explicitly set parameters match manual calculations."""
    # Set up explicit parameters
    # For simple_data: conditions are 'a' (reference) and 'b'
    # Mutations from simple_data are: M1E, M1W, G3P, G3R (in order by mutation index)

    # Create explicit parameter values
    # β0: intercepts (effect of wildtype)
    beta0_values = {
        "a": 0.0,  # Reference condition
        "b": 0.5,  # Non-reference has different intercept
    }

    # β: mutation effects (same order as simple_data.mutations)
    beta_values = {
        "a": np.array([1.0, 2.0, -0.5, -1.5]),  # Effects for M1E, M1W, G3P, G3R
        "b": np.array([1.2, 2.2, -0.3, -1.3]),  # Different effects in condition b
    }

    # α: shared scaling factor
    alpha_values = 1.0

    # Create model with Identity global epistasis for simple linear predictions
    model = multidms.Model(simple_data, ge_type="Identity", l2reg=0.0)

    # Fit with maxiter=0 to keep exactly the initialized parameters
    model.fit(
        warmstart=False,
        maxiter=0,
        beta0_init=beta0_values,
        beta_init=beta_values,
        alpha_init=alpha_values,
        verbose=False,
    )

    # Create test dataframe with explicit variants
    # Test M1E in condition 'a'
    df_test = pd.DataFrame(
        {
            "condition": ["a", "b", "a"],
            "aa_substitutions": ["M1E", "M1E", "M1W G3R"],
        }
    )

    # Get predictions
    result = model.add_phenotypes_to_df(df_test)

    # Manually calculate expected predictions
    # For Identity global epistasis: predicted_score = α * (φ(variant) - φ(wt))
    # where φ(x) = β0 + sum(β_i * x_i)

    # Condition 'a', variant 'M1E' (mutation index 0)
    # φ(M1E) = 0.0 + 1.0 = 1.0
    # φ(wt) = 0.0 + 0.0 = 0.0
    # predicted = 1.0 * (1.0 - 0.0) = 1.0
    expected_a_M1E = 1.0

    # Condition 'b', variant 'M1E'
    # φ(M1E) = 0.5 + 1.2 = 1.7
    # φ(wt) = 0.5 + 0.0 = 0.5
    # predicted = 1.0 * (1.7 - 0.5) = 1.2
    expected_b_M1E = 1.2

    # Condition 'a', variant 'M1W G3R' (mutations at indices 1 and 3)
    # φ(M1W G3R) = 0.0 + 2.0 + (-1.5) = 0.5
    # φ(wt) = 0.0
    # predicted = 1.0 * (0.5 - 0.0) = 0.5
    expected_a_M1W_G3R = 0.5

    # Check predictions match expected values
    assert abs(result.loc[0, "predicted_func_score"] - expected_a_M1E) < 1e-6, (
        f"Mismatch for 'a' M1E: expected {expected_a_M1E}, "
        f"got {result.loc[0, 'predicted_func_score']}"
    )

    assert abs(result.loc[1, "predicted_func_score"] - expected_b_M1E) < 1e-6, (
        f"Mismatch for 'b' M1E: expected {expected_b_M1E}, "
        f"got {result.loc[1, 'predicted_func_score']}"
    )

    assert abs(result.loc[2, "predicted_func_score"] - expected_a_M1W_G3R) < 1e-6, (
        f"Mismatch for 'a' M1W G3R: expected {expected_a_M1W_G3R}, "
        f"got {result.loc[2, 'predicted_func_score']}"
    )


# ==================== Latent Phenotype & Fitness Column Tests ====================


def test_get_variants_df_has_latent_and_fitness_columns(fitted_simple_model):
    """Test that get_variants_df includes latent and fitness columns."""
    vars_df = fitted_simple_model.get_variants_df()

    assert "predicted_latent" in vars_df.columns
    assert "predicted_fitness" in vars_df.columns
    assert "measured_fitness" in vars_df.columns

    # All values should be numeric and finite
    for col in ["predicted_latent", "predicted_fitness", "measured_fitness"]:
        assert pd.api.types.is_numeric_dtype(vars_df[col])
        assert not vars_df[col].isna().any()
        assert np.all(np.isfinite(vars_df[col]))


def test_predicted_fitness_equals_ge_of_latent(fitted_simple_model):
    """Test that predicted_fitness == g(predicted_latent) for predicted data.

    Since predicted_func_score = α * (g(φ(X)) - g(φ(x_wt))),
    then predicted_fitness = predicted_func_score / α + g(φ(x_wt)) = g(φ(X)).
    """
    vars_df = fitted_simple_model.get_variants_df()
    import jax.numpy as jnp

    for condition in fitted_simple_model.data.conditions:
        cond_df = vars_df[vars_df["condition"] == condition]
        latent_values = jnp.array(cond_df["predicted_latent"].values)
        ge_values = np.array(
            fitted_simple_model._jax_model.global_epistasis(latent_values)
        )
        assert np.allclose(
            cond_df["predicted_fitness"].values, ge_values, atol=1e-5
        ), f"predicted_fitness != g(predicted_latent) for condition {condition}"


def test_wildtype_latent_matches_predicted_latent(fitted_simple_model):
    """Test that WT rows have predicted_latent == wildtype_latent."""
    vars_df = fitted_simple_model.get_variants_df()
    wt_latent = fitted_simple_model.wildtype_latent

    for condition in fitted_simple_model.data.conditions:
        cond_df = vars_df[vars_df["condition"] == condition]
        # WT is the first row (empty aa_substitutions)
        wt_row = cond_df.iloc[0]
        assert wt_row["aa_substitutions"].strip() == ""
        assert abs(wt_row["predicted_latent"] - wt_latent[condition]) < 1e-5, (
            f"WT latent mismatch for condition {condition}: "
            f"{wt_row['predicted_latent']} != {wt_latent[condition]}"
        )


def test_identity_ge_fitness_formula(simple_data):
    """Test fitness formula with Identity global epistasis.

    With g = identity: fitness = func_score / α + φ(x_wt).
    """
    model = multidms.Model(simple_data, ge_type="Identity")
    model.fit(maxiter=3, warmstart=False, verbose=False)

    vars_df = model.get_variants_df()
    wt_latent = model.wildtype_latent

    for condition in model.data.conditions:
        cond_df = vars_df[vars_df["condition"] == condition]
        α = float(model._jax_model.α)
        φ_wt = wt_latent[condition]
        expected_measured = cond_df["func_score"].values / α + φ_wt
        assert np.allclose(
            cond_df["measured_fitness"].values, expected_measured, atol=1e-5
        )


def test_get_variants_df_single_condition_latent(fitted_single_condition_model):
    """Test latent and fitness columns with a single-condition model."""
    vars_df = fitted_single_condition_model.get_variants_df()

    assert "predicted_latent" in vars_df.columns
    assert "predicted_fitness" in vars_df.columns
    assert "measured_fitness" in vars_df.columns
    assert vars_df["condition"].nunique() == 1


def test_add_phenotypes_has_latent_and_fitness(fitted_simple_model):
    """Test that add_phenotypes_to_df includes latent and fitness columns."""
    df_new = pd.DataFrame(
        {
            "condition": ["a", "b"],
            "aa_substitutions": ["M1E", "M1E"],
        }
    )

    result = fitted_simple_model.add_phenotypes_to_df(df_new)

    assert "predicted_latent" in result.columns
    assert "predicted_fitness" in result.columns
    # No func_score column, so measured_fitness should NOT be present
    assert "measured_fitness" not in result.columns

    # Values should be numeric and finite
    for col in ["predicted_latent", "predicted_fitness"]:
        assert not result[col].isna().any()
        assert np.all(np.isfinite(result[col]))


def test_add_phenotypes_with_func_score_has_measured_fitness(fitted_simple_model):
    """Test that measured_fitness is added when func_score is present."""
    df_new = pd.DataFrame(
        {
            "condition": ["a", "b"],
            "aa_substitutions": ["M1E", "M1E"],
            "func_score": [1.5, 2.0],
        }
    )

    result = fitted_simple_model.add_phenotypes_to_df(df_new)

    assert "measured_fitness" in result.columns
    assert not result["measured_fitness"].isna().any()


def test_add_phenotypes_latent_consistency_with_get_variants_df(
    fitted_simple_model,
):
    """Test that predicted_latent matches between the two methods."""
    training_preds = fitted_simple_model.get_variants_df()
    df_test = training_preds[["condition", "aa_substitutions"]].copy()
    test_preds = fitted_simple_model.add_phenotypes_to_df(df_test)

    for idx in training_preds.index:
        expected = training_preds.loc[idx, "predicted_latent"]
        actual = test_preds.loc[idx, "predicted_latent"]
        assert (
            abs(expected - actual) < 1e-5
        ), f"Latent mismatch at index {idx}: {expected} vs {actual}"


# ==================== get_ge_landscape_df Tests ====================


def test_get_ge_landscape_df_returns_tuple(fitted_simple_model):
    """Test that get_ge_landscape_df returns a (variants_df, ge_curve) tuple."""
    result = fitted_simple_model.get_ge_landscape_df()

    assert isinstance(result, tuple)
    assert len(result) == 2

    variants_df, ge_curve = result
    assert isinstance(variants_df, pd.DataFrame)
    assert isinstance(ge_curve, pd.DataFrame)


def test_get_ge_landscape_df_variants_has_wildtype_latent(fitted_simple_model):
    """Test variants_df from get_ge_landscape_df has wildtype_latent column."""
    variants_df, _ = fitted_simple_model.get_ge_landscape_df()

    assert "wildtype_latent" in variants_df.columns
    assert not variants_df["wildtype_latent"].isna().any()

    # Check wildtype_latent values match the property
    wt_latent = fitted_simple_model.wildtype_latent
    for condition in fitted_simple_model.data.conditions:
        cond_vals = variants_df[variants_df["condition"] == condition][
            "wildtype_latent"
        ]
        assert np.allclose(cond_vals, wt_latent[condition], atol=1e-5)


def test_get_ge_landscape_df_curve_columns(fitted_simple_model):
    """Test ge_curve DataFrame has correct columns."""
    _, ge_curve = fitted_simple_model.get_ge_landscape_df()

    assert "predicted_latent" in ge_curve.columns
    assert "ge_curve_value" in ge_curve.columns
    assert len(ge_curve) == 200  # default n_curve_points


def test_get_ge_landscape_df_custom_curve_points(fitted_simple_model):
    """Test custom n_curve_points parameter."""
    _, ge_curve = fitted_simple_model.get_ge_landscape_df(n_curve_points=50)
    assert len(ge_curve) == 50


# ==================== ge_landscape Plot Tests ====================


def test_ge_landscape_plot_returns_chart(fitted_simple_model):
    """Test that ge_landscape returns an Altair chart."""
    import altair as alt
    import multidms.plot as mplt

    variants_df, ge_curve_df = fitted_simple_model.get_ge_landscape_df()
    chart = mplt.ge_landscape(variants_df, ge_curve_df)
    assert isinstance(chart, alt.LayerChart)


def test_ge_landscape_plot_with_predicted_fitness(fitted_simple_model):
    """Test ge_landscape with predicted_fitness column."""
    import altair as alt
    import multidms.plot as mplt

    variants_df, ge_curve_df = fitted_simple_model.get_ge_landscape_df()
    chart = mplt.ge_landscape(variants_df, ge_curve_df, fitness_col="predicted_fitness")
    assert isinstance(chart, alt.LayerChart)


def test_plot_ge_landscape_convenience(fitted_simple_model):
    """Test Model.plot_ge_landscape() convenience method."""
    import altair as alt

    chart = fitted_simple_model.plot_ge_landscape()
    assert isinstance(chart, alt.LayerChart)
