"""Tests for the new Model v2 wrapper around jaxmodels backend.

This test file is for the refactored Model class that wraps jaxmodels.
Tests for the legacy v1 Model class remain in test_data.py for now.
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


@pytest.fixture
def simple_data():
    """Create a simple Data object for testing."""
    return multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=True,
        include_counts=False,
    )


@pytest.fixture
def single_condition_data():
    """Create a single-condition Data object for testing."""
    return multidms.Data(
        TEST_FUNC_SCORES.query("condition == 'a'"),
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=False,
        include_counts=False,
    )


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
    result = model.fit(maxiter=5, tol=1e-6, warmstart=False)

    # Check that fit returns self for method chaining
    assert result is model

    # Check that model is now fitted
    assert model.params is not None
    assert model._jax_model is not None
    assert model._jax_data_sets is not None


def test_model_fit_single_condition(single_condition_data):
    """Test fitting a single-condition model."""
    model = multidms.Model(single_condition_data)
    model.fit(maxiter=5, warmstart=False)

    assert model.params is not None
    assert len(model._jax_data_sets) == 1
    assert "a" in model._jax_data_sets


def test_model_fit_with_identity_ge(simple_data):
    """Test fitting with Identity global epistasis (linear model)."""
    model = multidms.Model(simple_data, ge_type="Identity")
    model.fit(maxiter=5, warmstart=False)

    assert model.params is not None


def test_model_fit_with_warmstart(simple_data):
    """Test fitting with warmstart enabled (requires count data)."""
    # Skip this test for now - warmstart requires count data
    pytest.skip("Warmstart requires count data which test data doesn't have")


def test_model_fit_without_warmstart(simple_data):
    """Test fitting without warmstart."""
    model = multidms.Model(simple_data)
    model.fit(warmstart=False, maxiter=5)

    assert model.params is not None


def test_model_fit_convergence_trajectory(simple_data):
    """Test that convergence trajectory is recorded."""
    model = multidms.Model(simple_data)
    model.fit(maxiter=10, warmstart=False)

    traj_df = model.convergence_trajectory_df
    assert traj_df is not None
    assert isinstance(traj_df, pd.DataFrame)
    assert "iteration" in traj_df.columns
    assert "loss" in traj_df.columns
    assert "error" in traj_df.columns
    assert len(traj_df) > 0


# ==================== get_mutations_df Tests ====================


def test_get_mutations_df_before_fit_raises(simple_data):
    """Test that get_mutations_df raises error before fitting."""
    model = multidms.Model(simple_data)

    with pytest.raises(ValueError, match="Model has not been fitted"):
        model.get_mutations_df()


def test_get_mutations_df_after_fit(simple_data):
    """Test get_mutations_df returns correct format after fitting."""
    model = multidms.Model(simple_data)
    model.fit(maxiter=5, warmstart=False)

    muts_df = model.get_mutations_df()

    # Check it's a DataFrame
    assert isinstance(muts_df, pd.DataFrame)

    # Check mutation is the index
    assert muts_df.index.name == "mutation"

    # Check required columns exist (wide format: beta_a, beta_b, shift_b)
    assert "beta_a" in muts_df.columns
    assert "beta_b" in muts_df.columns
    assert "shift_b" in muts_df.columns

    # Check we have one row per mutation (wide format)
    n_mutations = len(simple_data.mutations)
    assert len(muts_df) == n_mutations

    # Check mutation values are strings (index)
    assert all(isinstance(m, str) for m in muts_df.index)

    # Check beta values are numeric
    assert pd.api.types.is_numeric_dtype(muts_df["beta_a"])
    assert pd.api.types.is_numeric_dtype(muts_df["beta_b"])


def test_get_mutations_df_contains_all_mutations(simple_data):
    """Test that get_mutations_df includes all mutations from data."""
    model = multidms.Model(simple_data)
    model.fit(maxiter=5, warmstart=False)

    muts_df = model.get_mutations_df()
    mutations_in_df = set(muts_df.index)
    mutations_in_data = set(simple_data.mutations)

    assert mutations_in_df == mutations_in_data


def test_get_mutations_df_single_condition(single_condition_data):
    """Test get_mutations_df with single condition."""
    model = multidms.Model(single_condition_data)
    model.fit(maxiter=5, warmstart=False)

    muts_df = model.get_mutations_df()

    # Should only have beta_a column (no shift columns for single condition)
    assert "beta_a" in muts_df.columns
    assert "shift_a" not in muts_df.columns  # No shift for reference condition


def test_get_mutations_df_has_shift_column(simple_data):
    """Test that get_mutations_df includes shift column."""
    model = multidms.Model(simple_data, l2reg=0.01)
    model.fit(maxiter=10, warmstart=False)

    muts_df = model.get_mutations_df()

    # Check shift_b column exists (wide format)
    assert "shift_b" in muts_df.columns

    # Reference condition should not have shift column
    assert "shift_a" not in muts_df.columns


def test_get_mutations_df_shift_values(simple_data):
    """Test that shift values are calculated correctly."""
    model = multidms.Model(simple_data, l2reg=0.01)
    model.fit(maxiter=10, warmstart=False)

    muts_df = model.get_mutations_df()

    # For each mutation, shift_b should equal beta_b - beta_a (wide format)
    for mutation in simple_data.mutations:
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


def test_get_variants_df_after_fit(simple_data):
    """Test get_variants_df returns correct format after fitting."""
    model = multidms.Model(simple_data)
    model.fit(maxiter=5, warmstart=False)

    vars_df = model.get_variants_df()

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
    assert len(vars_df) == len(simple_data.variants_df)


def test_get_variants_df_predictions_are_numeric(simple_data):
    """Test that predictions are valid numeric values."""
    model = multidms.Model(simple_data)
    model.fit(maxiter=5, warmstart=False)

    vars_df = model.get_variants_df()

    # Check no NaN values in predictions
    assert not vars_df["predicted_func_score"].isna().any()

    # Check predictions are finite
    assert np.all(np.isfinite(vars_df["predicted_func_score"]))


def test_get_variants_df_single_condition(single_condition_data):
    """Test get_variants_df with single condition."""
    model = multidms.Model(single_condition_data)
    model.fit(maxiter=5, warmstart=False)

    vars_df = model.get_variants_df()

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

    # Fit with more iterations
    model2 = multidms.Model(simple_data)
    model2.fit(maxiter=20, warmstart=False)
    vars_df2 = model2.get_variants_df()

    # Calculate error
    error2 = np.mean((vars_df2["func_score"] - vars_df2["predicted_func_score"]) ** 2)

    # More iterations should reduce error (or at least not increase it significantly)
    assert error2 <= error1 * 1.1  # Allow 10% tolerance for numerical variation


def test_model_with_regularization(simple_data):
    """Test that models can be fitted with different regularization."""
    # No regularization
    model1 = multidms.Model(simple_data, l2reg=0.0)
    model1.fit(maxiter=10, warmstart=False)
    muts_df1 = model1.get_mutations_df()

    # With L2 regularization
    model2 = multidms.Model(simple_data, l2reg=1.0)
    model2.fit(maxiter=10, warmstart=False)
    muts_df2 = model2.get_mutations_df()

    # Regularized model should have smaller parameter magnitudes (use beta_a)
    beta_mag1 = np.abs(muts_df1["beta_a"]).mean()
    beta_mag2 = np.abs(muts_df2["beta_a"]).mean()

    assert beta_mag2 < beta_mag1


def test_model_deterministic_with_same_params(simple_data):
    """Test that model fitting is deterministic."""
    model1 = multidms.Model(simple_data)
    model1.fit(maxiter=10, warmstart=False)
    muts_df1 = model1.get_mutations_df()

    model2 = multidms.Model(simple_data)
    model2.fit(maxiter=10, warmstart=False)
    muts_df2 = model2.get_mutations_df()

    # Results should be very close (allowing for small numerical differences)
    # Compare beta_a values (wide format)
    assert np.allclose(
        muts_df1["beta_a"].values,
        muts_df2["beta_a"].values,
        rtol=1e-5,
        atol=1e-8,
    )
