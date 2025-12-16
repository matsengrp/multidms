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
    """Test fitting with warmstart enabled (requires count data)."""
    # Skip this test for now - warmstart requires count data
    pytest.skip("Warmstart requires count data which test data doesn't have")


def test_model_fit_without_warmstart(simple_data):
    """Test fitting without warmstart."""
    model = multidms.Model(simple_data)
    model.fit(warmstart=False, maxiter=3)

    assert model.params is not None


def test_model_fit_convergence_trajectory(simple_data):
    """Test that convergence trajectory is recorded."""
    model = multidms.Model(simple_data)
    model.fit(maxiter=5, warmstart=False)

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

    # α: scaling factors
    alpha_values = {
        "a": 1.0,
        "b": 1.0,
    }

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
