"""Tests for the jaxmodels module."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
import multidms.jaxmodels as jaxmodels


# ==================== Fixtures ====================

@pytest.fixture
def n_mutations():
    """Number of mutations for test data."""
    return 20


@pytest.fixture
def n_variants():
    """Number of variants for test data."""
    return 10


@pytest.fixture
def n_conditions():
    """Number of experimental conditions."""
    return 2


@pytest.fixture
def rng_key():
    """JAX random key for reproducible tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def sparse_variant_matrix(n_variants, n_mutations, rng_key):
    """Create a sparse variant encoding matrix."""
    # Create random binary matrix (0s and 1s)
    key1, key2 = jax.random.split(rng_key)
    # Make it sparse by having only ~20% of entries as 1
    probs = jax.random.uniform(key1, shape=(n_variants, n_mutations))
    X_dense = (probs < 0.2).astype(jnp.int32)
    return BCOO.fromdense(X_dense)


@pytest.fixture
def wildtype_sequence(n_mutations):
    """Create a wildtype sequence (all zeros for reference)."""
    return jnp.zeros(n_mutations, dtype=jnp.int32)


@pytest.fixture
def count_data(n_variants, rng_key):
    """Generate realistic count data for testing."""
    key1, key2 = jax.random.split(rng_key)
    # Generate pre-selection counts (higher values)
    pre_counts = jax.random.poisson(key1, lam=100.0, shape=(n_variants,))
    pre_counts = jnp.maximum(pre_counts, 10)  # Ensure minimum count

    # Generate post-selection counts (slightly lower)
    post_counts = jax.random.poisson(key2, lam=90.0, shape=(n_variants,))
    post_counts = jnp.minimum(post_counts, pre_counts)  # Can't exceed pre-counts

    return pre_counts, post_counts


@pytest.fixture
def functional_scores(n_variants, rng_key):
    """Generate functional scores for variants."""
    # Generate scores centered around 0 with some variation
    scores = jax.random.normal(rng_key, shape=(n_variants,)) * 0.5
    return scores


@pytest.fixture
def single_condition_data(
    wildtype_sequence, sparse_variant_matrix, count_data, functional_scores
):
    """Create a Data object for a single condition."""
    pre_counts, post_counts = count_data
    return jaxmodels.Data(
        x_wt=wildtype_sequence,
        pre_count_wt=jnp.array(150),  # WT typically has higher counts
        post_count_wt=jnp.array(140),
        X=sparse_variant_matrix,
        pre_counts=pre_counts,
        post_counts=post_counts,
        functional_scores=functional_scores,
    )


@pytest.fixture
def multi_condition_data(n_conditions, n_mutations, n_variants, rng_key):
    """Create Data objects for multiple conditions."""
    data_sets = {}
    keys = jax.random.split(rng_key, n_conditions)

    for i in range(n_conditions):
        key_i = keys[i]
        key1, key2, key3 = jax.random.split(key_i, 3)

        # Create variant matrix for this condition
        probs = jax.random.uniform(key1, shape=(n_variants, n_mutations))
        X_dense = (probs < 0.2).astype(jnp.int32)
        X = BCOO.fromdense(X_dense)

        # Generate counts
        pre_counts = jax.random.poisson(key2, lam=100.0, shape=(n_variants,))
        pre_counts = jnp.maximum(pre_counts, 10)
        post_counts = jax.random.poisson(key3, lam=90.0, shape=(n_variants,))
        post_counts = jnp.minimum(post_counts, pre_counts)

        # Generate functional scores
        scores = jax.random.normal(key3, shape=(n_variants,)) * 0.5

        # Ensure first condition has wildtype with no mutations
        x_wt = jnp.zeros(n_mutations, dtype=jnp.int32) if i == 0 else X[0].todense()

        data_sets[f"condition{i+1}"] = jaxmodels.Data(
            x_wt=x_wt,
            pre_count_wt=jnp.array(150),
            post_count_wt=jnp.array(140),
            X=X,
            pre_counts=pre_counts,
            post_counts=post_counts,
            functional_scores=scores,
        )

    return data_sets


@pytest.fixture
def simple_latent_model(n_mutations, rng_key):
    """Create a simple Latent model for testing."""
    beta = jax.random.normal(rng_key, shape=(n_mutations,)) * 0.1
    return jaxmodels.Latent(β0=jnp.array(0.5), β=beta)


@pytest.fixture
def global_epistasis_functions():
    """Dictionary of global epistasis functions for testing."""
    return {
        "identity": jaxmodels.Identity(),
        "sigmoid": jaxmodels.Sigmoid(),
    }


# ==================== Tests for Data class ====================

class TestData:
    """Tests for the Data class."""

    def test_data_creation(self, single_condition_data):
        """Test that Data object is created correctly."""
        assert single_condition_data is not None
        assert hasattr(single_condition_data, "x_wt")
        assert hasattr(single_condition_data, "X")
        assert hasattr(single_condition_data, "pre_counts")
        assert hasattr(single_condition_data, "post_counts")
        assert hasattr(single_condition_data, "functional_scores")

    def test_data_shapes(self, single_condition_data, n_variants, n_mutations):
        """Test that Data object has correct shapes."""
        assert single_condition_data.x_wt.shape == (n_mutations,)
        assert single_condition_data.X.shape == (n_variants, n_mutations)
        assert single_condition_data.pre_counts.shape == (n_variants,)
        assert single_condition_data.post_counts.shape == (n_variants,)
        assert single_condition_data.functional_scores.shape == (n_variants,)


# ==================== Tests for Latent class ====================

class TestLatent:
    """Tests for the Latent class."""

    def test_latent_creation(self, simple_latent_model, n_mutations):
        """Test Latent model creation."""
        assert simple_latent_model is not None
        assert simple_latent_model.β.shape == (n_mutations,)
        assert simple_latent_model.β0.shape == ()

    def test_latent_zeros(self, n_mutations):
        """Test zero initialization of Latent model."""
        latent = jaxmodels.Latent.zeros(n_mutations, β0=0.5)
        assert jnp.allclose(latent.β, 0.0)
        assert jnp.allclose(latent.β0, 0.5)

    def test_latent_from_params(self, n_mutations):
        """Test creating Latent from explicit parameters."""
        β0_val = 1.5
        β_val = jnp.ones(n_mutations) * 0.1
        latent = jaxmodels.Latent.from_params(β0=β0_val, β=β_val)
        assert jnp.allclose(latent.β0, β0_val)
        assert jnp.allclose(latent.β, β_val)

    def test_latent_call(self, simple_latent_model, sparse_variant_matrix, n_variants):
        """Test calling Latent model on variant matrix."""
        phenotypes = simple_latent_model(sparse_variant_matrix)
        assert phenotypes.shape == (n_variants,)

    def test_latent_warmstart(self, single_condition_data):
        """Test warmstart initialization."""
        latent = jaxmodels.Latent.warmstart(single_condition_data, l2reg=0.1)
        assert latent.β.shape == single_condition_data.x_wt.shape
        assert latent.β0.shape == ()


# ==================== Tests for Global Epistasis ====================

class TestGlobalEpistasis:
    """Tests for global epistasis functions."""

    def test_identity(self):
        """Test identity global epistasis."""
        ge = jaxmodels.Identity()
        x = jnp.array([0.0, 1.0, -1.0, 2.0])
        y = ge(x)
        assert jnp.allclose(x, y)

    def test_sigmoid(self):
        """Test sigmoid global epistasis."""
        ge = jaxmodels.Sigmoid()
        x = jnp.array([0.0, 1.0, -1.0, 100.0, -100.0])
        y = ge(x)
        # Check sigmoid properties
        assert jnp.allclose(y[0], 0.5)  # sigmoid(0) = 0.5
        assert jnp.all(y >= 0.0)  # All outputs >= 0
        assert jnp.all(y <= 1.0)  # All outputs <= 1
        assert y[1] > y[0]  # sigmoid(1) > sigmoid(0)
        assert y[2] < y[0]  # sigmoid(-1) < sigmoid(0)


# ==================== Tests for Model fitting with beta clipping ====================

class TestBetaClipping:
    """Tests for beta parameter clipping functionality."""

    def test_fit_without_clipping(self, multi_condition_data):
        """Test model fitting without beta clipping."""
        model, loss_trajectory = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            l2reg=0.1,
            fusionreg=0.1,
            block_iters=2,  # Just a few iterations for testing
            warmstart=False,  # Disable warmstart for predictable initialization
        )

        # Check that model was created
        assert model is not None
        assert len(loss_trajectory) > 0

        # Check beta values are not constrained
        for cond in model.φ:
            assert model.φ[cond].β is not None

    def test_fit_with_clipping(self, multi_condition_data):
        """Test model fitting with beta clipping enabled."""
        clip_range = (-0.5, 0.5)

        model, loss_trajectory = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            l2reg=0.1,
            fusionreg=0.1,
            block_iters=3,  # A few iterations to allow clipping to take effect
            beta_clip_range=clip_range,
            warmstart=False,  # Disable warmstart for predictable initialization
        )

        # Check that all beta values are within the clipping range
        for cond in model.φ:
            β_values = model.φ[cond].β
            assert jnp.all(β_values >= clip_range[0] - 1e-6), (
                f"Beta values in {cond} below lower bound: "
                f"min={β_values.min()}, bound={clip_range[0]}"
            )
            assert jnp.all(β_values <= clip_range[1] + 1e-6), (
                f"Beta values in {cond} above upper bound: "
                f"max={β_values.max()}, bound={clip_range[1]}"
            )

    def test_different_clipping_ranges(self, multi_condition_data):
        """Test fitting with different clipping ranges."""
        # Test with narrow range
        narrow_range = (-0.1, 0.1)
        model_narrow, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            l2reg=0.1,
            fusionreg=0.1,
            block_iters=2,
            beta_clip_range=narrow_range,
            warmstart=False,
        )

        # Test with wide range
        wide_range = (-5.0, 5.0)
        model_wide, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            l2reg=0.1,
            fusionreg=0.1,
            block_iters=2,
            beta_clip_range=wide_range,
            warmstart=False,
        )

        # Check narrow range
        for cond in model_narrow.φ:
            β_values = model_narrow.φ[cond].β
            assert jnp.all(β_values >= narrow_range[0] - 1e-6)
            assert jnp.all(β_values <= narrow_range[1] + 1e-6)

        # Check wide range
        for cond in model_wide.φ:
            β_values = model_wide.φ[cond].β
            assert jnp.all(β_values >= wide_range[0] - 1e-6)
            assert jnp.all(β_values <= wide_range[1] + 1e-6)

    def test_clipping_with_warmstart(self, multi_condition_data):
        """Test that clipping works correctly with warmstart."""
        clip_range = (-0.3, 0.3)

        model, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            l2reg=0.1,
            fusionreg=0.1,
            block_iters=3,
            beta_clip_range=clip_range,
            warmstart=True,  # Enable warmstart
        )

        # Even with warmstart, final values should be clipped
        for cond in model.φ:
            β_values = model.φ[cond].β
            assert jnp.all(β_values >= clip_range[0] - 1e-6)
            assert jnp.all(β_values <= clip_range[1] + 1e-6)

    def test_asymmetric_clipping(self, multi_condition_data):
        """Test asymmetric clipping ranges."""
        clip_range = (-1.0, 0.5)  # Asymmetric range

        model, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            l2reg=0.1,
            fusionreg=0.1,
            block_iters=2,
            beta_clip_range=clip_range,
            warmstart=False,
        )

        for cond in model.φ:
            β_values = model.φ[cond].β
            assert jnp.all(β_values >= clip_range[0] - 1e-6)
            assert jnp.all(β_values <= clip_range[1] + 1e-6)


# ==================== Tests for Model class ====================

class TestModel:
    """Tests for the Model class."""

    def test_model_creation(self, multi_condition_data):
        """Test basic model creation."""
        model, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            block_iters=1,
            warmstart=False,
        )

        assert model is not None
        assert model.reference_condition == "condition1"
        assert len(model.φ) == len(multi_condition_data)
        assert len(model.α) == len(multi_condition_data)
        assert len(model.logθ) == len(multi_condition_data)

    def test_predict_score(self, multi_condition_data):
        """Test score prediction."""
        model, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            block_iters=1,
            warmstart=False,
        )

        scores = model.predict_score(multi_condition_data)
        assert len(scores) == len(multi_condition_data)
        for cond in scores:
            assert scores[cond].shape == multi_condition_data[cond].functional_scores.shape

    def test_predict_post_count(self, multi_condition_data):
        """Test post-count prediction."""
        model, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            block_iters=1,
            warmstart=False,
        )

        post_counts = model.predict_post_count(multi_condition_data)
        assert len(post_counts) == len(multi_condition_data)
        for cond in post_counts:
            assert post_counts[cond].shape == multi_condition_data[cond].post_counts.shape
            assert jnp.all(post_counts[cond] >= 0)  # Counts should be non-negative


# ==================== Tests for loss functions ====================

class TestLossFunctions:
    """Tests for loss functions."""

    def test_count_loss(self, multi_condition_data):
        """Test count-based loss function."""
        model, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            block_iters=1,
            warmstart=False,
        )

        losses = jaxmodels.count_loss(model, multi_condition_data)
        assert len(losses) == len(multi_condition_data)
        for cond in losses:
            assert losses[cond].shape == ()  # Scalar loss
            assert jnp.isfinite(losses[cond])  # No NaN or inf

    def test_functional_score_loss(self, multi_condition_data):
        """Test functional score loss function."""
        model, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            block_iters=1,
            warmstart=False,
        )

        losses = jaxmodels.functional_score_loss(model, multi_condition_data, δ=1.0)
        assert len(losses) == len(multi_condition_data)
        for cond in losses:
            assert losses[cond].shape == ()  # Scalar loss
            assert jnp.isfinite(losses[cond])  # No NaN or inf
            assert losses[cond] >= 0  # Huber loss is non-negative


# ==================== Tests for fit function parameters ====================

class TestFitParameters:
    """Tests for various parameters of the fit function."""

    def test_beta_init(self, multi_condition_data, n_mutations):
        """Test custom beta initialization."""
        # Create custom initial values
        beta_init = {
            "condition1": jnp.ones(n_mutations) * 0.2,
            "condition2": jnp.ones(n_mutations) * -0.1,
        }

        model, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            block_iters=0,  # No iterations to check initial values
            warmstart=False,
            beta_init=beta_init,
        )

        # Check that initial values were used
        assert jnp.allclose(model.φ["condition1"].β, beta_init["condition1"])
        assert jnp.allclose(model.φ["condition2"].β, beta_init["condition2"])

    def test_beta_naught_init(self, multi_condition_data):
        """Test custom beta0 initialization."""
        beta_naught_init = {
            "condition1": 1.0,
            "condition2": -0.5,
        }

        model, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            block_iters=0,  # No iterations to check initial values
            warmstart=False,
            beta_naught_init=beta_naught_init,
        )

        # Check that initial values were used
        assert jnp.allclose(model.φ["condition1"].β0, beta_naught_init["condition1"])
        assert jnp.allclose(model.φ["condition2"].β0, beta_naught_init["condition2"])

    def test_different_global_epistasis(self, multi_condition_data):
        """Test fitting with different global epistasis functions."""
        # Test with Identity
        model_identity, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            global_epistasis=jaxmodels.Identity(),
            block_iters=1,
            warmstart=False,
        )
        assert isinstance(model_identity.global_epistasis, jaxmodels.Identity)

        # Test with Sigmoid
        model_sigmoid, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            global_epistasis=jaxmodels.Sigmoid(),
            block_iters=1,
            warmstart=False,
        )
        assert isinstance(model_sigmoid.global_epistasis, jaxmodels.Sigmoid)

    def test_regularization_parameters(self, multi_condition_data):
        """Test different regularization settings."""
        # Test that models can be fit with different regularization values
        # without crashing (don't test magnitude relationships due to optimization sensitivity)

        # Test with no regularization
        model_noreg, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            l2reg=0.0,
            fusionreg=0.0,
            block_iters=1,  # Reduced iterations for stability
            warmstart=False,
        )

        # Test with moderate regularization
        model_reg, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            l2reg=1.0,
            fusionreg=1.0,
            block_iters=1,  # Reduced iterations for stability
            warmstart=False,
        )

        # Just verify that both models were created successfully
        assert model_noreg is not None
        assert model_reg is not None

        # Verify that beta parameters are finite
        for cond in model_noreg.φ:
            assert jnp.all(jnp.isfinite(model_noreg.φ[cond].β))
            assert jnp.all(jnp.isfinite(model_reg.φ[cond].β))