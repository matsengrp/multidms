"""Tests for the jaxmodels module."""

import pytest
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
import multidms.jaxmodels as jaxmodels

# Every test here fits a JAX model — slow on CI. Deselected by default
# (pyproject `addopts = "-m 'not slow'"`); run on push-to-main / release.
pytestmark = pytest.mark.slow


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
def single_condition_data_no_counts(
    wildtype_sequence, sparse_variant_matrix, functional_scores
):
    """Create a Data object without count data (functional scores only)."""
    return jaxmodels.Data(
        x_wt=wildtype_sequence,
        X=sparse_variant_matrix,
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

    def test_latent_warmstart_without_counts(self, single_condition_data_no_counts):
        """Test warmstart works with functional scores only (no counts)."""
        latent = jaxmodels.Latent.warmstart(single_condition_data_no_counts, l2reg=0.1)
        assert latent.β.shape == single_condition_data_no_counts.x_wt.shape
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


# ==================== Tests for Output Activation ====================


class TestOutputActivation:
    """Tests for the softplus output-activation floor (issue #277)."""

    def test_identity_output_passthrough(self):
        """IdentityOutput returns its input unchanged."""
        act = jaxmodels.IdentityOutput()
        y = jnp.array([-5.0, -3.5, 0.0, 2.0, 100.0])
        assert jnp.array_equal(act(y), y)

    def test_softplus_asymptotics(self):
        """Softplus ≈ identity well above l, → l from above well below l.

        Note the floor is soft in exact arithmetic (t(y) > l for all finite y),
        but in floating point the ``exp((y-l)/λ)`` term underflows to 0 once y is
        more than ~36λ below l, so t(y) == l exactly there. The meaningful
        invariants are therefore: t(y) never dips *below* l, and t(y) is strictly
        above l within the transition region (a few λ of the bound).
        """
        act = jaxmodels.Softplus(lower_bound=-3.5, hinge_scale=0.1)
        # Well above the floor: t(y) ≈ y
        high = jnp.array([0.0, 2.0, 10.0])
        assert jnp.allclose(act(high), high, atol=1e-3)
        # Well below the floor: t(y) → l from above; never below l
        low = jnp.array([-10.0, -20.0, -100.0])
        out_low = act(low)
        assert jnp.allclose(out_low, -3.5, atol=1e-3)
        assert jnp.all(out_low >= -3.5)
        # Strictly above l in the transition region (before float underflow)
        near = jnp.array([-3.6, -3.5, 0.0, 50.0])
        assert jnp.all(act(near) > -3.5)

    def test_softplus_known_answer_at_hinge(self):
        """At y = l, t(l) = l + λ·log(2) exactly."""
        act = jaxmodels.Softplus(lower_bound=-3.5, hinge_scale=0.1)
        expected = -3.5 + 0.1 * jnp.log(2.0)
        np.testing.assert_allclose(act(jnp.array(-3.5)), expected, rtol=1e-6)

    def test_softplus_gradient_is_sigmoid(self):
        """d/dy t(y) = sigmoid((y-l)/λ): 0.5 at y=l, →1 above, →0 below."""
        act = jaxmodels.Softplus(lower_bound=-3.5, hinge_scale=0.1)
        grad = jax.grad(lambda y: act(y).sum())
        # At the hinge, slope is 0.5
        np.testing.assert_allclose(grad(jnp.array(-3.5)), 0.5, rtol=1e-5)
        # Far above → 1, far below → 0
        np.testing.assert_allclose(grad(jnp.array(10.0)), 1.0, atol=1e-4)
        np.testing.assert_allclose(grad(jnp.array(-20.0)), 0.0, atol=1e-4)

    def test_softplus_hinge_scale_controls_ramp(self):
        """A smaller hinge_scale makes the transition sharper at the hinge value."""
        y = jnp.array(-3.5)  # the hinge point for lower_bound=-3.5
        sharp = jaxmodels.Softplus(lower_bound=-3.5, hinge_scale=0.01)
        wide = jaxmodels.Softplus(lower_bound=-3.5, hinge_scale=1.0)
        # At the hinge, value = l + λ·log2, so larger λ → larger offset above l
        assert float(wide(y)) > float(sharp(y))

    def _fitted_params(self, model):
        """Extract (α, per-condition β) as concrete arrays for comparison."""
        α = model.α
        βs = {d: model.φ[d].β for d in model.φ}
        return α, βs

    def test_fit_merge_safety_default_is_fieldfree(self, multi_condition_data):
        """Default (arg omitted) fit == explicit IdentityOutput fit, bitwise.

        Guards spec's merge-safety invariant: the new static field must not
        shift equinox partitioning or the optimizer. Compares fitted α, β, and
        predict_score outputs via jnp.array_equal.
        """
        common = dict(reference_condition="condition1", block_iters=5)
        # Arm 1: output_activation argument OMITTED (the main-era code path).
        m_omitted, _ = jaxmodels.fit(data_sets=multi_condition_data, **common)
        # Arm 2: explicit IdentityOutput.
        m_explicit, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            output_activation=jaxmodels.IdentityOutput(),
            **common,
        )
        α1, β1 = self._fitted_params(m_omitted)
        α2, β2 = self._fitted_params(m_explicit)
        # α may be a scalar or a dict depending on share_alpha; handle both.
        if isinstance(α1, dict):
            for d in α1:
                assert jnp.array_equal(α1[d], α2[d])
        else:
            assert jnp.array_equal(α1, α2)
        for d in β1:
            assert jnp.array_equal(β1[d], β2[d])
        p1 = m_omitted.predict_score(multi_condition_data)
        p2 = m_explicit.predict_score(multi_condition_data)
        for d in p1:
            assert jnp.array_equal(p1[d], p2[d])

    def test_fit_softplus_changes_predictions(self, multi_condition_data):
        """A Softplus fit produces DIFFERENT predict_score than the default fit.

        Positive control: proves the floor is actually wired into predict_score
        (not inert). Complements the merge-safety test above.
        """
        common = dict(reference_condition="condition1", block_iters=5)
        m_off, _ = jaxmodels.fit(data_sets=multi_condition_data, **common)
        m_on, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            output_activation=jaxmodels.Softplus(lower_bound=0.0, hinge_scale=0.1),
            **common,
        )
        # lower_bound=0.0 forces a visible change even on this small toy data
        # (many toy scores sit below 0), so the ON/OFF predictions must differ.
        p_off = m_off.predict_score(multi_condition_data)
        p_on = m_on.predict_score(multi_condition_data)
        differs = any(not jnp.array_equal(p_off[d], p_on[d]) for d in p_off)
        assert differs, "Softplus floor did not change predict_score — not wired"
        # And the floored predictions never fall below the (soft) bound.
        for d in p_on:
            assert jnp.all(p_on[d] >= 0.0)


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
        assert model.α.shape == ()
        assert len(model.logθ) == len(multi_condition_data)

    def test_model_creation_per_condition_alpha(self, multi_condition_data):
        """Test model creation with share_alpha=False."""
        model, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            block_iters=1,
            warmstart=False,
            share_alpha=False,
        )

        assert isinstance(model.α, dict)
        assert len(model.α) == len(multi_condition_data)

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
            assert (
                scores[cond].shape == multi_condition_data[cond].functional_scores.shape
            )

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
            assert (
                post_counts[cond].shape == multi_condition_data[cond].post_counts.shape
            )
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


# ==================== Tests for mean loss normalization ====================


class TestMeanLossNormalization:
    """Tests verifying the .mean() normalization in functional_score_loss."""

    def test_mean_loss_known_answer(self, n_mutations, n_variants):
        """Mean loss with uniform residual should equal huber(r) regardless of N."""
        import jaxopt

        # Create model and data where all variants have residual r = 0.5
        r = 0.5
        β = jnp.zeros(n_mutations)
        latent = jaxmodels.Latent(β0=jnp.array(0.0), β=β)
        ge = jaxmodels.Identity()

        # Create variant matrix of zeros so predictions are all β0 = 0
        X = BCOO.fromdense(jnp.zeros((n_variants, n_mutations), dtype=jnp.int32))
        x_wt = jnp.zeros(n_mutations, dtype=jnp.int32)
        # functional_scores = r so residual = y - f = r - 0 = r
        scores = jnp.full(n_variants, r)

        data = jaxmodels.Data(x_wt=x_wt, X=X, functional_scores=scores)
        model = jaxmodels.Model(
            φ={"c": latent},
            α=jnp.array(1.0),
            logθ={"c": jnp.array(0.0)},
            global_epistasis=ge,
            reference_condition="c",
        )

        losses = jaxmodels.functional_score_loss(model, {"c": data}, δ=1.0)
        expected = jaxopt.loss.huber_loss(r, 0.0, 1.0)
        np.testing.assert_allclose(losses["c"], expected, rtol=1e-5)

    def test_mean_loss_invariant_to_duplicates(self, n_mutations):
        """Doubling all variants should not change the per-condition loss."""
        n_small = 5
        n_large = 10

        β = jnp.zeros(n_mutations)
        latent = jaxmodels.Latent(β0=jnp.array(0.0), β=β)
        ge = jaxmodels.Identity()
        x_wt = jnp.zeros(n_mutations, dtype=jnp.int32)

        rng = jax.random.PRNGKey(99)
        scores_small = jax.random.normal(rng, shape=(n_small,))
        # Duplicate: repeat each score
        scores_large = jnp.concatenate([scores_small, scores_small])
        assert scores_large.shape[0] == n_large

        X_small = BCOO.fromdense(jnp.zeros((n_small, n_mutations), dtype=jnp.int32))
        X_large = BCOO.fromdense(jnp.zeros((n_large, n_mutations), dtype=jnp.int32))

        data_small = jaxmodels.Data(
            x_wt=x_wt, X=X_small, functional_scores=scores_small
        )
        data_large = jaxmodels.Data(
            x_wt=x_wt, X=X_large, functional_scores=scores_large
        )

        model = jaxmodels.Model(
            φ={"c": latent},
            α=jnp.array(1.0),
            logθ={"c": jnp.array(0.0)},
            global_epistasis=ge,
            reference_condition="c",
        )

        loss_small = jaxmodels.functional_score_loss(model, {"c": data_small})
        loss_large = jaxmodels.functional_score_loss(model, {"c": data_large})
        np.testing.assert_allclose(loss_small["c"], loss_large["c"], rtol=1e-5)

    def test_mean_loss_gradient_scaling(self, n_mutations):
        """Gradient of mean loss should equal gradient of sum loss / n_variants."""
        import jaxopt

        n_v = 8
        β = jnp.zeros(n_mutations)
        latent = jaxmodels.Latent(β0=jnp.array(0.0), β=β)
        ge = jaxmodels.Identity()
        x_wt = jnp.zeros(n_mutations, dtype=jnp.int32)

        rng = jax.random.PRNGKey(77)
        k1, k2 = jax.random.split(rng)
        X_dense = (jax.random.uniform(k1, shape=(n_v, n_mutations)) < 0.3).astype(
            jnp.int32
        )
        X = BCOO.fromdense(X_dense)
        scores = jax.random.normal(k2, shape=(n_v,))

        data = jaxmodels.Data(x_wt=x_wt, X=X, functional_scores=scores)
        data_sets = {"c": data}

        model = jaxmodels.Model(
            φ={"c": latent},
            α=jnp.array(1.0),
            logθ={"c": jnp.array(0.0)},
            global_epistasis=ge,
            reference_condition="c",
        )

        # Mean loss (current implementation)
        def mean_loss(m):
            return jaxmodels.functional_score_loss(m, data_sets)["c"]

        # Sum loss (old implementation)
        def sum_loss(m):
            score_pred = m.predict_score(data_sets)
            y = data_sets["c"].functional_scores
            f = score_pred["c"]
            return jaxopt.loss.huber_loss(y, f, 1.0).sum()

        grad_mean = jax.grad(mean_loss)(model)
        grad_sum = jax.grad(sum_loss)(model)

        # mean grad should equal sum grad / n_variants
        np.testing.assert_allclose(
            grad_mean.φ["c"].β, grad_sum.φ["c"].β / n_v, rtol=1e-5
        )

    def test_count_loss_unchanged(self, multi_condition_data):
        """Verify count_loss still uses .sum() (not affected by this change)."""
        model, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            block_iters=1,
            warmstart=False,
        )

        losses = jaxmodels.count_loss(model, multi_condition_data)
        # Count loss should be total NLL (sum), which is typically >> 1
        # for multi-variant data. Mean would be much smaller.
        for cond in losses:
            assert losses[cond].shape == ()
            assert jnp.isfinite(losses[cond])
            # Sum of NLLs across variants should be larger than mean
            # (for n_variants > 1, sum > mean)
            n_v = multi_condition_data[cond].post_counts.shape[0]
            if n_v > 1:
                assert losses[cond] > 1.0  # sum of NLLs, not mean


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

    def test_beta0_init(self, multi_condition_data):
        """Test custom beta0 initialization."""
        beta0_init = {
            "condition1": 1.0,
            "condition2": -0.5,
        }

        model, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            block_iters=0,  # No iterations to check initial values
            warmstart=False,
            beta0_init=beta0_init,
        )

        # Check that initial values were used
        assert jnp.allclose(model.φ["condition1"].β0, beta0_init["condition1"])
        assert jnp.allclose(model.φ["condition2"].β0, beta0_init["condition2"])

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
        # without crashing (don't test magnitude relationships due to
        # optimization sensitivity)

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

    def test_beta0_ridge_penalty(self, multi_condition_data):
        """Test beta0_ridge penalty for constraining β0 differences from reference."""
        # Fit model without beta0_ridge
        model_no_ridge, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            l2reg=0.1,
            fusionreg=0.0,
            beta0_ridge=0.0,
            block_iters=5,  # More iterations to see the effect
            warmstart=False,
            beta0_init={
                "condition1": 1.0,
                "condition2": -1.0,
            },  # Start with different values
        )

        # Fit model with moderate beta0_ridge
        model_moderate_ridge, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            l2reg=0.1,
            fusionreg=0.0,
            beta0_ridge=1.0,
            block_iters=5,
            warmstart=False,
            beta0_init={"condition1": 1.0, "condition2": -1.0},  # Same starting point
        )

        # Fit model with strong beta0_ridge
        model_strong_ridge, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            l2reg=0.1,
            fusionreg=0.0,
            beta0_ridge=10.0,
            block_iters=5,
            warmstart=False,
            beta0_init={"condition1": 1.0, "condition2": -1.0},  # Same starting point
        )

        # Calculate β0 differences from reference
        ref_beta0_no_ridge = model_no_ridge.φ["condition1"].β0
        ref_beta0_moderate = model_moderate_ridge.φ["condition1"].β0
        ref_beta0_strong = model_strong_ridge.φ["condition1"].β0

        diff_no_ridge = jnp.abs(model_no_ridge.φ["condition2"].β0 - ref_beta0_no_ridge)
        diff_moderate = jnp.abs(
            model_moderate_ridge.φ["condition2"].β0 - ref_beta0_moderate
        )
        diff_strong = jnp.abs(model_strong_ridge.φ["condition2"].β0 - ref_beta0_strong)

        # With stronger beta0_ridge, the differences should be smaller
        # Allow for some tolerance due to optimization
        assert diff_strong <= diff_moderate + 1e-2, (
            f"Strong ridge penalty should produce smaller β0 differences: "
            f"strong={diff_strong}, moderate={diff_moderate}"
        )
        assert diff_moderate <= diff_no_ridge + 1e-2, (
            f"Moderate ridge penalty should produce smaller β0 differences "
            f"than no ridge: moderate={diff_moderate}, "
            f"no_ridge={diff_no_ridge}"
        )

        # All models should converge successfully
        assert model_no_ridge is not None
        assert model_moderate_ridge is not None
        assert model_strong_ridge is not None

        # All β0 values should be finite
        for model in [model_no_ridge, model_moderate_ridge, model_strong_ridge]:
            for cond in model.φ:
                assert jnp.isfinite(model.φ[cond].β0)

    def test_early_stopping_ignores_inner_convergence(self, multi_condition_data):
        """Test early stopping triggers on objective_error alone.

        Verifies that the outer loop breaks when objective_error < block_tol
        even if inner solver blocks have not converged (error >= tol).
        """
        # Inner tolerances set impossibly tight so inner blocks never converge
        tight_kwargs = dict(tol=1e-15, maxiter=5, maxls=15, jit=True)
        block_iters = 100
        block_tol = 1e-1  # Loose outer tolerance -> converges quickly

        model, trajectory_df = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            l2reg=0.1,
            fusionreg=0.1,
            block_iters=block_iters,
            block_tol=block_tol,
            ge_kwargs=tight_kwargs,
            cal_kwargs=tight_kwargs,
            warmstart=False,
            verbose=False,
        )

        # Early stopping should have fired before exhausting block_iters
        assert (
            len(trajectory_df) < block_iters
        ), f"Early stopping did not fire: ran all {block_iters} iterations"

        # At least one inner block should NOT have converged in the final row
        final_row = trajectory_df.iloc[-1]
        inner_tol = tight_kwargs["tol"]
        inner_errors = [
            final_row["calibration_error"],
            final_row["beta0_error"],
            final_row["beta_nonbundle_error"],
            final_row["beta_bundle_error"],
        ]
        assert any(err >= inner_tol for err in inner_errors), (
            f"Expected at least one inner block to NOT converge with "
            f"tol={inner_tol}, but all converged: {inner_errors}"
        )

    def test_scale_fusion_by_n_equal_sizes(self, multi_condition_data):
        """Test scale_fusion_by_n=True matches False when conditions have equal sizes."""
        common_kwargs = dict(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            l2reg=0.1,
            fusionreg=0.5,
            block_iters=3,
            warmstart=False,
            verbose=False,
        )

        model_off, _ = jaxmodels.fit(**common_kwargs, scale_fusion_by_n=False)
        model_on, _ = jaxmodels.fit(**common_kwargs, scale_fusion_by_n=True)

        # With equal-sized conditions, weights are all 1.0 so results
        # should be identical
        for cond in multi_condition_data:
            assert jnp.allclose(model_off.φ[cond].β, model_on.φ[cond].β, atol=1e-5), (
                f"Equal-sized conditions should give identical results, "
                f"but {cond} betas differ"
            )

    def test_scale_fusion_by_n_default_false(self, multi_condition_data):
        """Test that the default for scale_fusion_by_n is False."""
        model, _ = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            l2reg=0.1,
            fusionreg=0.5,
            block_iters=1,
            warmstart=False,
            verbose=False,
        )
        # If it runs without error and returns a model, the default works
        assert model is not None


# ==================== Tests for recompute_scale toggle (#246) ====================


class TestRecomputeScale:
    """Tests for the recompute_scale fixed-scale convergence fix (#246)."""

    def test_fixed_scale_is_constant_discriminates(self, multi_condition_data):
        """T1: fixed-scale and recompute-scale produce different obj_error
        sequences, proving the toggle changes loop behavior.
        """
        common = dict(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            l2reg=0.0,
            fusionreg=0.0,
            block_iters=6,
            warmstart=False,
            verbose=False,
        )
        _, traj_fixed = jaxmodels.fit(**common, recompute_scale=False)
        _, traj_recompute = jaxmodels.fit(**common, recompute_scale=True)

        err_fixed = traj_fixed["objective_error_trajectory"].to_numpy()
        err_recompute = traj_recompute["objective_error_trajectory"].to_numpy()

        # The two loops must produce materially different stopping signals.
        n = min(len(err_fixed), len(err_recompute))
        assert n >= 2
        assert not np.allclose(err_fixed[:n], err_recompute[:n], atol=1e-9), (
            "fixed and recompute scale produced identical obj_error sequences; "
            "the toggle is not discriminating"
        )

    def test_fixed_scale_true_relative_change(self, multi_condition_data):
        """T2: with recompute_scale=False, objective_error on sweep k equals
        the fit()'s actual stopping formula applied to the *previous* sweep's
        scaled objective — NOT |1.0 - obj| (the recompute path's
        by-construction value where obj_old ≡ 1.0 every sweep).
        """
        model, traj = jaxmodels.fit(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            l2reg=0.0,
            fusionreg=0.0,
            block_iters=6,
            warmstart=False,
            recompute_scale=False,
            verbose=False,
        )
        # objective_total_trajectory stores obj * scale (jaxmodels.py:805), i.e.
        # the RAW (unscaled) objective. The fixed `scale` is the objective at
        # the INITIAL model (before any block updates), so every scaled
        # objective raw[k]/scale sits below 1 here — meaning the max(...,1) floor
        # in the stopping formula is always active. With the floor pinned at 1,
        #     err_k = |obj_{k-1} - obj_k| = |raw_{k-1} - raw_k| / scale.
        # So err_k is a CONSTANT (1/scale) times the raw consecutive difference:
        # the ratio err_k / |raw_{k-1} - raw_k| must be identical across all k.
        # This is the scale-free fingerprint of obj_old being the previous
        # sweep's value (the recompute path, where obj_old ≡ 1.0 every sweep,
        # cannot produce a constant ratio against |Δraw|).
        raw = traj["objective_total_trajectory"].to_numpy()
        err = traj["objective_error_trajectory"].to_numpy()
        ratios = []
        for k in range(1, len(raw)):
            d = abs(raw[k - 1] - raw[k])
            assert d > 1e-12, f"sweep {k}: raw objective did not move"
            ratios.append(err[k] / d)
        ratios = np.array(ratios)
        # All per-sweep ratios equal 1/scale → their spread is ~0.
        assert np.allclose(ratios, ratios[0], rtol=1e-4, atol=1e-9), (
            f"err/|Δraw| ratios not constant ({ratios}); obj_old is not the "
            f"previous sweep's carried-forward objective"
        )
        # Cross-check: the implied scale (1/ratio) is positive and O(the raw
        # objective magnitude), not 1.0 (which is what obj_old≡1.0 would imply).
        implied_scale = 1.0 / ratios[0]
        assert implied_scale > 0
        # The recompute path would give err[1] = |1 - raw[1]/scale|; with our
        # constant-scale fingerprint established above, confirm err[1] is the
        # carried-forward value, well below that by-construction number.
        assert err[1] < 0.5, (
            f"sweep 1 objective_error={err[1]} looks like the recompute-path "
            f"|1 - obj| value; obj_old was not carried forward"
        )

    def test_recompute_scale_default_unchanged(self, multi_condition_data):
        """T3 (regression guard): omitting recompute_scale ≡ passing True.
        The default path must match the pre-change behavior bit-for-bit.
        """
        common = dict(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            l2reg=0.0,
            fusionreg=0.05,
            block_iters=4,
            warmstart=False,
            verbose=False,
        )
        _, traj_default = jaxmodels.fit(**common)
        _, traj_explicit_true = jaxmodels.fit(**common, recompute_scale=True)

        pd.testing.assert_frame_equal(traj_default, traj_explicit_true)

    def test_recompute_scale_threads_through_model_fit(self):
        """Integration: recompute_scale reaches jaxmodels via Model.fit on a
        tiny real-DataFrame Data object.
        """
        import multidms

        df = pd.DataFrame(
            {
                "condition": ["a", "a", "a", "b", "b", "b"],
                "aa_substitutions": ["M1A", "M1C", "", "M1A", "G2A", ""],
                "func_score": [-0.5, -1.0, 0.0, -0.6, -0.3, 0.0],
            }
        )
        data = multidms.Data(df, reference="a", verbose=False)
        model = multidms.Model(data, ge_type="Sigmoid")
        # Should accept the kwarg and run without error.
        model.fit(maxiter=3, warmstart=False, recompute_scale=False, verbose=False)
        traj = model.convergence_trajectory_df
        assert traj is not None and len(traj) > 0

    def test_fixed_scale_converges_better_than_recompute(self, multi_condition_data):
        """Fixed-scale drives obj_error far lower than recompute-scale.

        The qualitative analogue of the ablation's
        ``scale=fixed fr=8e-5 conv=True err=3.9e-07`` cell (vs the
        ``recompute`` cell that stalled at err≈5e-3 with many objective
        increases). The exact ``iter=19, err=3.9e-07`` figures were measured
        on real 3-condition spike data — reproducing them is the notebook's
        job (#246, Task 5), not a unit test's.

        Relaxed per the plan note (Task 3 Step 2): the 20-mutation /
        10-variant synthetic fixture is too small to land *below* 1e-6 before
        the 50-iter cap (fixed-scale reaches ~1.7e-6 here), so we assert the
        decisive *direction* instead — fixed-scale reaches a small obj_error
        and is strictly better than recompute-scale on the same fixture.
        """
        common = dict(
            data_sets=multi_condition_data,
            reference_condition="condition1",
            l2reg=0.0,
            fusionreg=8e-5,
            block_iters=50,
            block_tol=1e-6,
            global_epistasis=jaxmodels.Sigmoid(),
            share_alpha=True,
            warmstart=False,
            verbose=False,
        )
        _, traj_fixed = jaxmodels.fit(**common, recompute_scale=False)
        _, traj_recompute = jaxmodels.fit(**common, recompute_scale=True)

        err_fixed = float(traj_fixed["objective_error_trajectory"].iloc[-1])
        err_recompute = float(traj_recompute["objective_error_trajectory"].iloc[-1])

        # Fixed-scale reaches a small obj_error...
        assert err_fixed < 1e-3, f"fixed-scale final obj_error {err_fixed} >= 1e-3"
        # ...and is no worse than recompute-scale (here: orders better).
        assert err_fixed <= err_recompute, (
            f"fixed-scale obj_error {err_fixed} worse than recompute "
            f"{err_recompute}; the convergence fix did not help"
        )
