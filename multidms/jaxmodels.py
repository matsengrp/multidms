r"""
jaxmodels
=========

A simple API for global epistasis modeling."""

import multidms

import jax
import jax.numpy as jnp

import equinox as eqx
from jaxtyping import Array, Float, Int
from typing import Any, Self

import jaxopt

import scipy
from sklearn import linear_model

from multidms import data

jax.config.update("jax_enable_x64", True)


class Data(eqx.Module):
    r"""Data for a DMS experiment."""

    x_wt: Int[Array, "  n_mutations"]
    """Binary encoding of the wildtype sequence."""
    pre_count_wt: Int[Array, ""]
    """Wildtype pre-selection count."""
    post_count_wt: Int[Array, ""]
    """Wildtype post-selection count."""
    X: Int[Array, "n_variants n_mutations"]
    """Variant encoding matrix (sparse format)."""
    pre_counts: Int[Array, " n_variants"]
    """Pre-selection counts for each variant."""
    post_counts: Int[Array, " n_variants"]
    """Post-selection counts for each variant."""
    functional_scores: Float[Array, " n_variants"]
    """Functional scores for each variant."""

    @staticmethod
    def from_multidms(multidms_data: multidms.Data, condition: str) -> Self:
        r"""Create data from a multidms data object.

        Arguments:
            multidms_data: The data to use. Note the WT must be the first variant
                        in each condition.
            condition: The condition to extract data for.

        Returns:
            Data object.
        """
        # NOTE: assumes WT is the first variant
        X = multidms_data.arrays["X"][condition]
        not_wt = X.indices[:, 0] != 0
        sparse_data = X.data[not_wt]
        sparse_idxs = X.indices[not_wt]
        sparse_idxs = sparse_idxs.at[:, 0].add(-1)
        X = jax.experimental.sparse.BCOO(
            (sparse_data, sparse_idxs),
            shape=(X.shape[0] - 1, X.shape[1]),
            unique_indices=True
        )
        X = X.sort_indices()

        return Data(
            x_wt=multidms_data.arrays["X"][condition][0].todense(),
            pre_count_wt=multidms_data.arrays["pre_count"][condition][0],
            post_count_wt=multidms_data.arrays["post_count"][condition][0],
            pre_counts=multidms_data.arrays["pre_count"][condition][1:],
            post_counts=multidms_data.arrays["post_count"][condition][1:],
            functional_scores=multidms_data.arrays["y"][condition][1:],
            X=X,
        )


class Latent(eqx.Module):
    r"""Model a latent phenotype."""

    β0: Float[Array, ""] = eqx.field(converter=lambda x: jnp.asarray(x) if not isinstance(x, bool) else x)
    """Intercept."""
    β: Float[Array, " n_mutations"] = eqx.field(converter=lambda x: jnp.asarray(x) if not isinstance(x, bool) else x)
    """Mutation effects."""

    @staticmethod
    def warmstart(
        data: Data,
        l2reg: float = 0.0,
    ) -> Self:
        r"""Warmstart the latent model.

        Args:
            data: Data to initialize the model for.
            l2reg: L2 regularization strength for warmstart.

        Returns:

        """
        X = scipy.sparse.csr_array(
            (data.X.data, data.X.indices.T), shape=(data.X.shape[0], len(data.x_wt))
        )
        y = data.functional_scores
        ridge_solver = linear_model.Ridge(alpha=l2reg)
        ridge_solver.fit(X, y, sample_weight=jnp.log(data.pre_counts))
        return Latent(
            β0=ridge_solver.intercept_,
            β=ridge_solver.coef_,
        )

    @jax.experimental.sparse.sparsify
    def __call__(
        self,
        X: Float[Array, "n_variants n_mutations"],
    ) -> Float[Array, " n_variants"]:
        r"""Evaluate latent phenotype of variant encodings.

        Args:
            X: Variant encoding matrix.

        Returns:
            Latent phenotype for each variant.
        """
        # NOTE: https://github.com/google/jax/discussions/17251
        return self.β0 + X @ self.β


class Model(eqx.Module):
    r"""Model DMS data.

    Args:
        φ: Latent models for each condition.
        α: fitness-functional score scaling factors for each condition.
        logθ: Overdispersion parameter for each condition.
        reference_condition: The condition to use as a reference.
    """

    φ: dict[str, Latent]
    """Latent models for each condition."""
    α: dict[str, Float[Array, ""]]
    """Fitness-functional score scaling factors for each condition."""
    logθ: dict[str, Float[Array, ""]]
    """Overdispersion parameter for each condition."""
    reference_condition: str = eqx.field(static=True)
    """The condition to use as a reference."""

    def g(self, φ_val: Float[Array, " n_variants"]) -> Float[Array, " n_variants"]:
        r"""The global epistasis function.

        Args:
            φ_val: The latent phenotype.

        Returns:
            The fitness score for the given latent phenotype.
        """
        return jax.scipy.special.expit(φ_val)

    def predict_score(
        self,
        data_sets: dict[str, Data],
    ) -> dict[str, Float[Array, " n_variants"]]:
        r"""Predict functional scores, interpreted as :math:`\log_e` enrichment wrt WT.

        Args:
            data_sets: Data sets for each condition.
        """
        result = {}
        for d in data_sets:
            φ = self.φ[d]
            α = self.α[d]
            X = data_sets[d].X
            x_wt = data_sets[d].x_wt
            result[d] = α * (self.g(φ(X)) - self.g(φ(x_wt)))
        return result

    def predict_post_count(
        self,
        data_sets: dict[str, Data],
    ) -> dict[str, Float[Array, " n_variants"]]:
        r"""Predict post-selection counts.

        Args:
            data_sets: Data sets for each condition.
        """
        result = {}
        score_pred = self.predict_score(data_sets)
        for d in data_sets:
            f = score_pred[d]
            n_v = data_sets[d].pre_counts
            n_wt = data_sets[d].pre_count_wt
            m_wt = data_sets[d].post_count_wt
            result[d] = jnp.exp(
                f + jnp.log(m_wt) - jnp.log(n_wt) + jnp.log(n_v),
            )
        return result

    def loss(
        self,
        data_sets: dict[str, Data],
    ) -> dict[str, Float[Array, ""]]:
        r"""Count-based loss.

        Args:
            data_sets: Data sets for each condition.
        """
        post_count_pred = self.predict_post_count(data_sets)
        result = {}
        for d in data_sets:
            k = data_sets[d].post_counts
            μ = post_count_pred[d]
            θ = jnp.exp(self.logθ[d])
            # standard negative binomial parameterization
            σ2 = μ + θ * μ**2
            p = μ / σ2
            n = μ**2 / (σ2 - μ)
            result[d] = -jax.scipy.stats.nbinom.logpmf(k, n, p).sum()
        return result


def fit(
    data_sets: dict[str, Data],
    reference_condition: str,
    l2reg: Float = 0.0,
    fusionreg: Float = 0.0,
    block_iters: int = 10,
    block_tol: Float = 1e-6,
    ge_kwargs: dict[str, Any] = dict(),
    cal_kwargs: dict[str, Any] = dict(),
) -> Model:
    r"""
    Fit a model to data.

    Args:
        data_sets: Data to fit to. Each key is a condition.
        reference_condition: The condition to use as a reference.
        l2reg: L2 regularization strength for mutation effects.
        fusionreg: Fusion (shift lasso) regularization strength.
        block_iters: Number iterations for block coordinate descent.
        block_tol: Tolerance on objective function for block coordinate descent.
        ge_kwargs: Keyword arguments for the global epistasis model optimizer.
        cal_kwargs: Keyword arguments for the experimental calibration
                    parameter optimizer.

    Returns:
        Fitted model.
    """
    if data_sets[reference_condition].x_wt.sum() != 0:
        raise ValueError(
            "WT sequence of the reference condition should have no mutations."
        )

    opt_calibration = jaxopt.GradientDescent(_objective_part, **cal_kwargs)
    opt_β0 = jaxopt.GradientDescent(_objective_part, **ge_kwargs)
    opt_β = jaxopt.ProximalGradient(_objective_block, prox=_prox_block, **ge_kwargs)

    filter_spec_calibration = Model(
        φ=False, α=True, logθ=True, reference_condition=reference_condition
        )
    filter_spec_β0 = Model(
        φ={d: Latent(β0=True, β=False) for d in data_sets},
        α=False,
        logθ=False,
        reference_condition=reference_condition,
        )

    # initialize
    model = Model(
        φ={d: Latent.warmstart(data_sets[d], l2reg=l2reg) for d in data_sets},
        α={d: jnp.array(1.0) for d in data_sets},
        logθ={d: jnp.array(0.0) for d in data_sets},
        reference_condition=reference_condition,
    )

    # numerical rescaling
    scale = abs(_objective_total(model, data_sets, l2reg=l2reg, fusionreg=fusionreg))

    try:
        for k in range(block_iters):
            print(f"iter {k + 1}:")
            obj_old = _objective_total(
                model, data_sets, l2reg=l2reg, fusionreg=fusionreg, scale=scale
            )

            # calibration block
            model_calibration, model_rest = eqx.partition(
                model, filter_spec=filter_spec_calibration
            )
            model_calibration, state_calibration = opt_calibration.run(model_calibration, model_rest, data_sets, scale=scale)
            model = eqx.combine(model_calibration, model_rest)
            print(
                f"  calibration block: error={state_calibration.error:.2e}, γ={state_calibration.stepsize:.1e}, iter={state_calibration.iter_num}"
            )
            for d in model.φ:
                print(
                    f"    {d}: α={model.α[d]:.2f}, θ={jnp.exp(model.logθ[d]):.2f}"
                )

            # β0 block
            model_β0, model_rest = eqx.partition(model, filter_spec=filter_spec_β0)
            model_β0, state_β0 = opt_β0.run(model_β0, model_rest, data_sets, scale=scale)
            model = eqx.combine(model_β0, model_rest)
            print(
                f"  β0 block: error={state_β0.error:.2e}, γ={state_β0.stepsize:.1e}, iter={state_β0.iter_num}"
            )
            for d in model.φ:
                print(f"    {d}: β0={model.φ[d].β0:.2f}")

            # β bundle block
            bundle_idxs = jax.lax.associative_scan(jnp.logical_or, jnp.array([data_sets[d].x_wt.astype(bool) for d in data_sets]))[-1]
            idxs = jnp.where(bundle_idxs)[0]
            β_block = {d: model.φ[d].β[idxs] for d in model.φ}
            hyperparameters_prox = dict(model=model, fusionreg=fusionreg, scale=scale)
            β_block, state_bundle = opt_β.run(β_block, hyperparameters_prox, idxs, model, data_sets, l2reg=l2reg, scale=scale)
            for d in β_block:
                model = eqx.tree_at(lambda model_: model_.φ[d].β, model, model.φ[d].β.at[idxs].set(β_block[d]))
            print(
                f"  β_bundle: error={state_bundle.error:.2e}, γ={state_bundle.stepsize:.1e}, iter={state_bundle.iter_num}"
            )
            # β non-bundle block
            idxs = jnp.where(~bundle_idxs)[0]
            β_block = {d: model.φ[d].β[idxs] for d in model.φ}
            hyperparameters_prox = dict(model=model, fusionreg=fusionreg, scale=scale)
            β_block, state_nonbundle = opt_β.run(β_block, hyperparameters_prox, idxs, model, data_sets, l2reg=l2reg, scale=scale)
            for d in β_block:
                model = eqx.tree_at(lambda model_: model_.φ[d].β, model, model.φ[d].β.at[idxs].set(β_block[d]))
            print(
                f"  β_nonbundle: error={state_nonbundle.error:.2e}, γ={state_nonbundle.stepsize:.1e}, iter={state_nonbundle.iter_num}"
            )
            for d in model.φ:
                if d != model.reference_condition:
                    sparsity = (
                        model.φ[d].β - model.φ[model.reference_condition].β == 0
                    ).mean()
                    print(f"  {d} sparsity={sparsity:.3f}")

            obj = _objective_total(model, data_sets, l2reg=l2reg, fusionreg=fusionreg, scale=scale)
            objective_error = abs(obj_old - obj) / max(abs(obj_old), abs(obj), 1)
            print(f"  {objective_error=:.2e}")

            if state_calibration.error < opt_calibration.tol and state_β0.error < opt_β0.tol and state_bundle.error < opt_β.tol and state_nonbundle.error < opt_β.tol and objective_error < block_tol:
                break

    except KeyboardInterrupt:
        pass

    return model


# The following private functions are used internally by the fit function


@jax.jit
def _objective_part(model_part, model_rest, data_sets, scale=1.0):
    model = eqx.combine(model_part, model_rest)
    loss = sum(model.loss(data_sets).values())
    return loss / scale


@jax.jit
def _objective_block(β_block, idxs, model, data_sets, l2reg=0.0, scale=1.0):
    for d in β_block:
        model = eqx.tree_at(lambda model_: model_.φ[d].β, model, model.φ[d].β.at[idxs].set(β_block[d]))
    loss = sum(model.loss(data_sets).values())
    ridge_β = 0.0
    for d in data_sets:
        β = β_block[d][idxs]
        ridge_β += jnp.sum((β - β.mean()) ** 2)
    return (loss + l2reg * ridge_β) / scale


@jax.jit
def _objective_total(model, data_sets, l2reg=0.0, fusionreg=0.0, scale=1.0):
    loss = sum(model.loss(data_sets).values())
    l2_penalty = 0.0
    fusion_penalty = 0.0
    for d in data_sets:
        β = model.φ[d].β
        l2_penalty += jnp.sum((β - β.mean()) ** 2)
        if d != model.reference_condition:
            fusion_penalty += jnp.abs(model.φ[d].β - model.φ[model.reference_condition].β).sum()
    return (loss + l2reg * l2_penalty + fusionreg * fusion_penalty) / scale


@jax.jit
def _prox_block(β_block, hyperparameters, scaling=1.0):
    model = hyperparameters["model"]
    fusionreg = hyperparameters["fusionreg"]
    scale = hyperparameters["scale"]
    β_ref = β_block[model.reference_condition]
    for d in β_block:
        if d != model.reference_condition:
            β = β_block[d]
            Δ = β - β_ref
            Δ_lasso = jaxopt.prox.prox_lasso(Δ, fusionreg / scale, scaling)
            β_block[d] = β_ref + Δ_lasso
    return β_block
