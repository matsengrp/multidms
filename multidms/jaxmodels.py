r"""
jaxmodels
=========

A simple API for global epistasis modeling.
"""

from __future__ import annotations

import multidms

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

import equinox as eqx
from jaxtyping import Array, Float, Int
from typing import Any, Self, Callable
import abc

import jaxopt

import scipy
from sklearn import linear_model


jax.config.update("jax_enable_x64", True)


class Data(eqx.Module):
    r"""Data for a DMS experiment."""

    x_wt: Int[Array, "n_mutations"]  # noqa: F821, UP037
    """Binary encoding of the wildtype sequence."""
    X: Int[Array, "n_variants n_mutations"]
    """Variant encoding matrix (sparse format)."""
    functional_scores: Float[Array, "n_variants"]  # noqa: F821, UP037
    """Functional scores for each variant."""
    pre_count_wt: Int[Array, ""] | None = None
    """Wildtype pre-selection count (optional)."""
    post_count_wt: Int[Array, ""] | None = None
    """Wildtype post-selection count (optional)."""
    pre_counts: Int[Array, " n_variants"] | None = None
    """Pre-selection counts for each variant (optional)."""
    post_counts: Int[Array, " n_variants"] | None = None
    """Post-selection counts for each variant (optional)."""

    @staticmethod
    def from_multidms(
        multidms_data: multidms.Data,
        condition: str,
    ) -> Self:
        r"""Create data from a multidms data object.

        Arguments:
            multidms_data: The data to use. Note the WT must be the first variant
                        in each condition.
            condition: The condition to extract data for.

        Returns:
            Data object with count data if available in the source.
        """
        # NOTE: assumes WT is the first variant!

        # slicing the BCOO array messes up indices, so we need to go to scipy
        X = multidms_data.arrays["X"][condition]
        X = scipy.sparse.csr_array(
            (X.data, (X.indices[:, 0], X.indices[:, 1])), shape=X.shape
        )
        X = X[1:]  # exclude WT
        X = BCOO.from_scipy_sparse(X)

        # Check if count data is available and extract if present
        if "pre_count" in multidms_data.arrays and "post_count" in multidms_data.arrays:
            pre_count_wt = multidms_data.arrays["pre_count"][condition][0]
            post_count_wt = multidms_data.arrays["post_count"][condition][0]
            pre_counts = multidms_data.arrays["pre_count"][condition][1:]
            post_counts = multidms_data.arrays["post_count"][condition][1:]
        else:
            pre_count_wt = None
            post_count_wt = None
            pre_counts = None
            post_counts = None

        return Data(
            x_wt=multidms_data.arrays["X"][condition][0].todense(),
            X=X,
            functional_scores=multidms_data.arrays["y"][condition][1:],
            pre_count_wt=pre_count_wt,
            post_count_wt=post_count_wt,
            pre_counts=pre_counts,
            post_counts=post_counts,
        )


class Latent(eqx.Module):
    r"""Model a latent phenotype."""

    β0: Float[Array, ""] = eqx.field(
        converter=lambda x: jnp.asarray(x) if not isinstance(x, bool) else x
    )
    """Intercept."""
    β: Float[Array, " n_mutations"] = eqx.field(
        converter=lambda x: jnp.asarray(x) if not isinstance(x, bool) else x
    )
    """Mutation effects."""

    @staticmethod
    def from_params(
        β0: Float,
        β: Float[Array, " n_mutations"],
    ) -> Self:
        r"""Create a latent model from explicit parameters.

        Args:
            β0: Intercept value.
            β: Mutation effects array.

        Returns:
            Latent model with specified parameters.
        """
        return Latent(β0=β0, β=β)

    @staticmethod
    def zeros(
        n_mutations: int,
        β0: Float = 0.0,
    ) -> Self:
        r"""Create a zero-initialized latent model with optional intercept.

        Args:
            n_mutations: Number of mutations.
            β0: Intercept value (default: 0.0).

        Returns:
            Latent model with β set to zeros and specified β0.
        """
        return Latent(
            β0=jnp.array(β0),
            β=jnp.zeros(n_mutations),
        )

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
            Latent model initialized with warmstart parameters.
        """
        if data.pre_counts is None:
            raise ValueError(
                "Warmstart requires pre_counts data. Either provide count data "
                "or disable warmstart by setting warmstart=False."
            )
        X = scipy.sparse.csr_array(
            (data.X.data, (data.X.indices[:, 0], data.X.indices[:, 1])),
            shape=(data.X.shape[0], len(data.x_wt)),
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


class GlobalEpistasis(eqx.Module, abc.ABC):
    r"""Global epistasis model."""

    @abc.abstractmethod
    def __call__(
        self, φ_val: Float[Array, " n_variants"]
    ) -> Float[Array, " n_variants"]:
        r"""The global epistasis function.

        Args:
            φ_val: The latent phenotype.

        Returns:
            The fitness score for the given latent phenotype.
        """


class Identity(GlobalEpistasis):
    r"""Identity function."""

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        r"""Return input."""
        return x


class Sigmoid(GlobalEpistasis):
    r"""Sigmoid function."""

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        r"""Return sigmoid of input."""
        return jax.scipy.special.expit(x)


class Model(eqx.Module):
    r"""Model DMS data."""

    φ: dict[str, Latent]
    """Latent models for each condition."""
    α: dict[str, Float[Array, ""]]
    """Fitness-functional score scaling factors for each condition."""
    logθ: dict[str, Float[Array, ""]]
    """Overdispersion parameter for each condition."""
    reference_condition: str = eqx.field(static=True)
    """The condition to use as a reference."""
    global_epistasis: GlobalEpistasis = eqx.field(default=Identity(), static=True)

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
            result[d] = α * (
                self.global_epistasis(φ(X)) - self.global_epistasis(φ(x_wt))
            )
        return result

    def predict_post_count(
        self,
        data_sets: dict[str, Data],
    ) -> dict[str, Float[Array, " n_variants"]]:
        r"""Predict post-selection counts.

        Args:
            data_sets: Data sets for each condition.
        """
        # Check that all required count data is available
        for condition, data in data_sets.items():
            if any(
                count_data is None
                for count_data in [
                    data.pre_counts,
                    data.pre_count_wt,
                    data.post_count_wt,
                ]
            ):
                raise ValueError(
                    f"predict_post_count requires count data for condition "
                    f"'{condition}'. Provide pre_counts, pre_count_wt, post_count_wt."
                )

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


def count_loss(
    model: Model,
    data_sets: dict[str, Data],
) -> dict[str, Float[Array, ""]]:
    r"""Count-based loss.

    Args:
        model: Model to evaluate.
        data_sets: Data sets for each condition.

    Returns:
        Loss for each condition.
    """
    # Check that all required count data is available
    for condition, data in data_sets.items():
        if data.post_counts is None:
            raise ValueError(
                f"count_loss requires post_counts data for condition '{condition}'. "
                "Use functional_score_loss instead if you only have functional scores."
            )

    post_count_pred = model.predict_post_count(data_sets)
    result = {}
    for d in data_sets:
        k = data_sets[d].post_counts
        μ = post_count_pred[d]
        θ = jnp.exp(model.logθ[d])
        # standard negative binomial parameterization
        σ2 = μ + θ * μ**2
        p = μ / σ2
        n = μ**2 / (σ2 - μ)
        result[d] = -jax.scipy.stats.nbinom.logpmf(k, n, p).sum()
    return result


def functional_score_loss(
    model: Model,
    data_sets: dict[str, Data],
    δ: Float = 1.0,
) -> dict[str, Float[Array, ""]]:
    r"""Huber loss on functional scores.

    Args:
        model: Model to evaluate.
        data_sets: Data sets for each condition.
        δ: Huber loss parameter.

    Returns:
        Loss for each condition.
    """
    score_pred = model.predict_score(data_sets)
    result = {}
    for d in data_sets:
        y = data_sets[d].functional_scores
        f = score_pred[d]
        result[d] = jaxopt.loss.huber_loss(y, f, δ).sum()
    return result


def fit(
    data_sets: dict[str, Data],
    reference_condition: str,
    l2reg: Float = 0.0,
    fusionreg: Float = 0.0,
    beta0_ridge: Float = 0.0,
    block_iters: int = 10,
    block_tol: Float = 1e-6,
    ge_kwargs: dict[str, Any] = dict(),
    cal_kwargs: dict[str, Any] = dict(),
    global_epistasis: GlobalEpistasis = Identity(),
    loss_fn: Callable[
        [Model, dict[str, Data]], dict[str, Float[Array, ""]]
    ] = functional_score_loss,
    loss_kwargs: dict[str, Any] = dict(δ=1.0),
    warmstart: bool = True,
    beta0_init: dict[str, Float] | None = None,
    beta_init: dict[str, Float[Array, " n_mutations"]] | None = None,
    alpha_init: dict[str, Float] | None = None,
    beta_clip_range: tuple[Float, Float] | None = None,
) -> tuple[Model, list[float]]:
    r"""
    Fit a model to data.

    Args:
        data_sets: Data to fit to. Each key is a condition.
        reference_condition: The condition to use as a reference.
        l2reg: L2 regularization strength for mutation effects.
        fusionreg: Fusion (shift lasso) regularization strength.
        beta0_ridge: Ridge penalty for β0 differences from reference condition.
        block_iters: Number iterations for block coordinate descent.
        block_tol: Tolerance on objective function for block coordinate descent.
        ge_kwargs: Keyword arguments for the global epistasis model optimizer.
        cal_kwargs: Keyword arguments for the experimental calibration
                    parameter optimizer.
        global_epistasis: Global epistasis model.
        loss_fn: Loss function.
        loss_kwargs: Keyword arguments for the loss function.
        warmstart: Whether to use Ridge regression warmstart (default: True).
                   If True, performs Ridge regression to initialize parameters.
                   The warmstart values will be overridden by any explicit values
                   provided in beta0_init or beta_init.
        beta0_init: Initial β0 (intercept) values for each condition.
                         If None, uses zeros (or warmstart values if warmstart=True).
                         If dict provided, uses those values for specified conditions.
        beta_init: Initial β (mutation effects) values for each condition.
                  If None, uses zeros (or warmstart values if warmstart=True).
                  If dict provided, uses those values for specified conditions.
        alpha_init: Initial α (fitness-functional score scaling) values
                   for each condition. If None, uses 1.0 for all conditions.
                   If dict provided, uses those values for specified conditions.
        beta_clip_range: Optional tuple of (min, max) values for clipping β parameters.
                        If None, no clipping is applied. Example: (-10.0, 10.0).
                        This constrains mutation effect parameters during optimization
                        to prevent extreme values.

    Returns:
        Tuple of (fitted model, loss trajectory).
    """
    if data_sets[reference_condition].x_wt.sum() != 0:
        raise ValueError(
            "WT sequence of the reference condition should have no mutations."
        )

    def _beta_ridge_penalty(model: Model, beta0_ridge=0.0) -> Float:
        r"""Calculate ridge penalty for β0 differences from reference condition."""
        penalty = 0.0
        ref_beta0 = model.φ[model.reference_condition].β0
        for d in model.φ:
            if d != model.reference_condition:
                penalty += (model.φ[d].β0 - ref_beta0) ** 2
        return penalty * beta0_ridge

    @jax.jit
    def objective_part(model_part, model_rest, data_sets, scale=1.0, beta0_ridge=0.0):
        model = eqx.combine(model_part, model_rest)
        loss = sum(loss_fn(model, data_sets, **loss_kwargs).values())
        # Add β0 ridge penalty for non-reference conditions
        # beta0_penalty = 0.0
        # ref_beta0 = model.φ[model.reference_condition].β0
        # for d in model.φ:
        #     if d != model.reference_condition:
        #         beta0_penalty += (model.φ[d].β0 - ref_beta0) ** 2
        return (loss + _beta_ridge_penalty(model, beta0_ridge)) / scale

    @jax.jit
    def objective_block(β_block, idxs, model, data_sets, l2reg=0.0, scale=1.0):
        for d in β_block:
            model = eqx.tree_at(
                lambda model_: model_.φ[d].β,
                model,
                model.φ[d].β.at[idxs].set(β_block[d]),
            )
        loss = sum(loss_fn(model, data_sets, **loss_kwargs).values())
        l2_penalty = 0.0
        for d in data_sets:
            β = β_block[d][idxs]
            l2_penalty += (β**2).sum()
        return (loss + l2reg * l2_penalty) / scale

    @jax.jit
    def objective_total(
        model, data_sets, l2reg=0.0, fusionreg=0.0, scale=1.0, beta0_ridge=0.0
    ):
        loss = sum(loss_fn(model, data_sets, **loss_kwargs).values())
        l2_penalty = 0.0
        fusion_penalty = 0.0
        for d in data_sets:
            β = model.φ[d].β
            l2_penalty += (β**2).sum()
            if d != model.reference_condition:
                fusion_penalty += jnp.abs(
                    model.φ[d].β - model.φ[model.reference_condition].β
                ).sum()
        return (
            loss
            + l2reg * l2_penalty
            + fusionreg * fusion_penalty
            + _beta_ridge_penalty(model, beta0_ridge)
        ) / scale

    @jax.jit
    def prox_block(β_block, hyperparameters, scaling=1.0):
        model = hyperparameters["model"]
        fusionreg = hyperparameters["fusionreg"]
        scale = hyperparameters["scale"]
        beta_clip_range = hyperparameters.get("beta_clip_range", None)
        # lasso
        β_ref = β_block[model.reference_condition]
        for d in β_block:
            if d != model.reference_condition:
                β = β_block[d]
                Δ = β - β_ref
                Δ_lasso = jaxopt.prox.prox_lasso(Δ, fusionreg / scale, scaling)
                β_block[d] = β_ref + Δ_lasso
        # box clipping (if specified)
        if beta_clip_range is not None:
            clip_min, clip_max = beta_clip_range
            for d in β_block:
                β_block[d] = jnp.clip(β_block[d], clip_min, clip_max)
        return β_block

    opt_calibration = jaxopt.GradientDescent(objective_part, **cal_kwargs)
    opt_β0 = jaxopt.GradientDescent(objective_part, **ge_kwargs)
    opt_β = jaxopt.ProximalGradient(objective_block, prox=prox_block, **ge_kwargs)

    filter_spec_calibration = Model(
        φ=False,
        α=True,
        logθ=True,
        reference_condition=reference_condition,
        global_epistasis=global_epistasis,
    )
    filter_spec_β0 = Model(
        φ={d: Latent(β0=True, β=False) for d in data_sets},
        α=False,
        logθ=False,
        reference_condition=reference_condition,
        global_epistasis=global_epistasis,
    )

    # initialize latent models with independent control over each parameter
    latent_models = {}

    for d in data_sets:
        n_mut = len(data_sets[d].x_wt)

        # Step 1: Start with zeros as the base
        β0_val = jnp.array(0.0)
        β_val = jnp.zeros(n_mut)

        # Step 2: If warmstart is True, use Ridge regression to get initial values
        if warmstart:
            warmstart_latent = Latent.warmstart(data_sets[d], l2reg=l2reg)
            β0_val = warmstart_latent.β0
            β_val = warmstart_latent.β

        # Step 3: Override with explicit values if provided
        if beta0_init is not None and d in beta0_init:
            β0_val = jnp.array(beta0_init[d])

        if beta_init is not None and d in beta_init:
            β_val = beta_init[d]

        # Create the Latent model with the final values
        latent_models[d] = Latent(β0=β0_val, β=β_val)

    # Initialize alpha values with control over each parameter
    alpha_models = {}
    for d in data_sets:
        # Default to 1.0
        α_val = jnp.array(1.0)

        # Override with explicit values if provided
        if alpha_init is not None and d in alpha_init:
            α_val = jnp.array(alpha_init[d])

        alpha_models[d] = α_val

    # initialize model
    model = Model(
        φ=latent_models,
        α=alpha_models,
        logθ={d: jnp.array(0.0) for d in data_sets},
        reference_condition=reference_condition,
        global_epistasis=global_epistasis,
    )

    # numerical rescaling
    scale = abs(
        objective_total(
            model, data_sets, l2reg=l2reg, fusionreg=fusionreg, beta0_ridge=beta0_ridge
        )
    )

    # track loss trajectory
    loss_trajectory = []

    try:
        for k in range(block_iters):
            print(f"iter {k + 1}:")
            obj_old = objective_total(
                model,
                data_sets,
                l2reg=l2reg,
                fusionreg=fusionreg,
                scale=scale,
                beta0_ridge=beta0_ridge,
            )

            # calibration block
            model_calibration, model_rest = eqx.partition(
                model, filter_spec=filter_spec_calibration
            )
            model_calibration, state_calibration = opt_calibration.run(
                model_calibration, model_rest, data_sets, scale=scale
            )
            model = eqx.combine(model_calibration, model_rest)
            print(
                f"  calibration block: error={state_calibration.error:.2e}, "
                f"stepsize={state_calibration.stepsize:.1e}, "
                f"iter={state_calibration.iter_num}"
            )
            for d in model.φ:
                print(f"    {d}: α={model.α[d]:.2f}, θ={jnp.exp(model.logθ[d]):.2f}")

            # β0 block
            model_β0, model_rest = eqx.partition(model, filter_spec=filter_spec_β0)
            model_β0, state_β0 = opt_β0.run(
                model_β0,
                model_rest,
                data_sets,
                scale=scale,
                beta0_ridge=beta0_ridge,
            )
            model = eqx.combine(model_β0, model_rest)
            print(
                f"  β0 block: error={state_β0.error:.2e}, "
                f"stepsize={state_β0.stepsize:.1e}, iter={state_β0.iter_num}"
            )
            for d in model.φ:
                print(f"    {d}: β0={model.φ[d].β0:.2f}")

            # determine bundle idxs (mutations that are non-wt in any condition)
            bundle_idxs = jax.lax.associative_scan(
                jnp.logical_or,
                jnp.array([data_sets[d].x_wt.astype(bool) for d in data_sets]),
            )[-1]

            # β non-bundle block
            idxs = jnp.where(~bundle_idxs)[0]
            β_block = {d: model.φ[d].β[idxs] for d in model.φ}
            hyperparameters_prox = dict(
                model=model,
                fusionreg=fusionreg,
                scale=scale,
                beta_clip_range=beta_clip_range,
            )
            β_block, state_nonbundle = opt_β.run(
                β_block,
                hyperparameters_prox,
                idxs,
                model,
                data_sets,
                l2reg=l2reg,
                scale=scale,
            )
            for d in β_block:
                model = eqx.tree_at(
                    lambda model_: model_.φ[d].β,
                    model,
                    model.φ[d].β.at[idxs].set(β_block[d]),
                )
            print(
                f"  β_nonbundle: error={state_nonbundle.error:.2e}, "
                f"stepsize={state_nonbundle.stepsize:.1e}, "
                f"iter={state_nonbundle.iter_num}"
            )

            # β bundle block
            idxs = jnp.where(bundle_idxs)[0]
            β_block = {d: model.φ[d].β[idxs] for d in model.φ}
            hyperparameters_prox = dict(
                model=model,
                fusionreg=fusionreg,
                scale=scale,
                beta_clip_range=beta_clip_range,
            )
            β_block, state_bundle = opt_β.run(
                β_block,
                hyperparameters_prox,
                idxs,
                model,
                data_sets,
                l2reg=l2reg,
                scale=scale,
            )
            for d in β_block:
                model = eqx.tree_at(
                    lambda model_: model_.φ[d].β,
                    model,
                    model.φ[d].β.at[idxs].set(β_block[d]),
                )
            print(
                f"  β_bundle: error={state_bundle.error:.2e}, "
                f"stepsize={state_bundle.stepsize:.1e}, "
                f"iter={state_bundle.iter_num}"
            )

            # diagnostics
            for d in model.φ:
                if d != model.reference_condition:
                    sparsity = (
                        model.φ[d].β - model.φ[model.reference_condition].β == 0
                    ).mean()
                    print(f"  {d} sparsity={sparsity:.1%}")

            obj = objective_total(
                model,
                data_sets,
                l2reg=l2reg,
                fusionreg=fusionreg,
                scale=scale,
                beta0_ridge=beta0_ridge,
            )
            print(f"  {obj=:.2e}")
            objective_error = abs(obj_old - obj) / max(abs(obj_old), abs(obj), 1)
            print(f"  {objective_error=:.2e}")

            # store loss for trajectory
            loss_trajectory.append(float(obj))

            if (
                state_calibration.error < opt_calibration.tol
                and state_β0.error < opt_β0.tol
                and state_bundle.error < opt_β.tol
                and state_nonbundle.error < opt_β.tol
                and objective_error < block_tol
            ):
                break

    except KeyboardInterrupt:
        pass

    return model, loss_trajectory
