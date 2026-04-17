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
from typing import Any, Callable
from typing_extensions import Self
import abc

import jaxopt

import pandas as pd
import scipy
from sklearn import linear_model


jax.config.update("jax_enable_x64", True)


class Data(eqx.Module):
    r"""Data for a DMS experiment."""

    x_wt: Int[Array, "n_mutations"]  # noqa: F821, UP037
    """Binary encoding of the wildtype sequence."""
    X: Int[Array, "n_variants n_mutations"]  # noqa: F821, UP037
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
        X = scipy.sparse.csr_array(
            (data.X.data, (data.X.indices[:, 0], data.X.indices[:, 1])),
            shape=(data.X.shape[0], len(data.x_wt)),
        )
        y = data.functional_scores
        ridge_solver = linear_model.Ridge(alpha=l2reg)

        if data.pre_counts is not None:
            ridge_solver.fit(X, y, sample_weight=jnp.log(data.pre_counts))
        else:
            ridge_solver.fit(X, y)
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
    α: Float[Array, ""] | dict[str, Float[Array, ""]]
    """Fitness-functional score scaling factor."""
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
        # Support both shared scalar α and legacy per-condition dict α
        α_is_dict = isinstance(self.α, dict)
        for d in data_sets:
            φ = self.φ[d]
            α = self.α[d] if α_is_dict else self.α
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
        # NOTE: count_loss uses .sum() (total NLL) rather than .mean() because
        # the total log-likelihood is the natural quantity for count models.
        # If condition imbalance becomes an issue for count_loss, consider
        # switching to .mean() here as well.
        result[d] = -jax.scipy.stats.nbinom.logpmf(k, n, p).sum()
    return result


def functional_score_loss(
    model: Model,
    data_sets: dict[str, Data],
    δ: Float = 1.0,
) -> dict[str, Float[Array, ""]]:
    r"""Huber loss on functional scores.

    Returns mean Huber loss per variant for each condition, so that
    conditions contribute equally to the total objective regardless
    of variant count.

    Args:
        model: Model to evaluate.
        data_sets: Data sets for each condition.
        δ: Huber loss parameter.

    Returns:
        Mean loss for each condition.
    """
    score_pred = model.predict_score(data_sets)
    result = {}
    for d in data_sets:
        y = data_sets[d].functional_scores
        f = score_pred[d]
        result[d] = jaxopt.loss.huber_loss(y, f, δ).mean()
    return result


def fit(
    data_sets: dict[str, Data],
    reference_condition: str,
    l2reg: Float = 0.0,
    fusionreg: Float = 0.0,
    beta0_ridge: Float = 0.0,
    scale_fusion_by_n: bool = False,
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
    alpha_init: Float | dict[str, Float] | None = None,
    share_alpha: bool = True,
    beta_clip_range: tuple[Float, Float] | None = None,
    verbose: bool = True,
) -> tuple[Model, pd.DataFrame]:
    r"""
    Fit a model to data.

    Args:
        data_sets: Data to fit to. Each key is a condition.
        reference_condition: The condition to use as a reference.
        l2reg: L2 regularization strength for mutation effects.
        fusionreg: Fusion (shift lasso) regularization strength.
        beta0_ridge: Ridge penalty for β0 differences from reference condition.
        scale_fusion_by_n: If True, weight each condition's fusion penalty by
            n_ref / n_d, reducing shrinkage for data-poor conditions.
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
        alpha_init: Initial α (fitness-functional score scaling) value.
                   Float applies to all conditions; dict maps condition names
                   to per-condition values. If None, uses 1.0.
        share_alpha: If True (default), optimize a single shared α across
                    all conditions. If False, each condition gets its own α.
        beta_clip_range: Optional tuple of (min, max) values for clipping β parameters.
                        If None, no clipping is applied. Example: (-10.0, 10.0).
                        This constrains mutation effect parameters during optimization
                        to prevent extreme values.
        verbose: Whether to print progress information during fitting (default: True).
                If False, suppresses all print output.

    Returns:
        Tuple of (fitted model, convergence trajectory DataFrame).

        The DataFrame has one row per outer iteration with columns:

        - ``iteration``, ``objective_total_trajectory``,
          ``objective_error_trajectory``, ``loss_trajectory``,
          ``loss_per_variant_trajectory``
        - Per-condition loss: ``loss_{condition}`` (mean Huber loss
          per variant for that condition) and ``loss_per_variant_{condition}``
          (identical to ``loss_{condition}`` since the loss is already
          per-variant). Per-condition losses sum to ``loss_trajectory``.
          ``loss_per_variant_trajectory`` is ``loss_trajectory`` divided
          by the number of conditions (average per-condition mean loss).
        - Block-level diagnostics for each optimization block
          (``calibration_error``, ``calibration_stepsize``,
          ``calibration_iter_num``, ``beta0_error``, etc.)
        - Per-condition parameters: ``alpha_{condition}``,
          ``theta_{condition}``, ``beta0_{condition}``,
          ``sparsity_{condition}`` (non-reference only)
    """
    if data_sets[reference_condition].x_wt.sum() != 0:
        raise ValueError(
            "WT sequence of the reference condition should have no mutations."
        )

    # Compute per-condition fusion weights
    n_ref = data_sets[reference_condition].functional_scores.shape[0]
    if scale_fusion_by_n:
        fusion_weights = {
            d: n_ref / data_sets[d].functional_scores.shape[0]
            for d in data_sets
            if d != reference_condition
        }
    else:
        fusion_weights = {d: 1.0 for d in data_sets if d != reference_condition}

    if verbose and scale_fusion_by_n:
        print("Fusion weights (scale_fusion_by_n=True):")
        for d, w in fusion_weights.items():
            n_d = data_sets[d].functional_scores.shape[0]
            print(f"  {d}: {w:.3f} (n_ref={n_ref}, n_d={n_d})")

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
                fusion_penalty += (
                    fusion_weights[d]
                    * jnp.abs(model.φ[d].β - model.φ[model.reference_condition].β).sum()
                )
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
        fw = hyperparameters["fusion_weights"]
        beta_clip_range = hyperparameters.get("beta_clip_range", None)
        # lasso
        β_ref = β_block[model.reference_condition]
        for d in β_block:
            if d != model.reference_condition:
                β = β_block[d]
                Δ = β - β_ref
                Δ_lasso = jaxopt.prox.prox_lasso(Δ, fw[d] * fusionreg / scale, scaling)
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

    α_true = True if share_alpha else {d: True for d in data_sets}
    α_false = False if share_alpha else {d: False for d in data_sets}
    filter_spec_calibration = Model(
        φ=False,
        α=α_true,
        logθ=True,
        reference_condition=reference_condition,
        global_epistasis=global_epistasis,
    )
    filter_spec_β0 = Model(
        φ={d: Latent(β0=True, β=False) for d in data_sets},
        α=α_false,
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

    # Initialize alpha
    if share_alpha:
        if isinstance(alpha_init, dict):
            α_val = jnp.array(list(alpha_init.values())[0])
        else:
            α_val = jnp.array(alpha_init) if alpha_init is not None else jnp.array(1.0)
    else:
        alpha_models = {}
        for d in data_sets:
            if isinstance(alpha_init, dict) and d in alpha_init:
                alpha_models[d] = jnp.array(alpha_init[d])
            elif isinstance(alpha_init, (int, float)):
                alpha_models[d] = jnp.array(alpha_init)
            else:
                alpha_models[d] = jnp.array(1.0)
        α_val = alpha_models

    # initialize model
    model = Model(
        φ=latent_models,
        α=α_val,
        logθ={d: jnp.array(0.0) for d in data_sets},
        reference_condition=reference_condition,
        global_epistasis=global_epistasis,
    )

    # track convergence trajectory
    has_counts = any(data_sets[d].post_counts is not None for d in data_sets)
    trajectory_rows = []

    try:
        for k in range(block_iters):
            if verbose:
                print(f"iter {k + 1}:")

            # Recompute scale each iteration so the proximal lasso
            # threshold (fusionreg / scale) stays calibrated as the
            # model evolves.
            raw_obj = float(
                abs(
                    objective_total(
                        model,
                        data_sets,
                        l2reg=l2reg,
                        fusionreg=fusionreg,
                        scale=1.0,
                        beta0_ridge=beta0_ridge,
                    )
                )
            )
            scale = raw_obj if raw_obj > 1e-30 else 1.0
            obj_old = raw_obj / scale  # 1.0 by construction

            # calibration block
            model_calibration, model_rest = eqx.partition(
                model, filter_spec=filter_spec_calibration
            )
            model_calibration, state_calibration = opt_calibration.run(
                model_calibration, model_rest, data_sets, scale=scale
            )
            model = eqx.combine(model_calibration, model_rest)
            if verbose:
                print(
                    f"  calibration block: error={state_calibration.error:.2e}, "
                    f"stepsize={state_calibration.stepsize:.1e}, "
                    f"iter={state_calibration.iter_num}"
                )
                if share_alpha:
                    print(f"    α={model.α:.2f}")
                for d in model.φ:
                    parts = []
                    if not share_alpha:
                        parts.append(f"α={model.α[d]:.2f}")
                    if has_counts:
                        parts.append(f"θ={jnp.exp(model.logθ[d]):.2f}")
                    if parts:
                        print(f"    {d}: {', '.join(parts)}")

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
            if verbose:
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
                fusion_weights=fusion_weights,
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
            if verbose:
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
                fusion_weights=fusion_weights,
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
            if verbose:
                print(
                    f"  β_bundle: error={state_bundle.error:.2e}, "
                    f"stepsize={state_bundle.stepsize:.1e}, "
                    f"iter={state_bundle.iter_num}"
                )

            # diagnostics
            if verbose:
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
            if verbose:
                print(f"  {obj=:.2e}")
            objective_error = abs(obj_old - obj) / max(abs(obj_old), abs(obj), 1)
            if verbose:
                print(f"  {objective_error=:.2e}")

            # store trajectory data
            per_condition = {}
            if share_alpha:
                per_condition["alpha"] = float(model.α)
            for d in model.φ:
                if not share_alpha:
                    per_condition[f"alpha_{d}"] = float(model.α[d])
                if has_counts:
                    per_condition[f"theta_{d}"] = float(jnp.exp(model.logθ[d]))
                per_condition[f"beta0_{d}"] = float(model.φ[d].β0)
                if d != model.reference_condition:
                    per_condition[f"sparsity_{d}"] = float(
                        (
                            model.φ[d].β - model.φ[model.reference_condition].β == 0
                        ).mean()
                    )

            per_condition_losses = loss_fn(model, data_sets, **loss_kwargs)
            loss_total = float(sum(per_condition_losses.values()))

            trajectory_rows.append(
                {
                    "iteration": k,
                    "objective_total_trajectory": float(obj * scale),
                    "objective_error_trajectory": float(objective_error),
                    "loss_trajectory": loss_total,
                    "loss_per_variant_trajectory": loss_total / len(data_sets),
                    **{f"loss_{d}": float(per_condition_losses[d]) for d in data_sets},
                    **{
                        f"loss_per_variant_{d}": float(per_condition_losses[d])
                        for d in data_sets
                    },
                    "calibration_error": float(state_calibration.error),
                    "calibration_stepsize": float(state_calibration.stepsize),
                    "calibration_iter_num": int(state_calibration.iter_num),
                    "beta0_error": float(state_β0.error),
                    "beta0_stepsize": float(state_β0.stepsize),
                    "beta0_iter_num": int(state_β0.iter_num),
                    "beta_nonbundle_error": float(state_nonbundle.error),
                    "beta_nonbundle_stepsize": float(state_nonbundle.stepsize),
                    "beta_nonbundle_iter_num": int(state_nonbundle.iter_num),
                    "beta_bundle_error": float(state_bundle.error),
                    "beta_bundle_stepsize": float(state_bundle.stepsize),
                    "beta_bundle_iter_num": int(state_bundle.iter_num),
                    **per_condition,
                }
            )

            if objective_error < block_tol:
                if verbose:
                    inner_states = {
                        "calibration": (state_calibration, opt_calibration),
                        "β0": (state_β0, opt_β0),
                        "β_nonbundle": (state_nonbundle, opt_β),
                        "β_bundle": (state_bundle, opt_β),
                    }
                    for name, (state, opt) in inner_states.items():
                        if state.error >= opt.tol:
                            print(
                                f"  warning: {name} block did not converge "
                                f"(error={state.error:.2e}, tol={opt.tol:.2e})"
                            )
                break

    except KeyboardInterrupt:
        pass

    conditions = list(data_sets.keys())
    base_columns = [
        "iteration",
        "objective_total_trajectory",
        "objective_error_trajectory",
        "loss_trajectory",
        "loss_per_variant_trajectory",
        "calibration_error",
        "calibration_stepsize",
        "calibration_iter_num",
        "beta0_error",
        "beta0_stepsize",
        "beta0_iter_num",
        "beta_nonbundle_error",
        "beta_nonbundle_stepsize",
        "beta_nonbundle_iter_num",
        "beta_bundle_error",
        "beta_bundle_stepsize",
        "beta_bundle_iter_num",
    ]
    condition_columns = ["alpha"] if share_alpha else []
    for d in conditions:
        condition_columns.append(f"loss_{d}")
        condition_columns.append(f"loss_per_variant_{d}")
        if not share_alpha:
            condition_columns.append(f"alpha_{d}")
        if has_counts:
            condition_columns.append(f"theta_{d}")
        condition_columns.append(f"beta0_{d}")
        if d != reference_condition:
            condition_columns.append(f"sparsity_{d}")

    if len(trajectory_rows) == 0:
        convergence_trajectory_df = pd.DataFrame(
            columns=base_columns + condition_columns
        )
    else:
        convergence_trajectory_df = pd.DataFrame(trajectory_rows)
    return model, convergence_trajectory_df
