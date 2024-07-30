r"""
jaxmodels
=========

A simple API for global epistasis modeling."""

import multidms

import jax
import jax.numpy as jnp

import equinox as eqx
from jaxtyping import Array, Float, Int
from typing import Any

import jaxopt

import scipy
from sklearn import linear_model

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

    def __init__(self, multidms_data: multidms.Data, condition: str) -> None:
        r"""
        Arguments:
            multidms_data: The data to use. Note the WT must be the first variant
                        in each condition.
            condition: The condition to extract data for.
        """        
        X = multidms_data.arrays["X"][condition]
        not_wt = X.indices[:, 0] != 0  # assumes WT is the first
        sparse_data = X.data[not_wt]
        sparse_idxs = X.indices[not_wt]
        sparse_idxs = sparse_idxs.at[:, 0].add(-1)  # assumes WT is the first
        X = jax.experimental.sparse.BCOO(
            (sparse_data, sparse_idxs), shape=(X.shape[0] - 1, X.shape[1])
        )

        self.x_wt = multidms_data.arrays["X"][condition][0].todense()
        self.pre_count_wt = multidms_data.arrays["pre_count"][condition][
            0
        ]  # assumes WT is the first
        self.post_count_wt = multidms_data.arrays["post_count"][condition][
            0
        ]  # assumes WT is the first
        self.X = X
        self.pre_counts = multidms_data.arrays["pre_count"][condition][
            1:
        ]  # assumes WT is the first
        self.post_counts = multidms_data.arrays["post_count"][condition][
            1:
        ]  # assumes WT is the first
        self.functional_scores = multidms_data.arrays["y"][condition][
            1:
        ]  # assumes WT is the first


class Latent(eqx.Module):
    r"""Model a latent phenotype."""

    β0: Float[Array, ""]
    """Intercept."""
    β: Float[Array, " n_mutations"]
    """Mutation effects."""

    def __init__(
        self,
        data: Data,
        l2reg=0.0,
    ) -> None:
        r"""
        Args:
            data: Data to initialize the model shape for.
            l2reg: L2 regularization strength for warmstart of latent models.
        """
        X = scipy.sparse.csr_array(
            (data.X.data, data.X.indices.T), shape=(data.X.shape[0], len(data.x_wt))
        )
        y = data.functional_scores
        ridge_solver = linear_model.Ridge(alpha=l2reg)
        ridge_solver.fit(X, y, sample_weight=jnp.log(data.pre_counts))
        self.β0 = jnp.asarray(ridge_solver.intercept_)
        self.β = jnp.asarray(ridge_solver.coef_)

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
        return self.β0 + X @ self.β


class Model(eqx.Module):
    r"""Model DMS data.

    Args:
        φ: Latent models for each condition.
        logα: fitness-functional score scaling factors for each condition.
        logθ: Overdispersion parameter for each condition.
        reference_condition: The condition to use as a reference.
    """

    φ: dict[str, Latent]
    """Latent models for each condition."""
    logα: dict[str, Float[Array, ""]]
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
        r"""Predict fitness-functional scores.

        Args:
            data_sets: Data sets for each condition.
        """
        result = {}
        for d in data_sets:
            φ = self.φ[d]
            α = jnp.exp(self.logα[d])
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
            result[d] = jnp.power(
                2,
                f + jnp.log2(m_wt) - jnp.log2(n_wt) + jnp.log2(n_v),
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
            logθ = self.logθ[d]
            θ = jnp.exp(logθ)
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
    precondition: bool = True,
    block_iters: int = 10,
    block_tol: Float = 1e-4,
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
        precondition: Whether to use preconditioning.
        block_iters: Number iterations for block coordinate descent.
        block_tol: Convergence tolerance for block coordinate descent.
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

    opt_φ = jaxopt.ProximalGradient(
        _objective_φ_smooth_value_and_grad_preconditioned,
        prox=_prox,
        value_and_grad=True,
        **ge_kwargs,
    )

    opt_cal = jaxopt.GradientDescent(
        _objective_cal,
        **cal_kwargs,
    )

    if precondition:
        Ps = {
            d: jnp.diag(1 / (1 + data_sets[d].X.sum(axis=0).todense()))
            for d in data_sets
        }
    else:
        Ps = {d: jnp.eye(data_sets[d].X.shape[1]) for d in data_sets}

    hyperparams_prox = dict(fusionreg=fusionreg, Ps=Ps)

    filter_spec = Model(
        φ=True, logα=False, logθ=False, reference_condition=reference_condition
    )

    # initialize
    model = Model(
        φ={d: Latent(data_sets[d], l2reg=l2reg) for d in data_sets},
        logα={d: jnp.array(0.0) for d in data_sets},
        logθ={d: jnp.array(0.0) for d in data_sets},
        reference_condition=reference_condition,
    )

    try:
        for k in range(block_iters):
            print(f"iter {k + 1}:")
            obj_old = _objective_total(
                model, data_sets, l2reg=l2reg, fusionreg=fusionreg
            )
            model_φ, model_cal = eqx.partition(model, filter_spec)
            model_cal, state_cal = opt_cal.run(
                model_cal, model_φ, data_sets, l2reg=l2reg,
                )
            print(
                f"  calibration block: model_error={state_cal.error:.2e}, "
                f"γ={state_cal.stepsize:.1e}, iter={state_cal.iter_num}"
            )
            model_φ, state_φ = opt_φ.run(
                model_φ, hyperparams_prox, model_cal, data_sets, Ps, l2reg=l2reg
            )
            print(
                f"  latent block: model_error={state_φ.error:.2e}, "
                f"γ={state_φ.stepsize:.1e}, iter={state_φ.iter_num}"
            )
            model = eqx.combine(model_φ, model_cal)
            obj = _objective_total(model, data_sets, l2reg=l2reg, fusionreg=fusionreg)
            objective_error = (obj_old - obj) / max(abs(obj_old), abs(obj), 1)
            jnp.abs(obj - obj_old) / max(obj_old, obj)
            print(f"  {objective_error=:.2e}")
            for d in model.φ:
                if d != model.reference_condition:
                    sparsity = (
                        model.φ[d].β - model.φ[model.reference_condition].β == 0
                    ).mean()
                    print(f"  {d} sparsity={sparsity:.2f}")

            if (
                state_φ.error < opt_φ.tol and
                state_cal.error < opt_cal.tol and
                objective_error < block_tol
                ):
                break

    except KeyboardInterrupt:
        pass

    return model


# The following private functions are used internally by the fit function


@jax.jit
def _objective_smooth(model, data_sets, l2reg=0.0):
    n = sum(data_set.X.shape[0] for data_set in data_sets.values())
    loss = sum(model.loss(data_sets).values())
    # L2 regularization for mutation effects wrt their mean
    ridge_β = sum(((model.φ[d].β - model.φ[d].β.mean()) ** 2).sum() for d in model.φ)
    return (loss + l2reg * ridge_β) / n



@jax.jit
def _objective_cal(model_cal, model_φ, data_sets, l2reg=0.0):
    model = eqx.combine(model_cal, model_φ)
    return _objective_smooth(model, data_sets, l2reg=l2reg)


@jax.jit
@jax.value_and_grad
def _objective_φ_smooth_value_and_grad(model_φ, model_cal, data_sets, l2reg=0.0):
    model = eqx.combine(model_φ, model_cal)
    return _objective_smooth(model, data_sets, l2reg=l2reg)


@jax.jit
def _objective_φ_smooth_value_and_grad_preconditioned(
    model_φ, model_cal, data_sets, Ps, l2reg=0.0
):
    value, grad = _objective_φ_smooth_value_and_grad(
        model_φ, model_cal, data_sets, l2reg=l2reg
    )
    grad = _precondition_mul(Ps, grad)
    return value, grad


def _precondition_mul(Ps, x):
    Px = x
    for d in Ps:
        Px = eqx.tree_at(lambda Px_: Px_.φ[d].β, Px, Ps[d] @ Px.φ[d].β)
    return Px


@jax.jit
def _objective_total(model, data_sets, l2reg=0.0, fusionreg=0.0):
    return _objective_smooth(model, data_sets, l2reg) + _shift_lasso(model, fusionreg)


@jax.jit
def _shift_lasso(model, fusionreg=0.0):
    return fusionreg * sum(
        jnp.abs(model.φ[d].β - model.φ[model.reference_condition].β).sum()
        for d in model.φ
        if d != model.reference_condition
    )


@jax.jit
def _prox(x, hyperparameters, scaling=1.0):
    fusionreg = hyperparameters["fusionreg"]
    Ps = hyperparameters["Ps"]
    # fused lasso for β
    β_ref = x.φ[x.reference_condition].β
    for d in x.φ:
        if d != x.reference_condition:
            β = x.φ[d].β
            P = Ps[d]
            β_prox = β_ref + jaxopt.prox.prox_lasso(
                β - β_ref, fusionreg * jnp.diag(P), scaling
            )
            x = eqx.tree_at(lambda x_: x_.φ[d].β, x, β_prox)
    return x
